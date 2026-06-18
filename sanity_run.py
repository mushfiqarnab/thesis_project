"""
sanity_run.py
=============
Quick sanity check using YOUR real FFHQ + WESAD data.
Uses only 50 images + 50 WESAD windows. Trains 3 epochs each model.
Total time: ~5-10 minutes on CPU.

Place this file in thesis_project/ root folder (same level as src/).

Usage:
    python sanity_run.py --ffhq_dir data/raw/FFHQ --wesad_dir data/raw/WESAD
    python sanity_run.py --ffhq_dir C:/Downloads/FFHQ --wesad_dir C:/Downloads/WESAD
    python sanity_run.py --ffhq_dir data/raw/FFHQ --wesad_dir data/raw/WESAD --n 100 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ---- add src/ to path so project imports work --------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models import MultimodalThreatModel, count_trainable_params

# ---- output directories ------------------------------------------------------
PROC_DIR = PROJECT_ROOT / "data" / "processed" / "sanity"
CSV_DIR  = PROJECT_ROOT / "data" / "csv"
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
REP_DIR  = PROJECT_ROOT / "outputs" / "reports"

for _d in [PROC_DIR / "faces_clean", PROC_DIR / "faces_scar",
           PROC_DIR / "masks", CSV_DIR, CKPT_DIR, REP_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
#  PART 1 - FFHQ helpers  (matches prepare_faces.py logic)
# ==============================================================================

def find_ffhq_images(ffhq_dir: Path, n: int) -> List[Path]:
    """Walk ffhq_dir recursively and collect first N image files."""
    exts = {".png", ".jpg", ".jpeg"}
    found: List[Path] = []
    for root, _, files in os.walk(str(ffhq_dir)):
        for f in sorted(files):
            if Path(f).suffix.lower() in exts:
                found.append(Path(root) / f)
            if len(found) >= n:
                return found
    return found


def draw_scar_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Brow-line scar mask matching prepare_faces.py eyebrow region (y 26-38%)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    x0 = int(w * rng.uniform(0.35, 0.43))
    y0 = int(h * rng.uniform(0.26, 0.31))
    x1 = int(w * rng.uniform(0.54, 0.64))
    y1 = int(h * rng.uniform(0.31, 0.38))
    thickness = max(2, w // 80)
    cv2.line(mask, (x0, y0), (x1, y1), 220, thickness)
    mask = cv2.GaussianBlur(mask, (7, 7), 2)
    return mask


def apply_scar(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend scar texture matching prepare_faces.py apply_scar()."""
    img  = img_bgr.astype(np.float32)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.45
    scar_color = np.array([30, 40, 55], dtype=np.float32)
    img = img * (1 - alpha) + scar_color * alpha
    return np.clip(img, 0, 255).astype(np.uint8)


def process_ffhq_images(img_paths: List[Path], size: int,
                         rng: np.random.Generator) -> List[dict]:
    """
    For each source image: save one clean copy + one scar copy.
    Returns list of row dicts with: image_path, scar, mask_path
    """
    rows: List[dict] = []
    print(f"  Processing {len(img_paths)} FFHQ images -> {len(img_paths)*2} face samples")

    for i, src in enumerate(tqdm(img_paths, desc="  FFHQ faces", leave=False)):
        img_bgr = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]
        name = f"{i:05d}"

        # Clean image (scar=0)
        clean_path = str(PROC_DIR / "faces_clean" / f"{name}.jpg")
        cv2.imencode(".jpg", img_bgr)[1].tofile(clean_path)
        rows.append({"image_path": clean_path, "scar": 0, "mask_path": ""})

        # Scar image (scar=1)
        mask      = draw_scar_mask(h, w, rng)
        scar_img  = apply_scar(img_bgr, mask)
        scar_path = str(PROC_DIR / "faces_scar" / f"{name}.jpg")
        mask_path = str(PROC_DIR / "masks"       / f"{name}.png")
        cv2.imencode(".jpg", scar_img)[1].tofile(scar_path)
        cv2.imencode(".png", mask    )[1].tofile(mask_path)
        rows.append({"image_path": scar_path, "scar": 1, "mask_path": mask_path})

    return rows  # 2 rows per source image


# ==============================================================================
#  PART 2 - WESAD helpers  (matches prepare_wesad.py logic exactly)
# ==============================================================================

def find_wesad_pickles(wesad_dir: Path) -> List[Path]:
    return sorted(wesad_dir.rglob("S*.pkl"))


def extract_wesad_windows(pkl_path: Path, n_windows: int,
                           fs: int = 700, window_sec: int = 30,
                           stride_sec: int = 15) -> List[dict]:
    """
    Extract HRV + GSR windows from one WESAD subject .pkl file.
    Output columns: hrv, gsr, threat
    Matches prepare_wesad.py logic exactly (including its RMSSD formula).
    """
    import neurokit2 as nk

    print(f"  Reading {pkl_path.name} ...", end=" ", flush=True)
    t0 = time.time()

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    chest  = data.get("signal", {}).get("chest", {})
    ecg    = np.asarray(chest.get("ECG", [])).reshape(-1)
    eda    = np.asarray(chest.get("EDA", [])).reshape(-1)
    labels = np.asarray(data.get("label", [])).reshape(-1)

    if len(ecg) == 0 or len(eda) == 0 or len(labels) == 0:
        print("SKIP (missing signals)")
        return []

    n      = min(len(ecg), len(eda), len(labels))
    ecg    = ecg[:n];  eda = eda[:n];  labels = labels[:n]
    win    = window_sec * fs
    stride = stride_sec * fs

    rows: List[dict] = []

    for start in range(0, n - win + 1, stride):
        if len(rows) >= n_windows:
            break
        end     = start + win
        lab_win = labels[start:end].astype(int)
        lab     = int(np.bincount(lab_win).argmax())

        if lab not in (1, 2):
            continue   # only baseline=1 and stress=2, matches prepare_wesad.py

        threat = 1 if lab == 2 else 0

        try:
            _, info = nk.ecg_process(ecg[start:end], sampling_rate=fs)
            rpeaks  = info.get("ECG_R_Peaks", [])
            if len(rpeaks) < 3:
                continue
            rr = np.diff(rpeaks) / fs

            # Matches prepare_wesad.py safe_mean(np.sqrt(np.diff(rr)**2))
            hrv_val = float(np.mean(np.sqrt(np.diff(rr) ** 2))) if len(rr) > 2 else 0.04

        except Exception:
            continue

        gsr_val = float(np.mean(eda[start:end]))

        rows.append({
            "hrv":    round(max(0.005, hrv_val), 6),
            "gsr":    round(max(0.01,  gsr_val), 6),
            "threat": threat,
        })

    elapsed = time.time() - t0
    safe_n   = sum(1 for r in rows if r["threat"] == 0)
    threat_n = sum(1 for r in rows if r["threat"] == 1)
    print(f"{len(rows)} windows in {elapsed:.0f}s  (safe={safe_n}, threat={threat_n})")
    return rows


# ==============================================================================
#  PART 3 - Dataset builder
# ==============================================================================

def build_dataset(face_rows: List[dict], phys_rows: List[dict],
                  n: int, seed: int) -> tuple:
    """
    Pair face images with physiology windows into a balanced dataset.

    CSV schema (matches MultimodalCSVDatasetWithCF requirements):
        image_path, hrv, gsr, scar, threat, mask_path
    - hrv + gsr: detected first by _infer_phys_cols in dataset_fair.py
    - scar: 0 or 1
    - threat: 0 or 1 (label column)
    - mask_path: path to scar mask (empty string when scar=0)
    """
    rng = np.random.default_rng(seed)

    clean_faces = [r for r in face_rows if r["scar"] == 0]
    scar_faces  = [r for r in face_rows if r["scar"] == 1]
    safe_phys   = [r for r in phys_rows if r["threat"] == 0]
    threat_phys = [r for r in phys_rows if r["threat"] == 1]

    per_cell = max(1, n // 4)
    rows: List[dict] = []

    for face_pool, scar_val in [(clean_faces, 0), (scar_faces, 1)]:
        for phys_pool, threat_val in [(safe_phys, 0), (threat_phys, 1)]:
            take = min(per_cell, len(face_pool), len(phys_pool))
            if take == 0:
                print(f"  Warning: no samples for scar={scar_val} threat={threat_val}")
                continue
            fi = rng.choice(len(face_pool), take,
                            replace=(len(face_pool) < take)).tolist()
            pi = rng.choice(len(phys_pool), take,
                            replace=(len(phys_pool) < take)).tolist()
            for f_i, p_i in zip(fi, pi):
                f = face_pool[f_i]
                p = phys_pool[p_i]
                rows.append({
                    "image_path": f["image_path"],
                    "hrv":        p["hrv"],
                    "gsr":        p["gsr"],
                    "scar":       scar_val,
                    "threat":     threat_val,
                    "mask_path":  f["mask_path"],
                })

    if not rows:
        raise RuntimeError("Dataset is empty - check FFHQ and WESAD paths.")

    df = (pd.DataFrame(rows)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
            .head(n))

    total      = len(df)
    csv_path   = CSV_DIR / f"sanity_{total}.csv"
    split_path = CSV_DIR / f"split_seed{seed}_sanity_{total}.json"

    df.to_csv(csv_path, index=False)

    idx = list(range(total))
    rng2 = np.random.default_rng(seed)
    rng2.shuffle(idx)
    sp = int(0.8 * total)

    # Key names MUST be train_idx / val_idx to match make_or_load_split()
    with open(split_path, "w") as f:
        json.dump({"train_idx": idx[:sp], "val_idx": idx[sp:]}, f, indent=2)

    return df, csv_path, split_path


# ==============================================================================
#  PART 4 - Inline training (mirrors train_baseline.py and train_counterfactual.py)
# ==============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed);  np.random.seed(seed)
    torch.manual_seed(seed);  torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def js_divergence(p: torch.Tensor, q: torch.Tensor,
                  eps: float = 1e-8) -> torch.Tensor:
    """JSD matching train_counterfactual.py Innovation-2."""
    p = p.clamp(eps, 1.0);  q = q.clamp(eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p.log() - m.log())).sum(1) +
                  (q * (q.log() - m.log())).sum(1))


@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader,
             device: torch.device) -> float:
    """Matches eval_acc in train_counterfactual.py (uses mask=)."""
    model.eval()
    correct = total = 0
    for b in loader:
        img  = b["img"].to(device)
        phys = b["phys"].to(device)
        y    = b["y"].to(device)
        mask = b["mask"].to(device)   # collate_samples always provides mask
        out  = model(img, phys, mask=mask)
        pred = out.logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    return correct / max(total, 1)


def train_one_model(csv_path: Path, split_path: Path,
                    fusion: str, epochs: int, batch_size: int, seed: int,
                    lambda_cf: float = 0.0,
                    lambda_gate: float = 0.0) -> dict:
    """
    Train one model using the project's actual dataset_fair + models classes.
    fusion='concat' mirrors train_baseline.py
    fusion='cgf'    mirrors train_counterfactual.py
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MultimodalCSVDatasetWithCF has NO zscore_phys param - that's done externally
    ds = MultimodalCSVDatasetWithCF(str(csv_path))

    with open(split_path) as f:
        sp = json.load(f)
    train_idx: list = sp["train_idx"]
    val_idx:   list = sp["val_idx"]

    train_loader = DataLoader(
        Subset(ds, train_idx), batch_size=batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx), batch_size=batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_samples,
    )

    phys_dim = ds[0].phys.numel()
    model = MultimodalThreatModel(
        phys_dim=phys_dim,
        vision_backbone="mobilenet_v3_small",
        fusion=fusion,
        num_classes=2,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss()

    label     = "Baseline (concat)" if fusion == "concat" else "CGF fair model"
    best_acc  = 0.0
    history   = []
    best_ckpt = CKPT_DIR / f"sanity_{'baseline' if fusion=='concat' else 'cgf'}_best.pt"

    print(f"\n  ---- {label} ----  (device={device})")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0;  n_batches = 0

        pbar = tqdm(train_loader,
                    desc=f"  Epoch {epoch}/{epochs}",
                    leave=False, ncols=80)

        for b in pbar:
            # collate_samples keys: img, img_cf, phys, y, scar, has_cf, mask
            img    = b["img"].to(device)
            img_cf = b["img_cf"].to(device)
            phys   = b["phys"].to(device)
            y      = b["y"].to(device)
            has_cf = b["has_cf"].to(device)
            mask   = b["mask"].to(device)

            out       = model(img, phys, mask=mask)
            loss_task = ce(out.logits, y)
            loss      = loss_task

            # CF consistency loss - only for cgf, only where CF image exists
            if lambda_cf > 0 and has_cf.any():
                out_cf = model(img_cf, phys, mask=mask)
                p      = F.softmax(out.logits,    dim=1)
                q      = F.softmax(out_cf.logits, dim=1)
                js     = js_divergence(p, q)
                loss   = loss + lambda_cf * js[has_cf].mean()

            # Gate regularizer - matches train_counterfactual.py exactly
            if (lambda_gate > 0 and
                    out.gate is not None and out.focus is not None):
                focus     = torch.log1p(out.focus.clamp(0.0, 1e3))
                loss_gate = (out.gate * focus).mean()
                loss      = loss + lambda_gate * loss_gate

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item();  n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        acc      = eval_acc(model, val_loader, device)
        marker   = " <- best" if acc > best_acc else ""
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
              f"val_acc={acc:.4f}  ({acc*100:.1f}%){marker}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_ckpt)

        history.append({"epoch": epoch,
                         "loss": round(avg_loss, 4),
                         "acc":  round(acc,      4)})

    return {"fusion":     fusion,
            "label":      label,
            "best_acc":   best_acc,
            "final_loss": history[-1]["loss"] if history else 0.0,
            "ckpt":       str(best_ckpt),
            "history":    history}


# ==============================================================================
#  MAIN
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sanity-run: 50 real FFHQ + WESAD samples, 3 epochs, ~5-10 min")
    ap.add_argument("--ffhq_dir",  required=True,
                    help="Path to FFHQ folder (with subfolders 00000/, 01000/)")
    ap.add_argument("--wesad_dir", required=True,
                    help="Path to WESAD folder (with subfolders S2/, S3/)")
    ap.add_argument("--n",       type=int, default=50,
                    help="Total samples to build (default 50)")
    ap.add_argument("--epochs",  type=int, default=3,
                    help="Training epochs per model (default 3)")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--no_train", action="store_true",
                    help="Build dataset only, skip training")
    args = ap.parse_args()

    print()
    print("=" * 62)
    print("  SANITY RUN - Pipeline Test with Real FFHQ + WESAD")
    print(f"  {args.n} samples  |  {args.epochs} epochs each  |  ~5-10 min")
    print("=" * 62)

    rng = np.random.default_rng(args.seed)

    # -- STEP 1: FFHQ ----------------------------------------------------------
    print(f"\n[1/4]  Finding FFHQ images...")
    ffhq_dir = Path(args.ffhq_dir)
    if not ffhq_dir.exists():
        print(f"  ERROR: not found: {ffhq_dir}");  sys.exit(1)

    n_src = max(1, args.n // 2)   # each source -> 1 clean + 1 scar = 2 rows
    img_paths = find_ffhq_images(ffhq_dir, n_src)
    if not img_paths:
        print(f"  ERROR: no PNG/JPG images inside: {ffhq_dir}");  sys.exit(1)

    print(f"  Images in: {img_paths[0].parent}")
    face_rows = process_ffhq_images(img_paths, size=224, rng=rng)
    print(f"  Done: {len(face_rows)} face rows")

    # -- STEP 2: WESAD ---------------------------------------------------------
    print(f"\n[2/4]  Extracting WESAD windows...")
    wesad_dir = Path(args.wesad_dir)
    if not wesad_dir.exists():
        print(f"  ERROR: not found: {wesad_dir}");  sys.exit(1)

    pkls = find_wesad_pickles(wesad_dir)
    if not pkls:
        print(f"  ERROR: no S*.pkl files in: {wesad_dir}")
        print("  Expected: WESAD/S2/S2.pkl, WESAD/S3/S3.pkl ...");  sys.exit(1)

    print(f"  Found {len(pkls)} subject(s): {[p.stem for p in pkls]}")
    phys_rows: List[dict] = []
    for pkl in pkls:
        if len(phys_rows) >= args.n:
            break
        phys_rows += extract_wesad_windows(pkl, n_windows=args.n)

    if not phys_rows:
        print("  ERROR: no windows extracted.");  sys.exit(1)

    safe_n   = sum(1 for r in phys_rows if r["threat"] == 0)
    threat_n = sum(1 for r in phys_rows if r["threat"] == 1)
    print(f"  Total: {len(phys_rows)} windows  (safe={safe_n}, threat={threat_n})")

    # -- STEP 3: Build CSV -----------------------------------------------------
    print(f"\n[3/4]  Building {args.n}-sample dataset...")
    df, csv_path, split_path = build_dataset(
        face_rows, phys_rows, args.n, args.seed)

    total = len(df)
    sp    = int(0.8 * total)
    print(f"  Samples   : {total}  (train={sp}, val={total-sp})")
    print(f"  CSV       : {csv_path}")
    print(f"  Group counts:")
    for (s, t), cnt in df.groupby(["scar", "threat"]).size().items():
        print(f"    scar={s} threat={t}  ->  {cnt}")

    if args.no_train:
        print("\n  Dataset ready. --no_train set, skipping training.")
        return

    # -- STEP 4: Train ---------------------------------------------------------
    print(f"\n[4/4]  Training {args.epochs} epochs each...")
    print("  Note: First run downloads MobileNetV3 weights (~9 MB). Normal.")

    bs = min(16, max(4, total // 5))
    results = []

    # Baseline: concat, no fairness (mirrors train_baseline.py)
    results.append(train_one_model(
        csv_path, split_path,
        fusion="concat", epochs=args.epochs, batch_size=bs, seed=args.seed,
        lambda_cf=0.0, lambda_gate=0.0,
    ))

    # CGF: with CF loss + gate regularizer (mirrors train_counterfactual.py)
    results.append(train_one_model(
        csv_path, split_path,
        fusion="cgf", epochs=args.epochs, batch_size=bs, seed=args.seed,
        lambda_cf=1.0, lambda_gate=0.05,
    ))

    # -- Summary ---------------------------------------------------------------
    print()
    print("=" * 62)
    print("  SANITY CHECK RESULTS")
    print("=" * 62)
    print(f"\n  {'Model':<22}  {'Best Val Acc':>13}  {'Final Loss':>10}")
    print(f"  {'-'*22}  {'-'*13}  {'-'*10}")
    for r in results:
        print(f"  {r['label']:<22}  {r['best_acc']:>12.1%}  {r['final_loss']:>10.4f}")

    print()
    print("  What this means:")
    print("  -> 50-65% accuracy with 50 samples is NORMAL. Not enough data yet.")
    print("  -> Loss decreasing each epoch = model IS learning.")
    print("  -> No crashes = pipeline works end-to-end. You're ready.")
    print()

    rep_path = REP_DIR / "sanity_run_report.json"
    with open(rep_path, "w") as f:
        json.dump({"n": total, "epochs": args.epochs,
                   "results": results}, f, indent=2)
    print(f"  Report saved: {rep_path}")
    print()
    print("  NEXT: run full training with all your data:")
    print(f"    python src/train_cgf_fair.py \\")
    print(f"      --csv data/csv/multimodal_10k_unbiased.csv \\")
    print(f"      --fusion cgf --epochs 30 --batch_size 32 \\")
    print(f"      --lambda_cf 1.0 --lambda_gate 0.05 \\")
    print(f"      --lambda_dp 0.3 --lambda_eo 0.3 \\")
    print(f"      --zscore_phys --balance_groups --seed 42")
    print("=" * 62)


if __name__ == "__main__":
    main()