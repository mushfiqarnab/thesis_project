from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _list_images(root: Path) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def _to_rel_posix(p: Path) -> str:
    """Write paths in CSV as project-relative POSIX strings when possible."""
    try:
        rp = p.resolve().relative_to(PROJECT_ROOT.resolve())
        return rp.as_posix()
    except Exception:
        return p.resolve().as_posix()


def _load_wesad_windows(wesad_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(wesad_csv)

    if "threat" not in df.columns:
        raise ValueError("wesad_windows.csv must contain a 'threat' column (0/1).")

    # Pick physiology columns from your file
    needed = ["hrv_rmssd", "gsr_mean", "threat"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"wesad_windows.csv missing required column: {c}")

    # Coerce numeric
    df["hrv_rmssd"] = pd.to_numeric(df["hrv_rmssd"], errors="coerce")
    df["gsr_mean"] = pd.to_numeric(df["gsr_mean"], errors="coerce")
    df["threat"] = pd.to_numeric(df["threat"], errors="coerce")

    df = df.dropna(subset=["hrv_rmssd", "gsr_mean", "threat"]).reset_index(drop=True)
    df["threat"] = df["threat"].astype(int).clip(0, 1)

    # Map to your pipeline’s 2-feature schema: hrv, gsr
    out = pd.DataFrame({
        "hrv": df["hrv_rmssd"].astype(np.float32),
        "gsr": df["gsr_mean"].astype(np.float32),
        "threat": df["threat"].astype(int),
    })
    return out


def _zscore_inplace(df: pd.DataFrame, cols: List[str], eps: float = 1e-8) -> None:
    for c in cols:
        mu = float(df[c].mean())
        sd = float(df[c].std()) + eps
        df[c] = (df[c] - mu) / sd


def _make_scar_mask(size: int, rng: np.random.Generator) -> Image.Image:
    """
    Synthetic “scar region” mask: a thin band on left/right cheek area.
    Output: L mode mask with values {0,255}.
    """
    W = H = size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    # Choose cheek side
    side = rng.choice(["left", "right"])
    cx = int(W * (0.35 if side == "left" else 0.65))
    cy = int(H * rng.uniform(0.45, 0.60))

    # Band dimensions
    band_w = int(W * rng.uniform(0.12, 0.22))
    band_h = int(H * rng.uniform(0.015, 0.035))

    x0 = max(0, cx - band_w // 2)
    y0 = max(0, cy - band_h // 2)
    x1 = min(W - 1, x0 + band_w)
    y1 = min(H - 1, y0 + band_h)

    draw.rectangle([x0, y0, x1, y1], fill=255)

    # Slight soften then binarize back to clean edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.2))
    mask = mask.point(lambda p: 255 if p > 40 else 0)
    return mask


def _apply_visible_scar(img: Image.Image, mask: Image.Image, rng: np.random.Generator) -> Image.Image:
    """
    Make scar visually present (so 'scar' isn't just metadata):
    blend a brown/red tint within mask + add a darker line.
    """
    img = img.convert("RGB")
    W, H = img.size

    # Tint overlay
    tint = Image.new("RGB", (W, H), (120, 70, 60))
    scarred = Image.composite(tint, img, mask)

    # Add a darker line roughly across the mask
    line = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(line)

    # Find mask bbox approximately
    # (fast approximate: choose around center)
    x0 = int(W * rng.uniform(0.25, 0.45))
    x1 = int(W * rng.uniform(0.55, 0.75))
    y = int(H * rng.uniform(0.45, 0.60))

    d.line([(x0, y), (x1, y)], fill=(70, 40, 35), width=max(1, int(H * 0.006)))
    line = line.filter(ImageFilter.GaussianBlur(radius=0.6))

    # Blend the line only inside mask
    scarred2 = Image.composite(line, scarred, mask)
    return scarred2


def _balanced_counts(n: int) -> List[Tuple[int, int, int]]:
    """
    Return list of (scar, threat, count) for a 4-way balanced dataset.
    """
    groups = [(0, 0), (0, 1), (1, 0), (1, 1)]
    base = n // 4
    rem = n % 4
    counts = [base + (i < rem) for i in range(4)]
    return [(groups[i][0], groups[i][1], counts[i]) for i in range(4)]


def main():
    ap = argparse.ArgumentParser("Create a real 10k multimodal CSV by pairing WESAD physiology windows with face images.")
    ap.add_argument("--wesad_csv", type=str, required=True)
    ap.add_argument("--faces_dir", type=str, action="append", required=True,
                    help="Can be passed multiple times. Example: --faces_dir data/raw/FFHQ --faces_dir data/raw/img_align_celeba")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_img_dir", type=str, required=True)
    ap.add_argument("--out_mask_dir", type=str, required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--zscore_phys", action="store_true", help="Z-score physiology (recommended).")
    ap.add_argument("--visible_scar", action="store_true", help="Actually draw a scar so scar=1 images look different.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    wesad_csv = (PROJECT_ROOT / args.wesad_csv).resolve() if not Path(args.wesad_csv).is_absolute() else Path(args.wesad_csv)
    out_csv = (PROJECT_ROOT / args.out_csv).resolve() if not Path(args.out_csv).is_absolute() else Path(args.out_csv)

    out_img_dir = (PROJECT_ROOT / args.out_img_dir).resolve() if not Path(args.out_img_dir).is_absolute() else Path(args.out_img_dir)
    out_mask_dir = (PROJECT_ROOT / args.out_mask_dir).resolve() if not Path(args.out_mask_dir).is_absolute() else Path(args.out_mask_dir)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = _load_wesad_windows(wesad_csv)

    if args.zscore_phys:
        _zscore_inplace(df, ["hrv", "gsr"])

    # Face images pool
    face_paths: List[Path] = []
    for d in args.faces_dir:
        dp = (PROJECT_ROOT / d).resolve() if not Path(d).is_absolute() else Path(d)
        face_paths.extend(_list_images(dp))

    if len(face_paths) == 0:
        raise FileNotFoundError("No face images found. Check --faces_dir paths and ensure they contain images.")

    rng.shuffle(face_paths)

    # Split WESAD rows by threat for balanced sampling
    idx_t0 = df.index[df["threat"] == 0].to_numpy()
    idx_t1 = df.index[df["threat"] == 1].to_numpy()
    if len(idx_t0) == 0 or len(idx_t1) == 0:
        raise ValueError("WESAD windows do not contain both threat classes (0 and 1).")

    plan = _balanced_counts(int(args.n))

    # Prepare CSV writer (your exact schema)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "hrv", "gsr", "scar", "threat", "mask_path"])

        img_i = 0
        written = 0
        counts_check = {(s, t): 0 for (s, t, _) in plan}

        for scar, threat, count in plan:
            src_idx = idx_t1 if threat == 1 else idx_t0
            pick_rows = rng.choice(src_idx, size=count, replace=True)

            for ridx in pick_rows:
                # choose face
                face_src = face_paths[img_i % len(face_paths)]
                img_i += 1

                # load + resize (this keeps disk small and training consistent)
                try:
                    img = Image.open(face_src).convert("RGB").resize((args.image_size, args.image_size), resample=Image.BICUBIC)
                except Exception:
                    # fallback: skip corrupted image
                    continue

                mask_path_str = ""
                if scar == 1:
                    mask = _make_scar_mask(args.image_size, rng)
                    if args.visible_scar:
                        img = _apply_visible_scar(img, mask, rng)

                    mask_name = f"mask_{written:06d}.png"
                    mask_out = out_mask_dir / mask_name
                    mask.save(mask_out)
                    mask_path_str = _to_rel_posix(mask_out)

                img_name = f"img_{written:06d}.jpg"
                img_out = out_img_dir / img_name
                img.save(img_out, quality=92)

                r = df.loc[int(ridx)]
                w.writerow([
                    _to_rel_posix(img_out),
                    float(r["hrv"]),
                    float(r["gsr"]),
                    int(scar),
                    int(threat),
                    mask_path_str
                ])

                counts_check[(scar, threat)] += 1
                written += 1

                if written >= args.n:
                    break
            if written >= args.n:
                break

    print(f"Saved {written} rows -> {out_csv}")
    print("Counts by (scar, threat):")
    for (scar, threat), c in sorted(counts_check.items()):
        print(f"  scar={scar} threat={threat}: {c}")


if __name__ == "__main__":
    main()
