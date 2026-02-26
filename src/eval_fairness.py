from __future__ import annotations

from pathlib import Path
import argparse
import json
from types import SimpleNamespace
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.models as tvm

from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models import MultimodalThreatModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "csv" / "multimodal.csv"
LEGACY_DEFAULT_SPLIT = PROJECT_ROOT / "data" / "csv" / "split_seed42.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate accuracy + fairness metrics on validation split.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")

    p.add_argument(
        "--fusion",
        type=str,
        default="concat",
        choices=["concat", "cgf"],
        help="Fusion type (used for current models; ignored for legacy ckpt auto-detect).",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="mobilenet_v3_small",
        choices=["mobilenet_v3_small", "vit_b_16"],
        help="Vision backbone (used for current models; ignored for legacy ckpt auto-detect).",
    )

    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="CSV path (e.g., multimodal.csv, multimodal_10k.csv)")
    # Permanent fix: split can be omitted; we auto-resolve based on csv stem + seed
    p.add_argument("--split", type=str, default="", help="Optional split json path. If omitted, auto-resolves.")
    p.add_argument("--seed", type=int, default=42, help="Seed used to name split files (default: 42).")

    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on P(threat=1)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out", type=str, default="", help="Optional output json path")

    # ✅ NEW: match training preprocessing
    p.add_argument(
        "--zscore_phys",
        action="store_true",
        help="Apply physiology z-scoring using TRAIN split only (requires train_idx in split json).",
    )
    return p.parse_args()


def resolve_split_path(csv_path: Path, split_arg: str, seed: int) -> Path:
    """
    Permanent fix:
    - If --split provided, use it.
    - Else prefer: split_seed{seed}_{csv_stem}.json (created by train_cgf_fair.py)
    - Else fallback: split_seed{seed}.json (legacy)
    """
    if split_arg and str(split_arg).strip():
        return Path(split_arg)

    preferred = csv_path.parent / f"split_seed{seed}_{csv_path.stem}.json"
    if preferred.exists():
        return preferred

    legacy = csv_path.parent / f"split_seed{seed}.json"
    if legacy.exists():
        return legacy

    if LEGACY_DEFAULT_SPLIT.exists() and seed == 42:
        return LEGACY_DEFAULT_SPLIT

    raise FileNotFoundError(
        "Split not found.\n"
        f"Tried:\n"
        f"  1) {preferred}\n"
        f"  2) {legacy}\n"
        f"Tip: run training first (train_cgf_fair.py) or pass --split explicitly."
    )


def _clean_indices(idx_list: List[int], n: int) -> List[int]:
    out = []
    for i in idx_list:
        ii = int(i)
        if 0 <= ii < n:
            out.append(ii)
    return out


def compute_phys_zscore_from_train(ds: MultimodalCSVDatasetWithCF, train_idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mu/sigma using TRAIN indices only (no leakage).
    ds must expose:
      - ds.df (pandas DataFrame)
      - ds.phys_cols (list of physiology feature column names)
    """
    if not hasattr(ds, "df") or not hasattr(ds, "phys_cols"):
        raise AttributeError("Dataset must expose df and phys_cols for z-scoring (MultimodalCSVDatasetWithCF).")

    if len(train_idx) == 0:
        raise ValueError("train_idx is empty; cannot compute z-score parameters.")

    X = ds.df.iloc[train_idx][ds.phys_cols].to_numpy(dtype=np.float32, copy=True)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return mu, sigma


def dp_gap_signed(yhat: np.ndarray, scar: np.ndarray) -> float:
    p1 = yhat[scar == 1].mean() if (scar == 1).any() else 0.0
    p0 = yhat[scar == 0].mean() if (scar == 0).any() else 0.0
    return float(p1 - p0)


def eo_gaps(yhat: np.ndarray, y: np.ndarray, scar: np.ndarray) -> dict:
    def rates(g: int) -> Tuple[float, float]:
        idx = (scar == g)
        if not idx.any():
            return 0.0, 0.0
        yy = y[idx]
        yh = yhat[idx]
        tp = ((yh == 1) & (yy == 1)).sum()
        fn = ((yh == 0) & (yy == 1)).sum()
        fp = ((yh == 1) & (yy == 0)).sum()
        tn = ((yh == 0) & (yy == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        return float(tpr), float(fpr)

    tpr1, fpr1 = rates(1)
    tpr0, fpr0 = rates(0)
    return {
        "tpr1": tpr1,
        "tpr0": tpr0,
        "fpr1": fpr1,
        "fpr0": fpr0,
        "tpr_gap": float(tpr1 - tpr0),
        "fpr_gap": float(fpr1 - fpr0),
        "eo_max_gap": float(max(abs(tpr1 - tpr0), abs(fpr1 - fpr0))),
    }


def f1_score_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    if tp == 0.0:
        return 0.0
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    denom = precision + recall
    return float(2.0 * precision * recall / denom) if denom > 0 else 0.0


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    tpr = tp / max(tp + fn, 1.0)
    tnr = tn / max(tn + fp, 1.0)
    return float(0.5 * (tpr + tnr))


def auc_roc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Pure-numpy ROC AUC (binary). Works without sklearn.
    Compatible with numpy versions where trapz may be removed (uses trapezoid).
    """
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)

    order = np.argsort(-y_score)
    y_true = y_true[order]

    P = int(y_true.sum())
    N = int(len(y_true) - P)
    if P == 0 or N == 0:
        return 0.5

    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)

    tpr = np.concatenate(([0.0], tps / P, [1.0]))
    fpr = np.concatenate(([0.0], fps / N, [1.0]))

    trap = getattr(np, "trapezoid", None)
    if trap is None:
        return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    return float(trap(tpr, fpr))


def load_raw_state_dict(ckpt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict.")

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        cleaned[k] = v
    return cleaned


def is_legacy_checkpoint(state: Dict[str, torch.Tensor]) -> bool:
    keys = list(state.keys())
    has_vit = any(k.startswith("vit.") for k in keys)
    has_phys_old = any(k.startswith("phys_net.") for k in keys)
    has_cls_old = any(k.startswith("classifier.") for k in keys)
    return has_vit and (has_phys_old or has_cls_old)


class LegacyViTConcatModel(nn.Module):
    """
    Matches old checkpoint family:
      vit.* + phys_net.* + classifier.*

    phys_net: Linear(phys_dim->32)->ReLU->Linear(32->32)->ReLU
    classifier: Linear(768+32->128)->ReLU->Dropout->Linear(128->2)
    """

    def __init__(self, phys_in_dim: int = 2, phys_emb: int = 32, num_classes: int = 2):
        super().__init__()
        try:
            vit = tvm.vit_b_16(weights=None)
        except TypeError:
            vit = tvm.vit_b_16(pretrained=False)

        if hasattr(vit, "heads"):
            vit.heads = nn.Identity()
        elif hasattr(vit, "head"):
            vit.head = nn.Identity()
        else:
            raise AttributeError("Unexpected ViT model structure: cannot remove head.")

        self.vit = vit

        self.phys_net = nn.Sequential(
            nn.Linear(phys_in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, phys_emb),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(768 + phys_emb, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, img: torch.Tensor, phys: torch.Tensor, mask: Optional[torch.Tensor] = None):
        v = self.vit(img)
        p = self.phys_net(phys)
        x = torch.cat([v, p], dim=1)
        logits = self.classifier(x)
        return SimpleNamespace(logits=logits, gate=None, focus=None)


@torch.no_grad()
def main():
    args = parse_args()

    ckpt_path = Path(args.ckpt)
    csv_path = Path(args.csv)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    split_path = resolve_split_path(csv_path, args.split, args.seed)
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found: {split_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MultimodalCSVDatasetWithCF(str(csv_path))
    split = json.loads(split_path.read_text(encoding="utf-8"))

    if "val_idx" not in split:
        raise KeyError(f"Split file missing 'val_idx': {split_path}")

    # If z-scoring requested, split MUST include train_idx (no leakage)
    mu_t = sigma_t = None
    if args.zscore_phys:
        if "train_idx" not in split:
            raise KeyError(
                f"--zscore_phys requires 'train_idx' in split json, but it's missing.\n"
                f"Split: {split_path}\n"
                f"Fix: use the split produced by train_cgf_fair.py (split_seed{args.seed}_{csv_path.stem}.json)."
            )
        train_idx = _clean_indices(split["train_idx"], len(ds))
        if len(train_idx) == 0:
            raise ValueError("train_idx is empty after filtering invalid indices; cannot z-score.")
        mu, sigma = compute_phys_zscore_from_train(ds, train_idx)
        mu_t = torch.tensor(mu, device=device, dtype=torch.float32).unsqueeze(0)
        sigma_t = torch.tensor(sigma, device=device, dtype=torch.float32).unsqueeze(0)

    # extra safety: keep only valid indices
    val_idx = _clean_indices(split["val_idx"], len(ds))
    if len(val_idx) == 0:
        raise ValueError(f"Validation split is empty after filtering invalid indices. Split file: {split_path}")

    val_ds = Subset(ds, val_idx)

    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_samples,
    )

    raw_state = load_raw_state_dict(ckpt_path, device)
    legacy = is_legacy_checkpoint(raw_state)

    phys_dim = ds[0].phys.numel()
    if legacy:
        model = LegacyViTConcatModel(phys_in_dim=phys_dim, phys_emb=32, num_classes=2).to(device)
        model.load_state_dict(raw_state, strict=True)
        model_family = "legacy_vit_concat_phys32"
        fusion_used = "concat"
        backbone_used = "vit_b_16"
    else:
        model = MultimodalThreatModel(
            phys_dim=phys_dim,
            vision_backbone=args.backbone,
            fusion=args.fusion,
            num_classes=2,
        ).to(device)
        model.load_state_dict(raw_state, strict=True)
        model_family = "current_multimodal"
        fusion_used = args.fusion
        backbone_used = args.backbone

    print(f"[Eval] model_family={model_family} legacy={legacy} ckpt={ckpt_path.name}")
    print(f"       csv={csv_path.name} split={split_path.name} backbone_used={backbone_used} fusion_used={fusion_used}")
    print(f"       zscore_phys={bool(args.zscore_phys)}")
    model.eval()

    probs_all, y_all, scar_all = [], [], []
    cf_gap_list = []
    gate_list, focus_list = [], []

    for b in loader:
        img = b["img"].to(device)
        img_cf = b["img_cf"].to(device)
        phys = b["phys"].to(device)
        y = b["y"].cpu().numpy()
        scar = b["scar"].cpu().numpy()
        has_cf = b["has_cf"].cpu().numpy().astype(bool)
        mask = b["mask"].to(device)

        if mu_t is not None and sigma_t is not None:
            phys = (phys - mu_t) / sigma_t

        out = model(img, phys, mask=mask)
        p = F.softmax(out.logits, dim=1)[:, 1].detach().cpu().numpy()

        if has_cf.any():
            out_cf = model(img_cf, phys, mask=mask)
            p_cf = F.softmax(out_cf.logits, dim=1)[:, 1].detach().cpu().numpy()
            cf_gap_list.extend(np.abs(p[has_cf] - p_cf[has_cf]).tolist())

        if getattr(out, "gate", None) is not None:
            gate_list.extend(out.gate.detach().cpu().numpy().reshape(-1).tolist())
        if getattr(out, "focus", None) is not None:
            focus_list.extend(out.focus.detach().cpu().numpy().reshape(-1).tolist())

        probs_all.append(p)
        y_all.append(y)
        scar_all.append(scar)

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)
    scar_all = np.concatenate(scar_all)

    yhat = (probs_all >= float(args.threshold)).astype(int)

    acc = float((yhat == y_all).mean())
    f1 = f1_score_binary(y_all, yhat)
    bacc = balanced_accuracy(y_all, yhat)
    auc = auc_roc_np(y_all, probs_all)

    dp_s = dp_gap_signed(yhat, scar_all)
    eo = eo_gaps(yhat, y_all, scar_all)
    cf_gap = float(np.mean(cf_gap_list)) if len(cf_gap_list) else 0.0

    out_json = Path(args.out) if args.out else (OUT_DIR / f"fairness_{model_family}_{ckpt_path.stem}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "checkpoint": str(ckpt_path),
        "csv": str(csv_path),
        "split": str(split_path),
        "model_family": model_family,
        "legacy_checkpoint": legacy,
        "fusion_used": fusion_used,
        "vision_backbone_used": backbone_used,
        "fusion_arg": args.fusion,
        "backbone_arg": args.backbone,
        "seed": int(args.seed),
        "zscore_phys": bool(args.zscore_phys),
        "n_val": int(len(y_all)),
        "threshold": float(args.threshold),
        "acc": acc,
        "f1": float(f1),
        "balanced_acc": float(bacc),
        "auc_roc": float(auc),
        "dp_gap_signed": dp_s,
        "dp_gap_abs": float(abs(dp_s)),
        "eo": eo,
        "cf_prob_gap_mean_abs": cf_gap,
        "gate_mean": float(np.mean(gate_list)) if gate_list else None,
        "focus_mean": float(np.mean(focus_list)) if focus_list else None,
        "cf_samples": int(len(cf_gap_list)),
    }

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
