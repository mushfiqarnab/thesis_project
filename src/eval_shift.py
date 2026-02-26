from __future__ import annotations

from pathlib import Path
import argparse
import json
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision.models as tvm
import torchvision.transforms.functional as TF

from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models import MultimodalThreatModel


# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "csv" / "multimodal.csv"
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "csv" / "split_seed42.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate robustness under distribution shift (image/phys).")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pt)")
    p.add_argument("--fusion", type=str, required=True, choices=["concat", "cgf"], help="Fusion type (for current models)")
    p.add_argument("--backbone", type=str, default="mobilenet_v3_small",
                   choices=["mobilenet_v3_small", "vit_b_16"], help="Backbone (for current models)")
    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="CSV path (multimodal.csv or multimodal_10k.csv)")
    p.add_argument("--split", type=str, default=str(DEFAULT_SPLIT), help="Split json path (split_seed42.json)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = leave default)")
    p.add_argument("--seed", type=int, default=42)

    # Shift settings
    p.add_argument("--shift", type=str, default="none", choices=[
        "none",
        "phys_scale", "phys_noise",
        "img_brightness", "img_blur", "img_noise", "img_occlusion",
        "combo_light", "combo_hard",
    ])
    p.add_argument("--severity", type=float, default=0.10,
                   help="Shift strength (meaning depends on shift). Typical: 0.05-0.30")

    p.add_argument("--out", type=str, default="", help="Optional output json path")
    return p.parse_args()


# ---------------- Metrics ----------------
def dp_gap_signed(yhat: np.ndarray, scar: np.ndarray) -> float:
    p1 = yhat[scar == 1].mean() if (scar == 1).any() else 0.0
    p0 = yhat[scar == 0].mean() if (scar == 0).any() else 0.0
    return float(p1 - p0)


def eo_gaps(yhat: np.ndarray, y: np.ndarray, scar: np.ndarray) -> dict:
    def rates(g):
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
        "tpr1": tpr1, "tpr0": tpr0,
        "fpr1": fpr1, "fpr0": fpr0,
        "tpr_gap": float(tpr1 - tpr0),
        "fpr_gap": float(fpr1 - fpr0),
        "eo_max_gap": float(max(abs(tpr1 - tpr0), abs(fpr1 - fpr0))),
    }


def f1_score(yhat: np.ndarray, y: np.ndarray) -> float:
    tp = ((yhat == 1) & (y == 1)).sum()
    fp = ((yhat == 1) & (y == 0)).sum()
    fn = ((yhat == 0) & (y == 1)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if (prec + rec) == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def balanced_accuracy(yhat: np.ndarray, y: np.ndarray) -> float:
    tp = ((yhat == 1) & (y == 1)).sum()
    fn = ((yhat == 0) & (y == 1)).sum()
    tn = ((yhat == 0) & (y == 0)).sum()
    fp = ((yhat == 1) & (y == 0)).sum()
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return float(0.5 * (tpr + tnr))


def auc_roc(probs: np.ndarray, y: np.ndarray) -> float:
    # Mann–Whitney U using average ranks (tie-safe via pandas)
    y = y.astype(int)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = pd.Series(probs).rank(method="average").to_numpy()
    rank_sum_pos = ranks[y == 1].sum()
    auc = (rank_sum_pos - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(auc)


# ---------------- Checkpoint loading (current + legacy) ----------------
def load_raw_state_dict(ckpt_path: Path, device: torch.device) -> dict:
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict.")

    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def is_legacy_checkpoint(state: dict) -> bool:
    keys = list(state.keys())
    has_vit = any(k.startswith("vit.") for k in keys)
    has_phys_old = any(k.startswith("phys_net.") for k in keys)
    has_cls_old = any(k.startswith("classifier.") for k in keys)
    return has_vit and (has_phys_old or has_cls_old)


class LegacyViTConcatModel(nn.Module):
    """
    Matches your old checkpoint shapes:
      vit.* (ViT-B/16, head removed)
      phys_net: Linear(2->32)->ReLU->Linear(32->32)->ReLU
      classifier: Linear(800->128)->ReLU->Dropout->Linear(128->2)
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
            raise AttributeError("Unexpected ViT model: cannot remove head/heads.")

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

    def forward(self, img: torch.Tensor, phys: torch.Tensor, mask: torch.Tensor | None = None):
        v = self.vit(img)            # (B,768)
        p = self.phys_net(phys)      # (B,32)
        x = torch.cat([v, p], dim=1) # (B,800)
        logits = self.classifier(x)
        return SimpleNamespace(logits=logits, gate=None, focus=None)


# ---------------- Shift functions ----------------
def denorm(x: torch.Tensor) -> torch.Tensor:
    # x: (B,3,H,W) normalized
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return x * std + mean


def renorm(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return (x - mean) / std


def apply_image_shift_pair(
    img: torch.Tensor,
    img_cf: torch.Tensor,
    shift: str,
    severity: float,
    g: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the SAME random effect to img and img_cf (important for CF-gap).
    img/img_cf are normalized tensors (B,3,H,W).
    """
    if shift == "none":
        return img, img_cf

    x = denorm(img).clamp(0, 1)
    xcf = denorm(img_cf).clamp(0, 1)

    B, C, H, W = x.shape

    if shift == "img_brightness":
        # severity ~ 0.05..0.30  (brightness factor = 1 +/- severity)
        factor = 1.0 + float(severity)
        x = (x * factor).clamp(0, 1)
        xcf = (xcf * factor).clamp(0, 1)

    elif shift == "img_blur":
        # severity ~ 1..5 (kernel size approx)
        s = max(1, int(round(severity * 10)))
        k = 2 * s + 1
        x = TF.gaussian_blur(x, [k, k], sigma=[max(0.1, severity), max(0.1, severity)])
        xcf = TF.gaussian_blur(xcf, [k, k], sigma=[max(0.1, severity), max(0.1, severity)])

    elif shift == "img_noise":
        # severity ~ 0.02..0.20 (std in pixel space)
        noise = torch.randn_like(x, generator=g) * float(severity)
        x = (x + noise).clamp(0, 1)
        xcf = (xcf + noise).clamp(0, 1)

    elif shift == "img_occlusion":
        # severity ~ 0.10..0.40 (patch ratio)
        r = float(severity)
        ph = max(8, int(H * r))
        pw = max(8, int(W * r))
        top = torch.randint(0, max(1, H - ph + 1), (B,), generator=g, device=x.device)
        left = torch.randint(0, max(1, W - pw + 1), (B,), generator=g, device=x.device)
        for i in range(B):
            x[i, :, top[i]:top[i]+ph, left[i]:left[i]+pw] = 0.0
            xcf[i, :, top[i]:top[i]+ph, left[i]:left[i]+pw] = 0.0

    elif shift == "combo_light":
        # mild brightness + mild noise
        factor = 1.0 + float(severity)
        noise = torch.randn_like(x, generator=g) * float(severity) * 0.5
        x = (x * factor + noise).clamp(0, 1)
        xcf = (xcf * factor + noise).clamp(0, 1)

    elif shift == "combo_hard":
        # stronger blur + noise
        s = max(1, int(round(severity * 10)))
        k = 2 * s + 1
        x = TF.gaussian_blur(x, [k, k], sigma=[max(0.1, severity), max(0.1, severity)])
        xcf = TF.gaussian_blur(xcf, [k, k], sigma=[max(0.1, severity), max(0.1, severity)])
        noise = torch.randn_like(x, generator=g) * float(severity)
        x = (x + noise).clamp(0, 1)
        xcf = (xcf + noise).clamp(0, 1)

    return renorm(x), renorm(xcf)


def apply_phys_shift(
    phys: torch.Tensor,
    shift: str,
    severity: float,
    g: torch.Generator,
) -> torch.Tensor:
    if shift == "none":
        return phys

    if shift == "phys_scale":
        # multiplicative sensor drift
        return phys * (1.0 + float(severity))

    if shift == "phys_noise":
        # additive noise proportional to feature scale
        scale = phys.detach().abs().mean(dim=0, keepdim=True).clamp(min=1e-6)
        noise = torch.randn_like(phys, generator=g) * (scale * float(severity))
        return phys + noise

    # for combo shifts, keep phys unchanged unless user chooses phys shifts
    return phys


# ---------------- Main ----------------
@torch.no_grad()
def main():
    args = parse_args()

    if args.threads and args.threads > 0:
        torch.set_num_threads(int(args.threads))

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    split_path = Path(args.split)
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found: {split_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset + val split (same contract as eval_fairness.py)
    ds = MultimodalCSVDatasetWithCF(str(csv_path))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    val_ds = Subset(ds, split["val_idx"])

    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_samples,
    )

    # Load model (current or legacy)
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

    model.eval()

    # RNG (reproducible shifts)
    g = torch.Generator(device=device)
    g.manual_seed(int(args.seed))

    probs_all, y_all, scar_all = [], [], []
    cf_gap_list = []

    # Decide which shifts are image vs phys
    img_shift = args.shift if args.shift.startswith("img_") or args.shift.startswith("combo") else "none"
    phys_shift = args.shift if args.shift.startswith("phys_") else "none"

    for b in loader:
        img = b["img"].to(device)
        img_cf = b["img_cf"].to(device)
        phys = b["phys"].to(device)
        mask = b["mask"].to(device)

        y = b["y"].cpu().numpy()
        scar = b["scar"].cpu().numpy()
        has_cf = b["has_cf"].cpu().numpy().astype(bool)

        # Apply shifts
        img_s, img_cf_s = apply_image_shift_pair(img, img_cf, img_shift, args.severity, g)
        phys_s = apply_phys_shift(phys, phys_shift, args.severity, g)

        out = model(img_s, phys_s, mask=mask)
        p = F.softmax(out.logits, dim=1)[:, 1].detach().cpu().numpy()

        if has_cf.any():
            out_cf = model(img_cf_s, phys_s, mask=mask)
            p_cf = F.softmax(out_cf.logits, dim=1)[:, 1].detach().cpu().numpy()
            cf_gap_list.extend(np.abs(p[has_cf] - p_cf[has_cf]).tolist())

        probs_all.append(p)
        y_all.append(y)
        scar_all.append(scar)

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)
    scar_all = np.concatenate(scar_all)

    yhat = (probs_all >= 0.5).astype(int)

    report = {
        "ckpt": str(ckpt_path),
        "csv": str(csv_path),
        "split": str(split_path),
        "model_family": model_family,
        "legacy_checkpoint": legacy,
        "fusion_used": fusion_used,
        "vision_backbone_used": backbone_used,
        "fusion_arg": args.fusion,
        "backbone_arg": args.backbone,
        "shift": args.shift,
        "severity": float(args.severity),
        "seed": int(args.seed),
        "threshold": 0.5,
        "n_val": int(len(y_all)),
        "acc": float((yhat == y_all).mean()),
        "f1": f1_score(yhat, y_all),
        "balanced_acc": balanced_accuracy(yhat, y_all),
        "auc_roc": auc_roc(probs_all, y_all),
        "dp_gap_signed": dp_gap_signed(yhat, scar_all),
        "dp_gap_abs": float(abs(dp_gap_signed(yhat, scar_all))),
        "eo": eo_gaps(yhat, y_all, scar_all),
        "cf_prob_gap_mean_abs": float(np.mean(cf_gap_list)) if len(cf_gap_list) else 0.0,
        "cf_samples": int(len(cf_gap_list)),
    }

    if args.out:
        out_json = Path(args.out)
    else:
        out_json = OUT_DIR / f"shift_{args.shift}_sev{args.severity:g}_{ckpt_path.stem}.json"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
