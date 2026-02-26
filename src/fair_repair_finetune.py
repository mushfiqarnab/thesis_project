from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
from contextlib import nullcontext
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models import MultimodalThreatModel, count_trainable_params


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_REP = PROJECT_ROOT / "outputs" / "reports"
OUT_REP.mkdir(parents=True, exist_ok=True)


def _resolve_path(p: str) -> Path:
    """Resolve user path relative to PROJECT_ROOT unless already absolute."""
    pp = Path(p)
    return pp if pp.is_absolute() else (PROJECT_ROOT / pp)


def parse_args():
    p = argparse.ArgumentParser("Fairness repair fine-tuning after pruning (thesis-safe).")

    p.add_argument("--ckpt_in", type=str, required=True, help="Input pruned checkpoint (.pt)")
    p.add_argument("--ckpt_out", type=str, required=True, help="Output repaired checkpoint (.pt)")

    # Require csv + split to prevent index mismatch
    p.add_argument("--csv", type=str, required=True, help="CSV used for training (must match split indices)")
    p.add_argument("--split", type=str, required=True, help="Split json produced by training for this CSV")

    p.add_argument("--fusion", type=str, default="cgf", choices=["cgf", "concat"])
    p.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "vit_b_16"])

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)  # Windows-safe
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--freeze_vision", action="store_true")

    # Loss weights (aligned with your training)
    p.add_argument("--lambda_cf", type=float, default=1.0)
    p.add_argument("--lambda_gate", type=float, default=0.05)
    p.add_argument("--lambda_dp", type=float, default=0.3)
    p.add_argument("--lambda_eo", type=float, default=0.3)

    # Data options (aligned with your training)
    p.add_argument("--zscore_phys", action="store_true", help="Z-score physiology using TRAIN split only.")
    p.add_argument("--balance_groups", action="store_true", help="Balance (scar,label) groups via sampler.")

    # Save best by score like training
    p.add_argument("--w_dp", type=float, default=1.0)
    p.add_argument("--w_eo", type=float, default=1.0)
    p.add_argument("--w_cf", type=float, default=0.2)

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_state_dict_any(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # torch.load weights_only is version-dependent
    try:
        state = torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(path), map_location=device)

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict.")

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p,q: (B,2) probabilities -> (B,)
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def dp_gap_prob(p1: torch.Tensor, scar: torch.Tensor) -> torch.Tensor:
    s1 = (scar == 1)
    s0 = (scar == 0)
    m1 = p1[s1].mean() if s1.any() else torch.tensor(0.0, device=p1.device)
    m0 = p1[s0].mean() if s0.any() else torch.tensor(0.0, device=p1.device)
    return (m1 - m0).abs()


def eo_gap_prob(p1: torch.Tensor, y: torch.Tensor, scar: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y01 = y.float()

    def rates(g: int):
        idx = (scar == g)
        if not idx.any():
            z = torch.tensor(0.0, device=p1.device)
            return z, z
        pg = p1[idx]
        yg = y01[idx]
        tpr = (pg * yg).sum() / (yg.sum() + eps)
        fpr = (pg * (1.0 - yg)).sum() / ((1.0 - yg).sum() + eps)
        return tpr, fpr

    tpr1, fpr1 = rates(1)
    tpr0, fpr0 = rates(0)
    return torch.max((tpr1 - tpr0).abs(), (fpr1 - fpr0).abs())


@torch.no_grad()
def eval_metrics(model, loader, device, phys_mu=None, phys_sigma=None):
    model.eval()
    probs_all, y_all, scar_all = [], [], []
    cf_gap_list = []

    for b in loader:
        img = b["img"].to(device)
        img_cf = b["img_cf"].to(device)
        phys = b["phys"].to(device)
        y = b["y"].to(device)
        scar = b["scar"].to(device)
        has_cf = b["has_cf"].to(device).bool()
        mask = b["mask"].to(device)

        if phys_mu is not None and phys_sigma is not None:
            phys = (phys - phys_mu) / phys_sigma

        out = model(img, phys, mask=mask)
        p1 = torch.softmax(out.logits, dim=1)[:, 1]

        if has_cf.any():
            out_cf = model(img_cf, phys, mask=mask)
            p1_cf = torch.softmax(out_cf.logits, dim=1)[:, 1]
            cf_gap_list.append((p1[has_cf] - p1_cf[has_cf]).abs().mean().item())

        probs_all.append(p1.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())
        scar_all.append(scar.detach().cpu().numpy())

    probs = np.concatenate(probs_all)
    y_np = np.concatenate(y_all)
    s_np = np.concatenate(scar_all)
    yhat = (probs >= 0.5).astype(int)

    acc = float((yhat == y_np).mean())
    dp = float(abs(yhat[s_np == 1].mean() - yhat[s_np == 0].mean())) if (s_np == 1).any() and (s_np == 0).any() else 0.0

    def eo_rates(g):
        idx = (s_np == g)
        if not idx.any():
            return 0.0, 0.0
        yy = y_np[idx]
        yh = yhat[idx]
        tp = ((yh == 1) & (yy == 1)).sum()
        fn = ((yh == 0) & (yy == 1)).sum()
        fp = ((yh == 1) & (yy == 0)).sum()
        tn = ((yh == 0) & (yy == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        return float(tpr), float(fpr)

    tpr1, fpr1 = eo_rates(1)
    tpr0, fpr0 = eo_rates(0)
    eo_max = float(max(abs(tpr1 - tpr0), abs(fpr1 - fpr0)))

    cf_gap = float(np.mean(cf_gap_list)) if len(cf_gap_list) else 0.0
    return {"acc": acc, "dp_abs": dp, "eo_max_gap": eo_max, "cf_gap": cf_gap}


def get_amp(device: torch.device, want_amp: bool):
    """
    AMP that works across your exact PyTorch behavior:
    - Your environment FAILED on GradScaler(device_type=...)
    - So we try the safe signatures in order.
    """
    use_amp = bool(want_amp and device.type == "cuda")
    if not use_amp:
        return (lambda: nullcontext()), None, False

    # Try torch.amp first
    try:
        from torch.amp import autocast, GradScaler  # type: ignore

        def ctx():
            # Some versions prefer device_type keyword, some accept positional
            try:
                return autocast(device_type="cuda", enabled=True)
            except TypeError:
                return autocast("cuda", enabled=True)

        # Try safest GradScaler call patterns
        try:
            scaler = GradScaler("cuda", enabled=True)
        except TypeError:
            try:
                scaler = GradScaler(device="cuda", enabled=True)
            except TypeError:
                scaler = GradScaler(enabled=True)

        return ctx, scaler, True

    except Exception:
        # Fallback: torch.cuda.amp
        from torch.cuda.amp import autocast, GradScaler  # type: ignore

        def ctx():
            return autocast(enabled=True)

        scaler = GradScaler(enabled=True)
        return ctx, scaler, True


def _validate_split_indices(train_idx, val_idx, n: int, csv_path: Path, split_path: Path):
    if not isinstance(train_idx, list) or not isinstance(val_idx, list):
        raise ValueError("Split file must contain lists: train_idx and val_idx.")

    # Ensure all are ints (json sometimes loads as int already, but be strict)
    train_idx = [int(x) for x in train_idx]
    val_idx = [int(x) for x in val_idx]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Split has empty train_idx or val_idx. Recreate split by retraining.")

    max_idx = max(max(train_idx), max(val_idx))
    min_idx = min(min(train_idx), min(val_idx))

    if min_idx < 0 or max_idx >= n:
        raise ValueError(
            f"[Split/CSV mismatch]\n"
            f"- csv={csv_path} len(ds)={n}\n"
            f"- split={split_path} min_index={min_idx} max_index={max_idx}\n"
            f"Fix: use the SAME --csv that produced this --split file (or delete split and retrain)."
        )

    return train_idx, val_idx


def main():
    args = parse_args()
    set_seed(args.seed)

    csv_path = _resolve_path(args.csv)
    split_path = _resolve_path(args.split)
    ckpt_in = _resolve_path(args.ckpt_in)
    ckpt_out = _resolve_path(args.ckpt_out)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found: {split_path}")
    if not ckpt_in.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {ckpt_in}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx, scaler, use_amp = get_amp(device, args.amp)

    ds = MultimodalCSVDatasetWithCF(str(csv_path))

    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_idx, val_idx = _validate_split_indices(split.get("train_idx"), split.get("val_idx"), len(ds), csv_path, split_path)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    # ---- optional z-score physiology using TRAIN split only ----
    phys_mu = phys_sigma = None
    if args.zscore_phys:
        X = ds.df.iloc[train_idx][ds.phys_cols].to_numpy(dtype=np.float32, copy=True)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)
        phys_mu = torch.tensor(mu, device=device, dtype=torch.float32).unsqueeze(0)
        phys_sigma = torch.tensor(sigma, device=device, dtype=torch.float32).unsqueeze(0)

    # ---- optional group balancing sampler (scar,label) ----
    sampler = None
    if args.balance_groups:
        sc, yy = [], []
        for i in train_idx:
            r = ds.df.iloc[i]
            sc.append(int(r["scar"]))
            yy.append(int(r[ds.label_col]))
        sc = np.asarray(sc)
        yy = np.asarray(yy)
        gid = 2 * sc + yy  # 4 groups
        counts = np.bincount(gid, minlength=4).astype(np.float64)
        counts = np.where(counts == 0, 1.0, counts)
        w = 1.0 / counts[gid]
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(w), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_samples,
    )

    # ---- model ----
    phys_dim = ds[0].phys.numel()
    model = MultimodalThreatModel(
        phys_dim=phys_dim,
        vision_backbone=args.backbone,
        fusion=args.fusion,
        num_classes=2,
        freeze_vision=args.freeze_vision,
    ).to(device)

    model.load_state_dict(load_state_dict_any(ckpt_in, device), strict=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    best_score = -1e9
    best_state: Optional[Dict[str, torch.Tensor]] = None

    print(f"[Repair] device={device} amp={use_amp} csv={csv_path.name} n={len(ds)} split={split_path.name}")
    print(f"[Repair] ckpt_in={ckpt_in} -> ckpt_out={ckpt_out}")
    print(f"[Repair] params_trainable={count_trainable_params(model)} backbone={args.backbone} fusion={args.fusion}")

    grad_accum = max(int(args.grad_accum), 1)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [repair]")

        opt.zero_grad(set_to_none=True)
        last_step = 0

        for step, b in enumerate(pbar, start=1):
            last_step = step

            img = b["img"].to(device)
            img_cf = b["img_cf"].to(device)
            phys = b["phys"].to(device)
            y = b["y"].to(device)
            scar = b["scar"].to(device)
            has_cf = b["has_cf"].to(device).bool()
            mask = b["mask"].to(device)

            if phys_mu is not None and phys_sigma is not None:
                phys = (phys - phys_mu) / phys_sigma

            with amp_ctx():
                out = model(img, phys, mask=mask)
                loss_task = ce(out.logits, y)

                loss_cf = torch.tensor(0.0, device=device)
                if has_cf.any():
                    out_cf = model(img_cf, phys, mask=mask)
                    p = F.softmax(out.logits, dim=1)
                    q = F.softmax(out_cf.logits, dim=1)
                    loss_cf = js_divergence(p, q)[has_cf].mean()

                loss_gate = torch.tensor(0.0, device=device)
                if getattr(out, "gate", None) is not None and getattr(out, "focus", None) is not None:
                    focus = torch.log1p(out.focus.clamp(min=0.0, max=1e3))
                    loss_gate = (out.gate * focus).mean()

                p1 = torch.softmax(out.logits, dim=1)[:, 1]
                loss_dp = dp_gap_prob(p1, scar)
                loss_eo = eo_gap_prob(p1, y, scar)

                loss = (
                    loss_task
                    + args.lambda_cf * loss_cf
                    + args.lambda_gate * loss_gate
                    + args.lambda_dp * loss_dp
                    + args.lambda_eo * loss_eo
                )

            loss = loss / grad_accum

            if use_amp:
                assert scaler is not None
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accum == 0:
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix(
                task=float(loss_task.item()),
                cf=float(loss_cf.item()),
                gate=float(loss_gate.item()),
                dp=float(loss_dp.item()),
                eo=float(loss_eo.item()),
            )

        # flush remainder if grad_accum does not divide steps
        if last_step % grad_accum != 0:
            if use_amp:
                assert scaler is not None
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        val = eval_metrics(model, val_loader, device, phys_mu=phys_mu, phys_sigma=phys_sigma)
        score = val["acc"] - args.w_dp * val["dp_abs"] - args.w_eo * val["eo_max_gap"] - args.w_cf * val["cf_gap"]
        print(f"[epoch {epoch}] val={val} score={score:.4f}")

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(f"[Best] updated (epoch={epoch}, score={best_score:.4f})")

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_out)
    print("[Saved repaired ckpt]:", ckpt_out)

    report = {
        "ckpt_in": str(ckpt_in),
        "ckpt_out": str(ckpt_out),
        "csv": str(csv_path),
        "split": str(split_path),
        "seed": int(args.seed),
        "backbone": args.backbone,
        "fusion": args.fusion,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "amp": bool(use_amp),
        "grad_accum": int(args.grad_accum),
        "freeze_vision": bool(args.freeze_vision),
        "zscore_phys": bool(args.zscore_phys),
        "balance_groups": bool(args.balance_groups),
        "lambda_cf": float(args.lambda_cf),
        "lambda_gate": float(args.lambda_gate),
        "lambda_dp": float(args.lambda_dp),
        "lambda_eo": float(args.lambda_eo),
        "best_score": float(best_score),
    }

    rep_path = OUT_REP / f"repair_{Path(args.csv).stem}_{args.backbone}.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved report:", rep_path)


if __name__ == "__main__":
    main()
