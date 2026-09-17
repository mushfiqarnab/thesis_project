from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models import MultimodalThreatModel, count_trainable_params


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_CKPT = PROJECT_ROOT / "outputs" / "checkpoints"
OUT_REP  = PROJECT_ROOT / "outputs" / "reports"
OUT_CKPT.mkdir(parents=True, exist_ok=True)
OUT_REP.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser("Train CGF with counterfactual + fairness losses (thesis version).")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "vit_b_16"])
    p.add_argument("--fusion", type=str, default="cgf", choices=["cgf", "concat"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)  # Windows-safe default
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.2)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_accum", type=int, default=1)

    p.add_argument("--ckpt_in", type=str, default="")
    p.add_argument("--freeze_vision", action="store_true")

    # CF + gate
    p.add_argument("--lambda_cf", type=float, default=1.0)
    p.add_argument("--lambda_gate", type=float, default=0.05)

    # Fairness penalties (differentiable, probability-based)
    p.add_argument("--lambda_dp", type=float, default=0.5)
    p.add_argument("--lambda_eo", type=float, default=0.5)

    # Model selection score weights
    p.add_argument("--w_dp", type=float, default=1.0)
    p.add_argument("--w_eo", type=float, default=1.0)
    p.add_argument("--w_cf", type=float, default=0.2)

    # Data handling
    p.add_argument("--zscore_phys", action="store_true", help="Z-score physiology using TRAIN split only.")
    p.add_argument("--balance_groups", action="store_true", help="Balance (scar, label) groups via sampler.")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_or_load_split(split_path: Path, n: int, seed: int, val_ratio: float):
    if split_path.exists():
        d = json.loads(split_path.read_text(encoding="utf-8"))
        return d["train_idx"], d["val_idx"]

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = int(val_ratio * n)
    val_idx = idx[:val_n].tolist()
    train_idx = idx[val_n:].tolist()

    split_path.write_text(
        json.dumps({"seed": seed, "val_ratio": val_ratio, "train_idx": train_idx, "val_idx": val_idx}, indent=2),
        encoding="utf-8",
    )
    return train_idx, val_idx


def load_state_dict_safely(path: str, device: torch.device) -> dict:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location=device)

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict.")

    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        cleaned[k] = v
    return cleaned


def make_amp(device: torch.device, enabled: bool):
    """
    Returns (scaler, amp_context_factory) in a torch-version-safe way.
    Works for both new torch.amp and legacy torch.cuda.amp.
    """
    if (not enabled) or (device.type != "cuda"):
        return None, (lambda: nullcontext())

    # Newer API: torch.amp (preferred). IMPORTANT: use positional "cuda" (no device_type kw).
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        return scaler, (lambda: torch.amp.autocast("cuda", enabled=True))
    except Exception:
        # Fallback: legacy API
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        return scaler, (lambda: torch.cuda.amp.autocast(enabled=True))


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p,q: (B,2) probabilities -> (B,)
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def dp_gap_prob(p1: torch.Tensor, scar: torch.Tensor) -> torch.Tensor:
    # p1: (B,) probability of class 1
    s1 = (scar == 1)
    s0 = (scar == 0)
    m1 = p1[s1].mean() if s1.any() else torch.tensor(0.0, device=p1.device)
    m0 = p1[s0].mean() if s0.any() else torch.tensor(0.0, device=p1.device)
    return (m1 - m0).abs()


def eo_gap_prob(p1: torch.Tensor, y: torch.Tensor, scar: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Differentiable approximation:
    # TPR_g ≈ sum(p*y)/sum(y), FPR_g ≈ sum(p*(1-y))/sum(1-y)
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
    cf_abs_sum = 0.0
    cf_count = 0

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
        p = torch.softmax(out.logits, dim=1)[:, 1]

        if has_cf.any():
            out_cf = model(img_cf, phys, mask=mask)
            p_cf = torch.softmax(out_cf.logits, dim=1)[:, 1]
            dif = (p[has_cf] - p_cf[has_cf]).abs()
            cf_abs_sum += float(dif.sum().item())
            cf_count += int(dif.numel())

        probs_all.append(p.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())
        scar_all.append(scar.detach().cpu().numpy())

    probs = np.concatenate(probs_all)
    y_np = np.concatenate(y_all)
    s_np = np.concatenate(scar_all)
    yhat = (probs >= 0.5).astype(int)

    acc = float((yhat == y_np).mean())

    # DP on hard predictions (matches your eval_fairness style)
    if (s_np == 1).any() and (s_np == 0).any():
        dp = float(abs(yhat[s_np == 1].mean() - yhat[s_np == 0].mean()))
    else:
        dp = 0.0

    # EO on hard predictions
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

    cf_gap = float(cf_abs_sum / max(cf_count, 1))
    return {"acc": acc, "dp_abs": dp, "eo_max_gap": eo_max, "cf_gap": cf_gap}


def main():
    args = parse_args()
    set_seed(args.seed)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and (device.type == "cuda"))

    ds = MultimodalCSVDatasetWithCF(str(csv_path))

    split_path = csv_path.parent / f"split_seed{args.seed}_{csv_path.stem}.json"
    train_idx, val_idx = make_or_load_split(split_path, len(ds), args.seed, args.val_ratio)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    # ---- optional z-score physiology using TRAIN split only (no leakage) ----
    phys_mu = phys_sigma = None
    if args.zscore_phys:
        # Your dataset exposes df + phys_cols
        X = ds.df.iloc[train_idx][ds.phys_cols].to_numpy(dtype=np.float32, copy=True)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)
        phys_mu = torch.tensor(mu, device=device, dtype=torch.float32).unsqueeze(0)
        phys_sigma = torch.tensor(sigma, device=device, dtype=torch.float32).unsqueeze(0)

    # ---- optional group balancing sampler (scar,label) ----
    sampler = None
    if args.balance_groups:
        sc = []
        yy = []
        for i in train_idx:
            r = ds.df.iloc[i]
            sc.append(int(r["scar"]))
            yy.append(int(r[ds.label_col]))
        sc = np.asarray(sc, dtype=np.int64)
        yy = np.asarray(yy, dtype=np.int64)

        gid = 2 * sc + yy  # 0..3 for (scar,y)
        counts = np.bincount(gid, minlength=4).astype(np.float64)
        counts = np.where(counts == 0, 1.0, counts)
        w = 1.0 / counts[gid]
        w = torch.tensor(w, dtype=torch.double)
        sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)

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

    if args.ckpt_in:
        sd = load_state_dict_safely(args.ckpt_in, device)
        model.load_state_dict(sd, strict=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    # AMP (torch-version safe)
    scaler, amp_ctx = make_amp(device, enabled=use_amp)

    best_score = -1e9
    best_path = OUT_CKPT / f"counterfactual_{args.fusion}_js_{args.backbone}_{csv_path.stem}_best.pt"

    print(f"[Train] device={device} amp={use_amp} csv={csv_path.name} n={len(ds)} split={split_path.name}")
    print(f"[Train] params_trainable={count_trainable_params(model)} backbone={args.backbone} fusion={args.fusion}")

    grad_accum = max(int(args.grad_accum), 1)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")

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

                # CF loss only where CF exists
                loss_cf = torch.tensor(0.0, device=device)
                if has_cf.any():
                    out_cf = model(img_cf, phys, mask=mask)
                    p = F.softmax(out.logits, dim=1)
                    q = F.softmax(out_cf.logits, dim=1)
                    js = js_divergence(p, q)
                    loss_cf = js[has_cf].mean()

                # Gate regularizer (stabilized by log1p on focus)
                loss_gate = torch.tensor(0.0, device=device)
                if out.gate is not None and out.focus is not None:
                    focus = torch.log1p(out.focus.clamp(min=0.0, max=1e3))
                    loss_gate = (out.gate * focus).mean()

                # Fairness penalties (probability-based)
                p1 = F.softmax(out.logits, dim=1)[:, 1]
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

            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            if step % grad_accum == 0:
                if scaler is None:
                    opt.step()
                else:
                    scaler.step(opt)
                    scaler.update()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix(
                task=float(loss_task.item()),
                cf=float(loss_cf.item()),
                gate=float(loss_gate.item()),
                dp=float(loss_dp.item()),
                eo=float(loss_eo.item()),
            )

        # If we ended mid-accumulation, do one final optimizer step
        if last_step % grad_accum != 0:
            if scaler is None:
                opt.step()
            else:
                scaler.step(opt)
                scaler.update()
            opt.zero_grad(set_to_none=True)

        val = eval_metrics(model, val_loader, device, phys_mu=phys_mu, phys_sigma=phys_sigma)
        score = val["acc"] - args.w_dp * val["dp_abs"] - args.w_eo * val["eo_max_gap"] - args.w_cf * val["cf_gap"]
        print(f"[epoch {epoch}] val={val} score={score:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
            print(f"[Best] Saved: {best_path} (epoch={epoch}, score={best_score:.4f})")

    report = {
        "csv": str(csv_path),
        "split_path": str(split_path),
        "backbone": args.backbone,
        "fusion": args.fusion,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "amp": use_amp,
        "grad_accum": grad_accum,
        "zscore_phys": args.zscore_phys,
        "balance_groups": args.balance_groups,
        "lambda_cf": args.lambda_cf,
        "lambda_gate": args.lambda_gate,
        "lambda_dp": args.lambda_dp,
        "lambda_eo": args.lambda_eo,
        "w_dp": args.w_dp,
        "w_eo": args.w_eo,
        "w_cf": args.w_cf,
        "best_score": best_score,
        "best_ckpt": str(best_path),
        "params_trainable": int(count_trainable_params(model)),
    }

    rep_path = OUT_REP / f"train_counterfactual_{csv_path.stem}_{args.backbone}.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved report:", rep_path)


if __name__ == "__main__":
    main()
