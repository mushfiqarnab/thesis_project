"""
train_fair_empirical.py
=======================
Standalone fair training script. Inherits the real MultimodalClinicalDataset
and GWPACDNet backbone, but fixes the critical flaw in train_empirical.py:
the fairness constraint now enters the GRADIENT via a differentiable
batch-level DP penalty term in the training loss, not just the eval score.

The DP penalty is:
    L_total = L_cls + lambda_dp * |P(y_hat=1|scar=1) - P(y_hat=1|scar=0)|

where probabilities are computed from soft sigmoid scores (not argmax),
making the term differentiable with respect to model weights.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.pacd_net import GWPACDNet
from src.data.clinical_dataloader import MultimodalClinicalDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FairEmpiricalTrainer")


def differentiable_dp_penalty(logits: torch.Tensor, scar: torch.Tensor) -> torch.Tensor:
    """
    Batch-level demographic parity penalty.
    Uses soft probabilities (sigmoid of positive-class logit) so the term
    is differentiable and gradients flow back to model weights.

    Returns scalar tensor (mean |P(pos|scar=1) - P(pos|scar=0)|) for the batch.
    If a group is absent from the batch, returns 0 to avoid NaN.
    """
    pos_prob = torch.sigmoid(logits[:, 1])

    mask1 = (scar == 1)
    mask0 = (scar == 0)

    if mask1.sum() == 0 or mask0.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=False)

    p1 = pos_prob[mask1].mean()
    p0 = pos_prob[mask0].mean()
    return torch.abs(p1 - p0)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    pos1, n1, pos0, n0 = 0, 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            img   = batch["img"].to(device)
            phys  = batch["phys"].to(device)
            y     = batch["y"].to(device)
            scar  = batch["scar"].to(device)
            out   = model(img=img, phys=phys, scar_labels=None)
            preds = torch.argmax(out["logits"], dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
            m1, m0  = (scar == 1), (scar == 0)
            pos1 += preds[m1].sum().item(); n1 += m1.sum().item()
            pos0 += preds[m0].sum().item(); n0 += m0.sum().item()

    acc    = correct / max(total, 1)
    dp_gap = abs(pos1 / max(n1, 1) - pos0 / max(n0, 1))
    pareto = acc - 5.0 * dp_gap
    return acc, dp_gap, pareto


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        type=str,   default="data/csv/multimodal_10k.csv")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--lambda_dp",  type=float, default=2.0,
                        help="Weight on the DP fairness penalty. Increase to suppress DP gap.")
    parser.add_argument("--val_frac",   type=float, default=0.15,
                        help="Fraction of data held out for validation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  lambda_dp: {args.lambda_dp}")

    full_ds  = MultimodalClinicalDataset(args.csv)
    n_val    = int(len(full_ds) * args.val_frac)
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Train: {n_train} samples  |  Val: {n_val} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model     = GWPACDNet(d=64, k=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_pareto   = -float("inf")
    best_ckpt     = "outputs/fair_model_best.pth"
    os.makedirs("outputs", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_cls, total_dp, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            img   = batch["img"].to(device)
            phys  = batch["phys"].to(device)
            y     = batch["y"].to(device)
            scar  = batch["scar"].to(device)

            optimizer.zero_grad()
            out      = model(img=img, phys=phys, scar_labels=None)
            logits   = out["logits"]

            loss_cls = criterion(logits, y)
            loss_dp  = differentiable_dp_penalty(logits, scar)
            loss     = loss_cls + args.lambda_dp * loss_dp

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_cls  += loss_cls.item()
            total_dp   += loss_dp.item()
            n_batches  += 1

        scheduler.step()
        avg_cls = total_cls / max(n_batches, 1)
        avg_dp  = total_dp  / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            val_acc, val_dp, pareto = evaluate(model, val_loader, device)
            logger.info(
                f"EPOCH {epoch+1:03d} | "
                f"L_cls={avg_cls:.4f}  L_dp={avg_dp:.4f} | "
                f"Val Acc={val_acc:.4f}  DP Gap={val_dp:.4f}  Pareto={pareto:.4f} | "
                f"LR={scheduler.get_last_lr()[0]:.2e}"
            )
            if pareto > best_pareto:
                best_pareto = pareto
                torch.save(model.state_dict(), best_ckpt)
                logger.info(f"    >>> Checkpoint saved -> {best_ckpt}")

    logger.info("Training complete.")
    logger.info(f"Best Pareto Score: {best_pareto:.4f}")
    logger.info(f"Best checkpoint  : {best_ckpt}")


if __name__ == "__main__":
    main()
