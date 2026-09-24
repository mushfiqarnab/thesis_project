#!/usr/bin/env python3
"""
train_equitas_rcmf.py
================================================================================
Master Trainer for EQUITAS-RCMF:
Riemannian Causal Manifold Fusion with Orthogonal Subspace Disentanglement

Mathematical Loss Functional:
  L = L_task
    + lambda_confounder * L_confounder       (Supervises W_confounder on scar)
    + lambda_causal_inv * L_causal_inv       (Forces v_causal to be scar-invariant)
    + lambda_latent_inv * L_latent_inv       (Isometrically aligns latent Z)
    + lambda_js * L_js                       (Output logit JS divergence)
    + lambda_dp * L_dp + lambda_eo * L_eo    (Fairness manifold regularization)

Constraint:
  W_causal^T @ W_confounder = 0 (topologically enforced by Stiefel layer)
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models.equitas_rcmf import EquitasRCMFModel
from train_cgf_fair import (
    dp_gap_prob,
    eo_gap_prob,
    get_annealing_factor,
    js_divergence_stable,
    make_amp,
    safe_divide,
    set_seed,
)

OUT_CKPT = PROJECT_ROOT / "outputs" / "checkpoints"
OUT_REP = PROJECT_ROOT / "outputs" / "reports"
OUT_CKPT.mkdir(parents=True, exist_ok=True)
OUT_REP.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("EquitasRCMF")


@torch.no_grad()
def eval_rcmf_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    phys_mu=None,
    phys_sigma=None,
    autonomous: bool = True,
):
    model.eval()
    probs_all, y_all, scar_all = [], [], []
    cf_abs_sum, cf_count = 0.0, 0
    latent_diff_sum = 0.0

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

        if autonomous:
            # Autonomous Edge Mode: no ground-truth masks are provided
            out = model(img, phys, mask=None)
            p = torch.softmax(out.logits, dim=1)[:, 1]

            if has_cf.any():
                out_cf = model(img_cf, phys, mask=None)
                p_cf = torch.softmax(out_cf.logits, dim=1)[:, 1]
                dif = (p[has_cf] - p_cf[has_cf]).abs()
                cf_abs_sum += float(dif.sum().item())
                cf_count += int(dif.numel())

                l_dif = F.mse_loss(out.latent_z[has_cf], out_cf.latent_z[has_cf])
                latent_diff_sum += float(l_dif.item()) * int(has_cf.sum().item())
        else:
            # Privileged Mode: pass exact factual and counterfactual masks
            effective_mask_factual = mask * scar.view(-1, 1, 1, 1)
            out = model(img, phys, mask=effective_mask_factual, scar_label=scar)
            p = torch.softmax(out.logits, dim=1)[:, 1]

            if has_cf.any():
                effective_mask_cf = mask * (1.0 - scar.view(-1, 1, 1, 1))
                out_cf = model(img_cf, phys, mask=effective_mask_cf, scar_label=(1.0 - scar))
                p_cf = torch.softmax(out_cf.logits, dim=1)[:, 1]
                dif = (p[has_cf] - p_cf[has_cf]).abs()
                cf_abs_sum += float(dif.sum().item())
                cf_count += int(dif.numel())

                l_dif = F.mse_loss(out.latent_z[has_cf], out_cf.latent_z[has_cf])
                latent_diff_sum += float(l_dif.item()) * int(has_cf.sum().item())

        probs_all.append(p.cpu().numpy())
        y_all.append(y.cpu().numpy())
        scar_all.append(scar.cpu().numpy())

    probs = np.concatenate(probs_all)
    y_np = np.concatenate(y_all)
    s_np = np.concatenate(scar_all)
    yhat = (probs >= 0.5).astype(int)

    acc = float((yhat == y_np).mean())
    majority_baseline = float(max((y_np == 0).mean(), (y_np == 1).mean()))
    threat_mask = (y_np == 1)
    minority_recall = float((yhat[threat_mask] == 1).mean()) if threat_mask.sum() > 0 else 0.0

    s1_mask, s0_mask = (s_np == 1), (s_np == 0)
    dp = float(abs(yhat[s1_mask].mean() - yhat[s0_mask].mean())) if (s1_mask.sum() and s0_mask.sum()) else 0.0

    def eo_rates(g):
        idx = (s_np == g)
        if not idx.any():
            return 0.0, 0.0
        yy, yh = y_np[idx], yhat[idx]
        tp = ((yh == 1) & (yy == 1)).sum()
        fn = ((yh == 0) & (yy == 1)).sum()
        fp = ((yh == 1) & (yy == 0)).sum()
        tn = ((yh == 0) & (yy == 0)).sum()
        return tp / max(tp + fn, 1), fp / max(fp + tn, 1)

    tpr1, fpr1 = eo_rates(1)
    tpr0, fpr0 = eo_rates(0)
    eo_max = float(max(abs(tpr1 - tpr0), abs(fpr1 - fpr0)))
    cf_gap = float(cf_abs_sum / max(cf_count, 1))
    mean_latent_diff = float(latent_diff_sum / max(cf_count, 1))

    return {
        "acc": acc,
        "dp_abs": dp,
        "eo_max_gap": eo_max,
        "cf_gap": cf_gap,
        "latent_diff": mean_latent_diff,
        "majority_baseline": majority_baseline,
        "minority_recall": minority_recall,
    }


def restratify_test_frame(df_test: pd.DataFrame, rho_target: float, seed: int = 42) -> pd.DataFrame:
    res = df_test.copy().reset_index(drop=True)
    res["scar"] = 0
    for threat, prob in ((1, rho_target), (0, 1.0 - rho_target)):
        indices = res.index[res["threat"] == threat].tolist()
        count = int(round(len(indices) * prob))
        ordered = sorted(
            indices,
            key=lambda idx: int(hashlib.sha256(f"{seed}|{threat}|{idx}".encode()).hexdigest()[:8], 16),
        )
        res.loc[ordered[:count], "scar"] = 1

    res["image_path"] = np.where(res["scar"] == 1, res["scarred_path"], res["clean_path"])
    res["counterfactual_image_path"] = np.where(res["scar"] == 1, res["clean_path"], res["scarred_path"])
    return res


def evaluate_regime(model, df_regime, device, phys_mu, phys_sigma, batch_size=64, autonomous: bool = True):
    temp_csv = PROJECT_ROOT / "outputs" / f"temp_eval_rcmf_{int(time.time()*1000)%100000}_{os.getpid()}.csv"
    df_regime.to_csv(temp_csv, index=False)
    try:
        ds = MultimodalCSVDatasetWithCF(str(temp_csv), verbose=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_samples, pin_memory=(device.type == "cuda"))
        return eval_rcmf_metrics(model, loader, device, phys_mu, phys_sigma, autonomous=autonomous)
    finally:
        if temp_csv.exists():
            temp_csv.unlink()


def main():
    parser = argparse.ArgumentParser("Train EQUITAS-RCMF with Orthogonal Subspace Disentanglement.")
    parser.add_argument("--csv", default="data/publishable_scar_production/multimodal_publishable.csv")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--vision_backbone", type=str, default="mobilenet_v3_small")

    # Architectural Hyperparameters
    parser.add_argument("--d_causal", type=int, default=192)
    parser.add_argument("--d_confounder", type=int, default=64)
    parser.add_argument("--initial_kappa", type=float, default=1.5)

    # Disentanglement & Invariance Penalties
    parser.add_argument("--lambda_confounder", type=float, default=0.5)
    parser.add_argument("--lambda_causal_inv", type=float, default=1.0)
    parser.add_argument("--lambda_latent_inv", type=float, default=0.5)
    parser.add_argument("--lambda_js", type=float, default=1.0)
    parser.add_argument("--lambda_dp", type=float, default=0.5)
    parser.add_argument("--lambda_eo", type=float, default=0.5)

    # Annealing
    parser.add_argument("--anneal_warmup_epochs", type=int, default=3)
    parser.add_argument("--anneal_duration_epochs", type=int, default=12)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Starting EQUITAS-RCMF Training on %s", device)

    csv_path = Path(PROJECT_ROOT / args.csv)
    
    # Pre-registered SHA256 Integrity Lock
    PREREGISTERED_HASH = "2035777f957fa4c5cc830989f4fe40fb9190589f17c567cb29a3ef0e6b2acae2"
    with open(csv_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    if file_hash != PREREGISTERED_HASH:
        raise ValueError(f"FATAL: Dataset integrity mismatch! Expected {PREREGISTERED_HASH}, got {file_hash}")
    LOGGER.info("Dataset SHA-256 Integrity Verified.")
    
    ds = MultimodalCSVDatasetWithCF(str(csv_path))

    train_idx = ds.df.index[ds.df["split"] == "train"].tolist()
    val_idx = ds.df.index[ds.df["split"] == "val"].tolist()
    test_idx = ds.df.index[ds.df["split"] == "test"].tolist()

    LOGGER.info("Splits: Train=%d, Val=%d, Test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Physiology Normalization
    X = ds.df.iloc[train_idx][ds.phys_cols].to_numpy(dtype=np.float32)
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    phys_mu = torch.tensor(mu, device=device, dtype=torch.float32).unsqueeze(0)
    phys_sigma = torch.tensor(sigma, device=device, dtype=torch.float32).unsqueeze(0)

    # Group balanced sampler
    sc, yy = ds.df.iloc[train_idx]["scar"].to_numpy(int), ds.df.iloc[train_idx]["threat"].to_numpy(int)
    gid = 2 * sc + yy
    counts = np.bincount(gid, minlength=4).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weights = torch.tensor(1.0 / counts[gid], dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_samples,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        Subset(ds, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize EQUITAS-RCMF Model
    model = EquitasRCMFModel(
        phys_dim=len(ds.phys_cols),
        vision_backbone=args.vision_backbone,
        d_causal=args.d_causal,
        d_confounder=args.d_confounder,
        num_classes=2,
        initial_kappa=args.initial_kappa,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    scaler, amp_ctx = make_amp(device, enabled=args.amp)

    best_score = -1e9
    best_ckpt_path = OUT_CKPT / "equitas_rcmf_master_best.pt"

    LOGGER.info("Initial Stiefel Orthogonality: %.2e", model.stiefel_decomp.verify_mutual_orthogonality())

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        anneal_factor = get_annealing_factor(epoch, args.anneal_warmup_epochs, args.anneal_duration_epochs)
        cur_lambda_dp = args.lambda_dp * anneal_factor
        cur_lambda_eo = args.lambda_eo * anneal_factor

        for step, b in enumerate(pbar, start=1):
            img = b["img"].to(device)
            img_cf = b["img_cf"].to(device)
            phys = b["phys"].to(device)
            y = b["y"].to(device)
            scar = b["scar"].to(device).float()
            has_cf = b["has_cf"].to(device).bool()
            mask = b["mask"].to(device)

            phys = (phys - phys_mu) / phys_sigma

            opt.zero_grad(set_to_none=True)

            with amp_ctx():
                # Factual Forward with privileged mask
                effective_mask_factual = mask * scar.view(-1, 1, 1, 1)
                out = model(img, phys, mask=effective_mask_factual, scar_label=scar)
                loss_task = ce_loss(out.logits, y)

                # 1. Confounder Subspace Supervision
                confounder_pred = model.confounder_head(out.v_confounder).squeeze(1)
                loss_confounder = bce_loss(confounder_pred, scar)

                # 2. Counterfactual Forward (for invariance)
                loss_causal_inv = torch.tensor(0.0, device=device)
                loss_latent_inv = torch.tensor(0.0, device=device)
                loss_js = torch.tensor(0.0, device=device)

                if has_cf.any():
                    effective_mask_cf = mask * (1.0 - scar.view(-1, 1, 1, 1))
                    out_cf = model(img_cf, phys, mask=effective_mask_cf, scar_label=(1.0 - scar))

                    # A. Causal Subspace Invariance: v_causal must be invariant between clean & scarred
                    diff_causal = F.mse_loss(out.v_causal[has_cf], out_cf.v_causal[has_cf])
                    loss_causal_inv = diff_causal

                    # B. Latent Manifold Isometric Invariance: Z must be invariant
                    diff_latent = F.mse_loss(out.latent_z[has_cf], out_cf.latent_z[has_cf])
                    loss_latent_inv = diff_latent

                    # C. Output Logit JS Divergence
                    js = js_divergence_stable(out.logits, out_cf.logits)
                    loss_js = js[has_cf].mean()

                # 3. Fairness Regularization
                p1 = F.softmax(out.logits, dim=1)[:, 1]
                loss_dp = dp_gap_prob(p1, scar.long())
                loss_eo = eo_gap_prob(p1, y, scar.long())

                # Total Multi-Objective Functional
                loss = (
                    loss_task
                    + args.lambda_confounder * loss_confounder
                    + args.lambda_causal_inv * loss_causal_inv
                    + args.lambda_latent_inv * loss_latent_inv
                    + args.lambda_js * loss_js
                    + cur_lambda_dp * loss_dp
                    + cur_lambda_eo * loss_eo
                )

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            pbar.set_postfix(
                task=f"{loss_task.item():.3f}",
                dp=f"{loss_dp.item():.3f}",
                cinv=f"{loss_causal_inv.item():.3f}",
                gate=f"{out.gate.mean().item():.2f}",
                kappa=f"{model.gate_engine.kappa.item():.2f}",
            )

        scheduler.step()

        # Validation Check in Autonomous Mode (no mask)
        val = eval_rcmf_metrics(model, val_loader, device, phys_mu, phys_sigma, autonomous=True)
        ortho_dev = model.stiefel_decomp.verify_mutual_orthogonality()

        if val["acc"] < val["majority_baseline"] + 0.05 or val["minority_recall"] < 0.10:
            score = -999.0
            LOGGER.info("[Epoch %d] DEGENERATE - Acc=%.4f, Rec=%.4f (Skipping)", epoch, val["acc"], val["minority_recall"])
        else:
            # Multi-criterion score: prioritize accuracy while penalizing DP, EO, and CF gaps
            score = val["acc"] - 0.5 * val["dp_abs"] - 0.5 * val["eo_max_gap"] - 0.2 * val["cf_gap"]
            LOGGER.info(
                "[Epoch %d] Score=%.4f | Acc=%.4f (Base=%.2f) | DP=%.4f | EO=%.4f | CF=%.4f | Ortho=%.1e",
                epoch, score, val["acc"], val["majority_baseline"], val["dp_abs"], val["eo_max_gap"], val["cf_gap"], ortho_dev
            )

        if score != -999.0 and score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_ckpt_path)
            LOGGER.info("  -> Saved NEW BEST Checkpoint to %s", best_ckpt_path)

    # If no non-degenerate checkpoint was found, save final epoch as best
    if not best_ckpt_path.exists():
        torch.save(model.state_dict(), best_ckpt_path)
        LOGGER.warning("No checkpoint surpassed degenerate threshold; saved final epoch.")

    LOGGER.info("=========================================================")
    LOGGER.info("TRAINING COMPLETE. Running 3-Regime OOD Benchmark...")
    LOGGER.info("=========================================================")

    model.load_state_dict(torch.load(str(best_ckpt_path), map_location=device, weights_only=True))
    model.eval()

    full_df = pd.read_csv(csv_path)
    test_df = full_df[full_df["split"] == "test"].copy().reset_index(drop=True)

    regimes = {
        "Biased In-Distribution (rho=0.85)": restratify_test_frame(test_df, 0.85, seed=args.seed),
        "Unbiased Neutral (rho=0.50)": restratify_test_frame(test_df, 0.50, seed=args.seed),
        "Inverted Adversarial (rho=0.15)": restratify_test_frame(test_df, 0.15, seed=args.seed),
    }

    # 1. Primary Autonomous Mode Evaluation (Edge/Production ready, no masks)
    results_autonomous = {}
    for r_name, df_reg in regimes.items():
        m = evaluate_regime(model, df_reg, device, phys_mu, phys_sigma, batch_size=args.batch_size, autonomous=True)
        results_autonomous[r_name] = m

    # 2. Privileged Mode Evaluation (with masks, for scientific comparison)
    results_privileged = {}
    for r_name, df_reg in regimes.items():
        m = evaluate_regime(model, df_reg, device, phys_mu, phys_sigma, batch_size=args.batch_size, autonomous=False)
        results_privileged[r_name] = m

    # Subgroup Demographic Audit (Autonomous Mode)
    df_unbiased = regimes["Unbiased Neutral (rho=0.50)"]
    demographics = {}

    for g_val, g_label in ((0, "Female"), (1, "Male")):
        sub_df = df_unbiased[df_unbiased["estimated_gender"] == g_val]
        if len(sub_df) > 0:
            demographics[f"Gender_{g_label}"] = evaluate_regime(model, sub_df, device, phys_mu, phys_sigma, args.batch_size, autonomous=True)

    age_bins = [
        ("Age_18_30", lambda d: (d["estimated_age"] >= 18) & (d["estimated_age"] < 30)),
        ("Age_30_45", lambda d: (d["estimated_age"] >= 30) & (d["estimated_age"] < 45)),
        ("Age_45_65", lambda d: (d["estimated_age"] >= 45) & (d["estimated_age"] <= 65)),
    ]
    for a_name, cond in age_bins:
        sub_df = df_unbiased[cond(df_unbiased)]
        if len(sub_df) > 0:
            demographics[a_name] = evaluate_regime(model, sub_df, device, phys_mu, phys_sigma, args.batch_size, autonomous=True)

    # Save Complete Results
    report = {
        "model": "EQUITAS-RCMF (Master Model)",
        "d_causal": args.d_causal,
        "d_confounder": args.d_confounder,
        "best_score": best_score,
        "autonomous_evaluations": results_autonomous,
        "privileged_evaluations": results_privileged,
        "demographic_evaluations": demographics,
        "stiefel_orthogonality": model.stiefel_decomp.verify_mutual_orthogonality(),
        "checkpoint_path": str(best_ckpt_path.absolute()),
        "checkpoint_hash": hashlib.sha256(best_ckpt_path.read_bytes()).hexdigest() if best_ckpt_path.exists() else None,
    }
    report_file = OUT_REP / "equitas_rcmf_master_benchmark_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print Final Thesis Results Table
    print("\n" + "=" * 85)
    print("      EQUITAS-RCMF MASTER MODEL: AUTONOMOUS CAUSAL INVARIANCE (NO MASKS)")
    print("=" * 85)
    print(f"{'Evaluation Regime':<35} | {'Accuracy':<9} | {'DP Gap':<8} | {'EO Gap':<8} | {'CF Gap':<8}")
    print("-" * 85)
    for r_name, m in results_autonomous.items():
        print(f"{r_name:<35} | {m['acc']*100:>7.2f}% | {m['dp_abs']:>8.4f} | {m['eo_max_gap']:>8.4f} | {m['cf_gap']:>8.4f}")
    print("-" * 85)

    print("\n" + "=" * 85)
    print("      EQUITAS-RCMF MASTER MODEL: PRIVILEGED MODE (GROUND TRUTH MASKS)")
    print("=" * 85)
    print(f"{'Evaluation Regime':<35} | {'Accuracy':<9} | {'DP Gap':<8} | {'EO Gap':<8} | {'CF Gap':<8}")
    print("-" * 85)
    for r_name, m in results_privileged.items():
        print(f"{r_name:<35} | {m['acc']*100:>7.2f}% | {m['dp_abs']:>8.4f} | {m['eo_max_gap']:>8.4f} | {m['cf_gap']:>8.4f}")
    print("-" * 85)

    print("\n" + "=" * 85)
    print("      EQUITAS-RCMF: SUBGROUP DEMOGRAPHIC AUDIT (rho=0.50, Autonomous)")
    print("=" * 85)
    print(f"{'Subgroup':<25} | {'Accuracy':<9} | {'DP Gap':<8} | {'EO Gap':<8} | {'CF Gap':<8}")
    print("-" * 85)
    for s_name, m in demographics.items():
        print(f"{s_name:<25} | {m['acc']*100:>7.2f}% | {m['dp_abs']:>8.4f} | {m['eo_max_gap']:>8.4f} | {m['cf_gap']:>8.4f}")
    print("-" * 85)


if __name__ == "__main__":
    main()
