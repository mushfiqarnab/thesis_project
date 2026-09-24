"""
prune_eqarnb.py
================================================================================
Fair Pruning & Sparsity Audit Suite for EQARNB
Investigating the Hooker et al. (NeurIPS 2019) Compression-Fairness Hypothesis:
"Does Stiefel Orthogonal Subspace Disentanglement shield the model from
pruning-induced fairness degradation under heavy parameter sparsity?"
================================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models.equitas_rcmf import EquitasRCMFModel
from train_equitas_rcmf import eval_rcmf_metrics, evaluate_regime, restratify_test_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("FairPruning")


def apply_selective_pruning(model: nn.Module, amount: float, prune_stiefel: bool = False) -> nn.Module:
    """
    Applies L1 unstructured magnitude pruning to linear and projection layers.
    If prune_stiefel is False, preserves the Riemannian Stiefel orthogonal projection
    manifold to maintain strict algebraic independence W_c @ W_b^T = 0.
    """
    pruned_model = copy.deepcopy(model)
    pruned_layers = []

    for name, m in pruned_model.named_modules():
        if isinstance(m, nn.Linear):
            # Do not prune Stiefel raw matrix if preserving strict orthogonality
            if not prune_stiefel and "stiefel_decomp" in name:
                continue
            prune.l1_unstructured(m, name="weight", amount=amount)
            pruned_layers.append((m, "weight", name))

    # Make pruning permanent (removes mask hooks, zeroes out weights)
    for m, name, layer_path in pruned_layers:
        if hasattr(m, "weight_orig"):
            prune.remove(m, name)

    return pruned_model


def measure_model_sparsity(model: nn.Module) -> Tuple[int, int, float]:
    total_params = 0
    zero_params = 0
    for p in model.parameters():
        total_params += p.numel()
        zero_params += int((p == 0).sum().item())
    sparsity = float(zero_params / max(total_params, 1))
    return total_params, zero_params, sparsity


def run_pruning_audit(
    ckpt_path: Path,
    csv_path: Path,
    sparsity_levels: list[float] = [0.15, 0.30, 0.45],
):
    print("=" * 80)
    print("      EQARNB: FAIR PRUNING & RIEMANNIAN SPARSITY AUDIT")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device for Pruning Audit: %s", device)

    # 1. Load Golden Master Model
    model = EquitasRCMFModel(
        phys_dim=2,
        vision_backbone="mobilenet_v3_small",
        d_causal=192,
        d_confounder=64,
        num_classes=2,
    ).to(device)

    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Precompute normalization statistics
    full_df = pd.read_csv(csv_path)
    train_df = full_df[full_df["split"] == "train"].copy()
    test_df = full_df[full_df["split"] == "test"].copy().reset_index(drop=True)

    phys_cols = ["hrv", "gsr"] if "hrv" in train_df.columns else ["hrv_rmssd", "gsr_mean"]
    X = train_df[phys_cols].to_numpy(dtype=np.float32)
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    phys_mu = torch.tensor(mu, device=device, dtype=torch.float32).unsqueeze(0)
    phys_sigma = torch.tensor(sigma, device=device, dtype=torch.float32).unsqueeze(0)

    # Prepare OOD evaluation frames
    regimes = {
        "Biased In-Distribution (rho=0.85)": restratify_test_frame(test_df, 0.85, seed=42),
        "Unbiased Neutral (rho=0.50)": restratify_test_frame(test_df, 0.50, seed=42),
        "Inverted Adversarial (rho=0.15)": restratify_test_frame(test_df, 0.15, seed=42),
    }

    # Evaluate Unpruned Baseline
    LOGGER.info("Evaluating Unpruned Master Model (0% Sparsity)...")
    base_results = {}
    for r_name, df_reg in regimes.items():
        base_results[r_name] = evaluate_regime(model, df_reg, device, phys_mu, phys_sigma, batch_size=64, autonomous=True)

    audit_results = {
        "unpruned": {
            "sparsity_ratio": 0.0,
            "regimes": base_results,
            "orthogonality": model.stiefel_decomp.verify_mutual_orthogonality(),
        }
    }

    best_pruned_model = None
    best_pruned_amount = 0.0
    best_score = -1e9

    for amount in sparsity_levels:
        LOGGER.info("-" * 80)
        LOGGER.info("Auditing Pruning Level: %.1f%% Linear Sparsity...", amount * 100)
        
        pruned_model = apply_selective_pruning(model, amount=amount, prune_stiefel=False)
        total_p, zero_p, actual_sparsity = measure_model_sparsity(pruned_model)
        ortho_dev = pruned_model.stiefel_decomp.verify_mutual_orthogonality()

        LOGGER.info(
            "      Active Parameters: %d / %d (Effective Sparsity: %.2f%%)",
            total_p - zero_p, total_p, actual_sparsity * 100
        )
        LOGGER.info("      Preserved Stiefel Orthogonality: %.2e", ortho_dev)

        pruned_regime_results = {}
        for r_name, df_reg in regimes.items():
            pruned_regime_results[r_name] = evaluate_regime(
                pruned_model, df_reg, device, phys_mu, phys_sigma, batch_size=64, autonomous=True
            )

        audit_results[f"pruned_{int(amount*100)}pct"] = {
            "requested_amount": amount,
            "actual_sparsity": actual_sparsity,
            "total_params": total_p,
            "sparse_params": zero_p,
            "orthogonality": ortho_dev,
            "regimes": pruned_regime_results,
        }

        # Calculate retention score on neutral split
        neutral_m = pruned_regime_results["Unbiased Neutral (rho=0.50)"]
        score = neutral_m["acc"] - 0.5 * neutral_m["dp_abs"] - 0.5 * neutral_m["eo_max_gap"] - 0.2 * neutral_m["cf_gap"]

        if score > best_score:
            best_score = score
            best_pruned_model = pruned_model
            best_pruned_amount = amount

    # Save Best Pruned Checkpoint
    out_ckpt = PROJECT_ROOT / "outputs" / "checkpoints" / f"eqarnb_pruned_{int(best_pruned_amount*100)}_best.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_pruned_model.state_dict(), str(out_ckpt))
    LOGGER.info("Saved Optimal Fair-Pruned Checkpoint to %s", out_ckpt)

    # Save Audit Report
    report_path = PROJECT_ROOT / "outputs" / "reports" / "eqarnb_pruning_fairness_report.json"
    report_path.write_text(json.dumps(audit_results, indent=2), encoding="utf-8")
    LOGGER.info("Saved Fair Pruning Audit Report to %s", report_path)

    # Print Master Pruning Comparative Table
    print("\n" + "=" * 90)
    print("      EQARNB: FAIR PRUNING ABLATION TABLE (UNBIASED NEUTRAL rho=0.50)")
    print("=" * 90)
    print(f"{'Pruning Level':<20} | {'Sparsity':<10} | {'Accuracy':<9} | {'DP Gap':<8} | {'EO Gap':<8} | {'CF Gap':<8}")
    print("-" * 90)
    
    # Unpruned row
    u_m = audit_results["unpruned"]["regimes"]["Unbiased Neutral (rho=0.50)"]
    print(f"{'Unpruned FP32':<20} | {'0.0%':<10} | {u_m['acc']*100:>7.2f}% | {u_m['dp_abs']:>8.4f} | {u_m['eo_max_gap']:>8.4f} | {u_m['cf_gap']:>8.4f}")

    for amount in sparsity_levels:
        key = f"pruned_{int(amount*100)}pct"
        p_data = audit_results[key]
        p_m = p_data["regimes"]["Unbiased Neutral (rho=0.50)"]
        sp_str = f"{p_data['actual_sparsity']*100:.1f}%"
        print(f"{f'Pruned ({int(amount*100)}%)':<20} | {sp_str:<10} | {p_m['acc']*100:>7.2f}% | {p_m['dp_abs']:>8.4f} | {p_m['eo_max_gap']:>8.4f} | {p_m['cf_gap']:>8.4f}")
    print("-" * 90)

    # Adversarial robustness under pruning
    print("\n" + "=" * 90)
    print("      EQARNB: ADVERSARIAL STABILITY UNDER PRUNING (INVERTED rho=0.15)")
    print("=" * 90)
    print(f"{'Pruning Level':<20} | {'Accuracy':<9} | {'DP Gap':<8} | {'EO Gap':<8} | {'CF Gap':<8}")
    print("-" * 90)
    adv_u = audit_results["unpruned"]["regimes"]["Inverted Adversarial (rho=0.15)"]
    print(f"{'Unpruned FP32':<20} | {adv_u['acc']*100:>7.2f}% | {adv_u['dp_abs']:>8.4f} | {adv_u['eo_max_gap']:>8.4f} | {adv_u['cf_gap']:>8.4f}")
    for amount in sparsity_levels:
        key = f"pruned_{int(amount*100)}pct"
        adv_p = audit_results[key]["regimes"]["Inverted Adversarial (rho=0.15)"]
        print(f"{f'Pruned ({int(amount*100)}%)':<20} | {adv_p['acc']*100:>7.2f}% | {adv_p['dp_abs']:>8.4f} | {adv_p['eo_max_gap']:>8.4f} | {adv_p['cf_gap']:>8.4f}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Fair Pruning Audit on EQARNB.")
    parser.add_argument("--ckpt", default="outputs/checkpoints/equitas_rcmf_master_best.pt")
    parser.add_argument("--csv", default="data/publishable_scar_production/multimodal_publishable.csv")
    args = parser.parse_args()

    run_pruning_audit(
        ckpt_path=PROJECT_ROOT / args.ckpt,
        csv_path=PROJECT_ROOT / args.csv,
    )
