#!/usr/bin/env python3
"""
Master Execution Suite for Thesis Production Benchmark.

Orchestrates:
1. Baseline (Concat) Multimodal Model Training (35 epochs)
2. Causal Gated Fusion (CGF Fair) Multimodal Model Training (35 epochs)
3. Multi-Regime Out-of-Distribution Causal Invariance Evaluation:
   - In-Distribution Biased (rho = 0.85)
   - Unbiased Benchmark (rho = 0.50)
   - Inverted Adversarial Shortcut (rho = 0.15)
4. Subgroup Demographic Audit (Age & Gender)
5. Publication-grade summary generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models_arch import MultimodalThreatModel
from train_cgf_fair import eval_metrics, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("ThesisBenchmark")


def restratify_test_frame(df_test: pd.DataFrame, rho_target: float, seed: int = 42) -> pd.DataFrame:
    """Restratifies scar labels for held-out test set according to rho_target."""
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


def evaluate_regime(
    model: torch.nn.Module,
    df_regime: pd.DataFrame,
    device: torch.device,
    phys_mu: torch.Tensor | None = None,
    phys_sigma: torch.Tensor | None = None,
    batch_size: int = 64,
) -> dict:
    temp_csv = PROJECT_ROOT / "outputs" / "temp_eval_regime.csv"
    temp_csv.parent.mkdir(parents=True, exist_ok=True)
    df_regime.to_csv(temp_csv, index=False)
    try:
        ds = MultimodalCSVDatasetWithCF(str(temp_csv), verbose=False)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_samples,
            pin_memory=(device.type == "cuda"),
        )
        metrics = eval_metrics(model, loader, device, phys_mu, phys_sigma)
        return metrics
    finally:
        if temp_csv.exists():
            temp_csv.unlink()


def run_training_command(cmd: list[str]) -> None:
    LOGGER.info("Executing: %s", " ".join(cmd))
    import subprocess
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    LOGGER.info("Completed in %.1f seconds (%.2f minutes)", elapsed, elapsed / 60.0)


def main():
    parser = argparse.ArgumentParser("Run complete thesis production benchmark.")
    parser.add_argument("--csv", default="data/publishable_scar_production/multimodal_publishable.csv")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_training", action="store_true", help="Skip training if checkpoints exist")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Starting Thesis Benchmark Suite on device: %s", device)

    py_exe = sys.executable
    train_script = str(PROJECT_ROOT / "src" / "train_cgf_fair.py")
    csv_path = str(PROJECT_ROOT / args.csv)

    ckpt_baseline = PROJECT_ROOT / "outputs" / "checkpoints" / f"counterfactual_concat_js_mobilenet_v3_small_{Path(args.csv).stem}_best_baseline_prod.pt"
    ckpt_cgf = PROJECT_ROOT / "outputs" / "checkpoints" / f"counterfactual_cgf_js_mobilenet_v3_small_{Path(args.csv).stem}_best_cgf_fair_prod.pt"

    # =========================================================================
    # Step 1: Train Baseline (Concat)
    # =========================================================================
    if not args.skip_training or not ckpt_baseline.exists():
        LOGGER.info("=========================================================")
        LOGGER.info("[PHASE 3.1] TRAINING BASELINE (CONCAT) - 35 EPOCHS")
        LOGGER.info("=========================================================")
        cmd_baseline = [
            py_exe, train_script,
            "--csv", csv_path,
            "--fusion", "concat",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--lambda_cf", "0.0",
            "--lambda_gate", "0.0",
            "--lambda_dp", "0.0",
            "--lambda_eo", "0.0",
            "--w_dp", "0.0",
            "--w_eo", "0.0",
            "--w_cf", "0.0",
            "--amp",
            "--out_suffix", "baseline_prod",
        ]
        run_training_command(cmd_baseline)
    else:
        LOGGER.info("Baseline checkpoint found at %s, skipping training.", ckpt_baseline)

    # =========================================================================
    # Step 2: Train CGF Fair (Ours)
    # =========================================================================
    if not args.skip_training or not ckpt_cgf.exists():
        LOGGER.info("=========================================================")
        LOGGER.info("[PHASE 3.2] TRAINING CAUSAL GATED FUSION (CGF FAIR) - 35 EPOCHS")
        LOGGER.info("=========================================================")
        cmd_cgf = [
            py_exe, train_script,
            "--csv", csv_path,
            "--fusion", "cgf",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--lambda_cf", "1.0",
            "--lambda_gate", "0.05",
            "--lambda_dp", "0.5",
            "--lambda_eo", "0.5",
            "--anneal_warmup_epochs", "3",
            "--anneal_duration_epochs", "12",
            "--w_dp", "1.0",
            "--w_eo", "1.0",
            "--w_cf", "0.2",
            "--zscore_phys",
            "--balance_groups",
            "--amp",
            "--out_suffix", "cgf_fair_prod",
        ]
        run_training_command(cmd_cgf)
    else:
        LOGGER.info("CGF Fair checkpoint found at %s, skipping training.", ckpt_cgf)

    # =========================================================================
    # Step 3: Out-of-Distribution Multi-Regime Evaluation
    # =========================================================================
    LOGGER.info("=========================================================")
    LOGGER.info("[PHASE 4] OUT-OF-DISTRIBUTION CAUSAL INVARIANCE EVALUATION")
    LOGGER.info("=========================================================")
    full_df = pd.read_csv(csv_path)
    test_df = full_df[full_df["split"] == "test"].copy().reset_index(drop=True)
    train_df = full_df[full_df["split"] == "train"].copy().reset_index(drop=True)

    # Calculate physiology normalization parameters from train split
    X_train = train_df[["hrv", "gsr"]].to_numpy(dtype=np.float32)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    phys_mu = torch.tensor(mu, device=device, dtype=torch.float32).unsqueeze(0)
    phys_sigma = torch.tensor(sigma, device=device, dtype=torch.float32).unsqueeze(0)

    # Instantiate models
    model_baseline = MultimodalThreatModel(phys_dim=2, fusion="concat").to(device)
    model_baseline.load_state_dict(torch.load(str(ckpt_baseline), map_location=device))
    model_baseline.eval()

    model_cgf = MultimodalThreatModel(phys_dim=2, fusion="cgf").to(device)
    model_cgf.load_state_dict(torch.load(str(ckpt_cgf), map_location=device))
    model_cgf.eval()

    regimes = {
        "Biased In-Distribution (rho=0.85)": restratify_test_frame(test_df, 0.85, seed=args.seed),
        "Unbiased Neutral (rho=0.50)": restratify_test_frame(test_df, 0.50, seed=args.seed),
        "Inverted Adversarial (rho=0.15)": restratify_test_frame(test_df, 0.15, seed=args.seed),
    }

    results = {}
    for r_name, df_reg in regimes.items():
        LOGGER.info("Evaluating regime: %s ...", r_name)
        base_metrics = evaluate_regime(model_baseline, df_reg, device, phys_mu=None, phys_sigma=None, batch_size=args.batch_size)
        cgf_metrics = evaluate_regime(model_cgf, df_reg, device, phys_mu=phys_mu, phys_sigma=phys_sigma, batch_size=args.batch_size)
        results[r_name] = {
            "baseline": base_metrics,
            "cgf_fair": cgf_metrics,
        }

    # =========================================================================
    # Step 4: Subgroup Demographic Audits (Unbiased Regime rho=0.50)
    # =========================================================================
    df_unbiased = regimes["Unbiased Neutral (rho=0.50)"]
    demographics = {}

    # Gender Subgroups
    for g_val, g_label in ((0, "Female"), (1, "Male")):
        sub_df = df_unbiased[df_unbiased["estimated_gender"] == g_val]
        if len(sub_df) > 0:
            b_m = evaluate_regime(model_baseline, sub_df, device, None, None, args.batch_size)
            c_m = evaluate_regime(model_cgf, sub_df, device, phys_mu, phys_sigma, args.batch_size)
            demographics[f"Gender_{g_label}"] = {"baseline": b_m, "cgf_fair": c_m, "count": len(sub_df)}

    # Age Subgroups
    age_bins = [
        ("Age_18_30", lambda d: (d["estimated_age"] >= 18) & (d["estimated_age"] < 30)),
        ("Age_30_45", lambda d: (d["estimated_age"] >= 30) & (d["estimated_age"] < 45)),
        ("Age_45_65", lambda d: (d["estimated_age"] >= 45) & (d["estimated_age"] <= 65)),
    ]
    for a_name, cond in age_bins:
        sub_df = df_unbiased[cond(df_unbiased)]
        if len(sub_df) > 0:
            b_m = evaluate_regime(model_baseline, sub_df, device, None, None, args.batch_size)
            c_m = evaluate_regime(model_cgf, sub_df, device, phys_mu, phys_sigma, args.batch_size)
            demographics[a_name] = {"baseline": b_m, "cgf_fair": c_m, "count": len(sub_df)}

    # Save comprehensive report
    report_data = {
        "dataset": args.csv,
        "test_samples": len(test_df),
        "regime_evaluations": results,
        "demographic_evaluations": demographics,
    }

    report_path = PROJECT_ROOT / "outputs" / "reports" / "thesis_production_benchmark_report.json"
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    LOGGER.info("Report saved to %s", report_path)

    # Print publication summary table
    print("\n" + "=" * 80)
    print("        THESIS PRODUCTION BENCHMARK: CAUSAL INVARIANCE AUDIT")
    print("=" * 80)
    header = f"{'Evaluation Regime':<35} | {'Model':<12} | {'Accuracy':<9} | {'DP Gap':<8} | {'EO Gap':<8} | {'CF Gap':<8}"
    print(header)
    print("-" * 80)

    for r_name, r_data in results.items():
        b = r_data["baseline"]
        c = r_data["cgf_fair"]
        print(f"{r_name:<35} | {'Baseline':<12} | {b['acc']*100:>7.2f}% | {b['dp_abs']:>8.4f} | {b['eo_max_gap']:>8.4f} | {b['cf_gap']:>8.4f}")
        print(f"{'':<35} | {'CGF (Ours)':<12} | {c['acc']*100:>7.2f}% | {c['dp_abs']:>8.4f} | {c['eo_max_gap']:>8.4f} | {c['cf_gap']:>8.4f}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("        SUBGROUP DEMOGRAPHIC AUDIT (UNBIASED BENCHMARK rho=0.50)")
    print("=" * 80)
    print(f"{'Subgroup':<25} | {'Count':<6} | {'Model':<12} | {'Accuracy':<9} | {'DP Gap':<8} | {'EO Gap':<8}")
    print("-" * 80)
    for g_name, g_data in demographics.items():
        b = g_data["baseline"]
        c = g_data["cgf_fair"]
        cnt = g_data["count"]
        print(f"{g_name:<25} | {cnt:<6} | {'Baseline':<12} | {b['acc']*100:>7.2f}% | {b['dp_abs']:>8.4f} | {b['eo_max_gap']:>8.4f}")
        print(f"{'':<25} | {'':<6} | {'CGF (Ours)':<12} | {c['acc']*100:>7.2f}% | {c['dp_abs']:>8.4f} | {c['eo_max_gap']:>8.4f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
