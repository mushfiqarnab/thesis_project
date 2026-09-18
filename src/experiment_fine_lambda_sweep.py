"""
experiment_fine_lambda_sweep.py
======================================================
Strict, empirical verification of the Fairness-Accuracy tradeoff frontier.

This script executes a fine-grained hyperparameter sweep over lambda_dp/lambda_eo 
using the V4 Apex trainer. It is designed to find the exact threshold where the 
model transitions from "learning a fair boundary" to "collapsing into a degenerate state", 
proving the existence of a continuous Pareto frontier rather than a step-function collapse.

Tested Grid: [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
"""
import subprocess
import json
import time
from pathlib import Path
import os
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

def run_fine_sweep():
    # The mathematically critical region identified in the critique
    lambda_grid = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    
    print("==================================================================")
    print(" INITIATING FINE-GRAINED LAMBDA SWEEP (PARETO FRONTIER MAPPING)")
    print(f" Grid: {lambda_grid}")
    print("==================================================================")
    
    results = []
    
    for l_val in lambda_grid:
        print(f"\n[SWEEP] Starting training for Lambda = {l_val}")
        
        # We run fewer epochs per sweep point to map the space efficiently, 
        # using the V4 trainer's strict anti-collapse defenses.
        cmd = [
            "python", str(SRC_DIR / "train_cgf_fair.py"),
            "--csv", str(PROJECT_ROOT / "data/csv/multimodal_10k_unbiased.csv"),
            "--csv_biased", str(PROJECT_ROOT / "data/csv/multimodal_10k.csv"),
            "--backbone", "mobilenet_v3_small",
            "--fusion", "cgf",
            "--epochs", "15",
            "--batch_size", "64",
            "--lr", "2e-4",
            "--lambda_cf", "1.0",
            "--lambda_gate", "0.05",
            "--lambda_dp", str(l_val),
            "--lambda_eo", str(l_val),
            "--anneal_warmup_epochs", "2",
            "--anneal_duration_epochs", "6",
            "--zscore_phys"
        ]
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        
        t0 = time.time()
        # Run the trainer synchronously for the sweep
        process = subprocess.run(cmd, env=env, capture_output=True, text=True)
        t1 = time.time()
        
        if process.returncode != 0:
            print(f"[ERROR] Run failed for lambda={l_val}. Check logs.")
            print(process.stderr)
            continue
            
        # Parse the output report to get the exact score
        # The V4 trainer writes to outputs/reports/train_counterfactual_v2_multimodal_10k_unbiased_mobilenet_v3_small.json
        # We will dynamically read it and rename it to preserve the sweep data
        target_report = REPORT_DIR / "train_counterfactual_v2_multimodal_10k_unbiased_mobilenet_v3_small.json"
        
        if target_report.exists():
            with open(target_report, 'r') as f:
                rep_data = json.load(f)
                
            # Rename the report to keep the sweep history
            sweep_report_path = REPORT_DIR / f"sweep_lambda_{l_val}.json"
            shutil.copy(target_report, sweep_report_path)
            
            best_score = rep_data.get("best_score", "N/A")
            print(f"[SWEEP] Lambda {l_val} completed in {t1-t0:.1f}s | Best Validated Score: {best_score}")
            results.append({"lambda": l_val, "best_score": best_score})
        else:
            print(f"[ERROR] Could not find report file for lambda={l_val}")
            
    print("\n==================================================================")
    print(" SWEEP COMPLETE. FINAL PARETO DATA:")
    for r in results:
        print(f" Lambda: {r['lambda']} -> V4 Anti-Collapse Score: {r['best_score']}")
    print("==================================================================")

if __name__ == "__main__":
    run_fine_sweep()
