import subprocess
import os

CSV_UNBIASED = "data/csv/multimodal_10k_unbiased.csv"
CSV_BIASED = "data/csv/multimodal_10k.csv"
SPLIT_FILE = "data/csv/multimodal_10k_strict_split_seed42.json"
EPOCHS = 50

# Ensure strict failure
os.environ["PYTHONUNBUFFERED"] = "1"

runs = [
    {
        "name": "1. Baseline",
        "env": {"EQUITAS_DISABLE_STIEFEL": "1"},
        "args": [
            "--fusion", "concat",
            "--lambda_cf", "0.0", "--lambda_gate", "0.0", "--lambda_dp", "0.0", "--lambda_eo", "0.0",
            "--w_cf", "0.0", "--w_dp", "0.0", "--w_eo", "0.0",
            "--out_suffix", "strict_baseline"
        ]
    },
    {
        "name": "2. Naive Concat + Stiefel",
        "env": {"EQUITAS_DISABLE_STIEFEL": "0"},
        "args": [
            "--fusion", "concat",
            "--lambda_cf", "1.0", "--lambda_gate", "0.0", "--lambda_dp", "1.0", "--lambda_eo", "1.0",
            "--w_cf", "0.2", "--w_dp", "1.0", "--w_eo", "1.0",
            "--out_suffix", "strict_concat"
        ]
    },
    {
        "name": "3. V4 Software Penalty (No Stiefel)",
        "env": {"EQUITAS_DISABLE_STIEFEL": "1"},
        "args": [
            "--fusion", "cgf",
            "--lambda_cf", "1.0", "--lambda_gate", "0.1", "--lambda_dp", "1.0", "--lambda_eo", "1.0",
            "--w_cf", "0.2", "--w_dp", "1.0", "--w_eo", "1.0",
            "--out_suffix", "strict_v4"
        ]
    },
    {
        "name": "4. Stiefel Geometric Lock (Flagship)",
        "env": {"EQUITAS_DISABLE_STIEFEL": "0"},
        "args": [
            "--fusion", "cgf",
            "--lambda_cf", "1.0", "--lambda_gate", "0.1", "--lambda_dp", "1.0", "--lambda_eo", "1.0",
            "--w_cf", "0.2", "--w_dp", "1.0", "--w_eo", "1.0",
            "--out_suffix", "strict_stiefel"
        ]
    },
    {
        "name": "5. Pure Counterfactual",
        "env": {"EQUITAS_DISABLE_STIEFEL": "0"},
        "args": [
            "--fusion", "cgf",
            "--lambda_cf", "1.0", "--lambda_gate", "0.1", "--lambda_dp", "0.0", "--lambda_eo", "0.0",
            "--w_cf", "0.2", "--w_dp", "0.0", "--w_eo", "0.0",
            "--out_suffix", "strict_purecf"
        ]
    },
    {
        "name": "6. Stiefel-Only Zero-Loss",
        "env": {"EQUITAS_DISABLE_STIEFEL": "0"},
        "args": [
            "--fusion", "cgf",
            "--lambda_cf", "0.0", "--lambda_gate", "0.0", "--lambda_dp", "0.0", "--lambda_eo", "0.0",
            "--w_cf", "0.0", "--w_dp", "0.0", "--w_eo", "0.0",
            "--out_suffix", "strict_stiefel_zeroloss"
        ]
    }
]

for run in runs:
    print(f"\n{'='*80}\nSTARTING RUN: {run['name']}\n{'='*80}")
    
    cmd = [
        "python", "src/train_cgf_fair.py",
        "--csv", CSV_UNBIASED,
        "--csv_biased", CSV_BIASED,
        "--split_file", SPLIT_FILE,
        "--epochs", str(EPOCHS),
        "--batch_size", "64", # Ensure we don't OOM or hang
    ] + run["args"]
    
    env = os.environ.copy()
    env.update(run["env"])
    
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"FATAL ERROR during {run['name']}. Aborting sequence.")
        exit(1)

print("\nALL 6 RUNS COMPLETED SUCCESSFULLY.")
