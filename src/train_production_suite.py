import subprocess
import sys
import time
from pathlib import Path

# The pristine pre-registered multimodal CSV dataset
CSV_PATH = "data/publishable_scar_production/multimodal_publishable.csv"
PYTHON = r".venv_worldclass\Scripts\python.exe"

def run_command(cmd, name):
    print(f"\n{'='*80}")
    print(f"STARTING: {name}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    start_t = time.time()
    
    # Run process and stream output to stdout
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='', flush=True)
        
    process.wait()
    duration = time.time() - start_t
    
    print(f"\n{'-'*80}")
    if process.returncode == 0:
        print(f"COMPLETED: {name} in {duration:.1f} seconds")
    else:
        print(f"FAILED: {name} (Exit code: {process.returncode})")
        sys.exit(1)

def main():
    print("Initiating 4-Model Comparative Matrix Training (Production Suite)...")
    print(f"Dataset Lock: {CSV_PATH}")
    
    # Shared training hyperparameters for apple-to-apple comparison
    EPOCHS = "50"  # Solid hard training
    BATCH = "32"
    LR = "2e-4"
    SEED = "42"

    # Model A: Naive ERM Baseline
    cmd_a = [
        PYTHON, "src/train_baseline.py",
        "--csv", CSV_PATH,
        "--suffix", "naive_erm_baseline",
        "--epochs", EPOCHS,
        "--batch_size", BATCH,
        "--lr", LR,
        "--seed", SEED
    ]
    
    # Model B: Camera-Off Baseline (Physiology Only)
    cmd_b = [
        PYTHON, "src/train_baseline.py",
        "--csv", CSV_PATH,
        "--suffix", "camera_off_baseline",
        "--camera_off",
        "--epochs", EPOCHS,
        "--batch_size", BATCH,
        "--lr", LR,
        "--seed", SEED
    ]
    
    # Model C: CGP Pruned Edge Model (Causal Graph Fairness)
    cmd_c = [
        PYTHON, "src/train_cgf_fair.py",
        "--csv", CSV_PATH,
        "--out_suffix", "_cgp_pruned",
        "--epochs", EPOCHS,
        "--batch_size", BATCH,
        "--lr", LR,
        "--seed", SEED
    ]
    
    # Model D: EQUITAS-RCMF Master
    cmd_d = [
        PYTHON, "src/train_equitas_rcmf.py",
        "--csv", CSV_PATH,
        "--epochs", EPOCHS,
        "--batch_size", BATCH,
        "--lr", LR,
        "--seed", SEED
    ]

    # Execute Matrix
    run_command(cmd_a, "Model A (Naive ERM)")
    run_command(cmd_b, "Model B (Camera-Off Physiology)")
    run_command(cmd_c, "Model C (CGP Pruned Edge Model)")
    run_command(cmd_d, "Model D (EQUITAS-RCMF Master)")
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE! Matrix reports saved to outputs/reports/")
    print("="*80)

if __name__ == "__main__":
    main()
