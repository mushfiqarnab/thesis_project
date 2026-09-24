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
    print("Initiating THE PURE ACCEPTANCE SWEEP (Edge Architecture Only)...")
    
    # Pure Edge-Deployment Constraints
    BACKBONE = "mobilenet_v3_small"
    EPOCHS = "250"
    BATCH = "32" 
    LR = "1e-4"
    
    # We substitute parameter bloat with massive empirical rigor.
    SEEDS = ["42", "100", "2026", "777", "888"] 

    for seed in SEEDS:
        print(f"\n\n>>> COMMENCING EVALUATION FOR SEED: {seed} <<<")
        
        # Model A: Naive ERM Baseline (Proves the capacity to cheat exists)
        cmd_a = [
            PYTHON, "src/train_baseline.py",
            "--csv", CSV_PATH,
            "--suffix", f"naive_erm_baseline_mobilenet_seed{seed}",
            "--vision_backbone", BACKBONE,
            "--epochs", EPOCHS,
            "--batch_size", BATCH,
            "--lr", LR,
            "--seed", seed
        ]
        
        # Model B: Camera-Off Baseline
        cmd_b = [
            PYTHON, "src/train_baseline.py",
            "--csv", CSV_PATH,
            "--suffix", f"camera_off_baseline_mobilenet_seed{seed}",
            "--vision_backbone", BACKBONE,
            "--camera_off",
            "--epochs", EPOCHS,
            "--batch_size", BATCH,
            "--lr", LR,
            "--seed", seed
        ]
        
        # Model C: CGP Pruned Edge Model
        cmd_c = [
            PYTHON, "src/train_cgf_fair.py",
            "--csv", CSV_PATH,
            "--out_suffix", f"_cgp_pruned_mobilenet_seed{seed}",
            "--backbone", BACKBONE,
            "--epochs", EPOCHS,
            "--batch_size", BATCH,
            "--lr", LR,
            "--seed", seed
        ]
        
        # Model D: EQUITAS-RCMF Master (Proves the causal isolation works)
        cmd_d = [
            PYTHON, "src/train_equitas_rcmf.py",
            "--csv", CSV_PATH,
            "--vision_backbone", BACKBONE,
            "--epochs", EPOCHS,
            "--batch_size", BATCH,
            "--lr", LR,
            "--seed", seed
        ]

        run_command(cmd_a, f"Model A (Naive ERM) [Seed {seed}]")
        run_command(cmd_b, f"Model B (Camera-Off) [Seed {seed}]")
        run_command(cmd_c, f"Model C (CGP Pruned Edge) [Seed {seed}]")
        run_command(cmd_d, f"Model D (EQUITAS-RCMF) [Seed {seed}]")
    
    print("\n" + "="*80)
    print("ALL EDGE TRAINING COMPLETE! Matrix reports saved to outputs/reports/")
    print("="*80)

if __name__ == "__main__":
    main()
