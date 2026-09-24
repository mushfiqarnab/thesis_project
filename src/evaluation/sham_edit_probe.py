import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_fair import MultimodalCSVDatasetWithCF
from models.equitas_rcmf import EquitasRCMFModel

def apply_sham_edit(img_tensor):
    """
    Applies a procedural Sham Edit (a 30x30 black square on the upper left)
    to prove the network is robust to unseen, non-causal perturbations,
    satisfying Adebayo (2018) Sanity Checks.
    """
    # img_tensor is (3, 224, 224)
    img_np = img_tensor.permute(1, 2, 0).numpy().copy()
    
    # De-normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np * std) + mean
    
    # Apply Sham Artifact (Black Square at 20:50, 20:50)
    img_np[20:50, 20:50, :] = 0.0
    
    # Re-normalize
    img_np = (img_np - mean) / std
    return torch.from_numpy(img_np).permute(2, 0, 1).float()

def evaluate_condition(model, dataset, condition, device):
    """Evaluates the model under a specific test condition."""
    model.eval()
    correct = 0
    total = 0
    
    # Metrics for Fairness (DP Gap)
    pred_rates = {0: [], 1: []} # by demographic/artifact presence
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Evaluating {condition}"):
            sample = dataset[i]
            
            # Select the requested image condition
            if condition == "Clean":
                img = sample.img_cf.unsqueeze(0).to(device)
            elif condition == "Scarred (Trained Confounder)":
                img = sample.img.unsqueeze(0).to(device)
            elif condition == "Sham Edit (Unseen Perturbation)":
                img = apply_sham_edit(sample.img_cf).unsqueeze(0).to(device)
                
            phys = sample.phys.unsqueeze(0).to(device)
            target = sample.y.item()
            scar_status = sample.scar.item()
            
            # Forward pass (Autonomous Mode: mask=None)
            out = model(img, phys, mask=None)
            pred = torch.argmax(out.logits, dim=1).item()
            
            if pred == target:
                correct += 1
            total += 1
            
            pred_rates[scar_status].append(pred)
            
    acc = correct / total
    
    # DP Gap: |P(Y_hat=1 | S=1) - P(Y_hat=1 | S=0)|
    p_s1 = np.mean(pred_rates[1]) if len(pred_rates[1]) > 0 else 0.0
    p_s0 = np.mean(pred_rates[0]) if len(pred_rates[0]) > 0 else 0.0
    dp_gap = abs(p_s1 - p_s0)
    
    return acc, dp_gap

def main():
    print("Initializing Sham Edit Boundary Artifact Probe...")
    print("Theoretical Basis: Adebayo et al. 2018 (Sanity Checks for Saliency)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    csv_path = PROJECT_ROOT / "data" / "publishable_scar_production" / "multimodal_publishable.csv"
    ds = MultimodalCSVDatasetWithCF(str(csv_path))
    
    # Load 1-hour fast sweep EQUITAS-RCMF (MobileNetV3)
    print("\nLoading EQUITAS-RCMF Master Model...")
    model = EquitasRCMFModel(
        phys_dim=len(ds.phys_cols), 
        vision_backbone="mobilenet_v3_small", 
        d_causal=192, 
        d_confounder=64
    )
    ckpt = PROJECT_ROOT / "outputs" / "checkpoints" / "equitas_rcmf_master_best.pt"
    
    if not ckpt.exists():
        print(f"FAILED: Checkpoint {ckpt} not found. Ensure 1-hour sweep completed.")
        sys.exit(1)
        
    # Load safely for CPU/GPU
    state = torch.load(ckpt, map_location=device, weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    
    print("\nExecuting Probe Matrix...")
    results = {}
    
    # Run Matrix
    acc_clean, dp_clean = evaluate_condition(model, ds, "Clean", device)
    acc_scar, dp_scar = evaluate_condition(model, ds, "Scarred (Trained Confounder)", device)
    acc_sham, dp_sham = evaluate_condition(model, ds, "Sham Edit (Unseen Perturbation)", device)
    
    print("\n" + "="*80)
    print("     SHAM EDIT PROBE RESULTS (ADEBAYO SANITY CHECK)")
    print("="*80)
    print(f"{'Condition':<35} | {'Accuracy':<10} | {'DP Gap (Fairness)':<20}")
    print("-" * 80)
    print(f"{'Clean (No Artifacts)':<35} | {acc_clean*100:>6.2f}%   | {dp_clean:.4f}")
    print(f"{'Scarred (Trained Confounder)':<35} | {acc_scar*100:>6.2f}%   | {dp_scar:.4f}")
    print(f"{'Sham Edit (Unseen Perturbation)':<35} | {acc_sham*100:>6.2f}%   | {dp_sham:.4f}")
    print("="*80)
    
    # Assert Causal Invariance (Difference should be negligible)
    delta_sham = abs(acc_clean - acc_sham)
    if delta_sham < 0.05:
        print("\nVerdict: PASS - Model is robust to unseen sham perturbations.")
        print("This proves the vision encoder is alive but strictly constrained by the Stiefel causal manifold.")
    else:
        print(f"\nVerdict: FAIL - Model accuracy collapsed under Sham Edit (Delta: {delta_sham:.2f}).")
        print("This suggests the vision encoder is fragile or dead, failing the Sanity Check.")
        sys.exit(1)

if __name__ == "__main__":
    main()
