import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
import sys
import os
import warnings

warnings.filterwarnings('ignore')
sys.path.append('src')

from models.equitas_rcmf import EquitasRCMFModel
from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples

def evaluate_condition(model, loader, device, phys_mu, phys_sigma, condition='standard'):
    model.eval()
    probs_all, y_all, scar_all = [], [], []
    
    with torch.no_grad():
        for b in loader:
            img = b["img"].to(device)
            phys = b["phys"].to(device)
            y = b["y"].to(device)
            scar = b["scar"].to(device)
            
            if phys_mu is not None and phys_sigma is not None:
                phys = (phys - phys_mu) / phys_sigma
                
            if condition == 'k1':
                img = torch.zeros_like(img)
            elif condition == 'k2':
                phys = torch.zeros_like(phys)
                
            out = model(img, phys, mask=None)
            p = torch.softmax(out.logits, dim=1)[:, 1]
            
            probs_all.append(p.cpu().numpy())
            y_all.append(y.cpu().numpy())
            scar_all.append(scar.cpu().numpy())
            
    probs = np.concatenate(probs_all)
    y_np = np.concatenate(y_all)
    s_np = np.concatenate(scar_all)
    yhat = (probs >= 0.5).astype(int)
    
    acc = float((yhat == y_np).mean())
    
    s1_mask, s0_mask = (s_np == 1), (s_np == 0)
    p1 = (yhat[s1_mask] == 1).mean() if s1_mask.sum() > 0 else 0.0
    p0 = (yhat[s0_mask] == 1).mean() if s0_mask.sum() > 0 else 0.0
    dp_gap = abs(p1 - p0)
    
    return acc, dp_gap

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    csv_path = "data/publishable_scar_production/multimodal_publishable.csv"
    ds = MultimodalCSVDatasetWithCF(csv_path, verbose=False)
    
    train_df = ds.df[ds.df["split"] == "train"]
    X = train_df[ds.phys_cols].to_numpy(dtype=np.float32)
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    phys_mu = torch.tensor(mu, device=device, dtype=torch.float32).unsqueeze(0)
    phys_sigma = torch.tensor(sigma, device=device, dtype=torch.float32).unsqueeze(0)
    
    val_df = ds.df[ds.df["split"] == "val"]
    temp_csv = "temp_val2.csv"
    val_df.to_csv(temp_csv, index=False)
    val_ds = MultimodalCSVDatasetWithCF(temp_csv, verbose=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_samples)
    
    model = EquitasRCMFModel(phys_dim=2, num_classes=2, initial_kappa=1.5).to(device)
    
    checkpoint_path = "outputs/legacy_checkpoints_quarantine/counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_strict_stiefel.pt"
    
    print(f"Loading and translating {checkpoint_path}...")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        translated_state_dict = OrderedDict()
        for k, v in ckpt.items():
            new_k = k
            if new_k.startswith("phys.net"):
                new_k = new_k.replace("phys.net", "phys_mlp")
            if new_k.startswith("fuse.v_proj.weight_raw"):
                new_k = new_k.replace("fuse.v_proj.weight_raw", "stiefel_decomp.weight_raw")
            if new_k.startswith("fuse.gate_mlp"):
                new_k = new_k.replace("fuse.gate_mlp", "gate_engine")
            if new_k.startswith("vision."):
                new_k = new_k.replace("vision.", "vision_encoder.")
            if new_k.startswith("fuse.cls"):
                new_k = new_k.replace("fuse.cls", "task_classifier")
                
            translated_state_dict[new_k] = v
            
        model.load_state_dict(translated_state_dict, strict=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    print("Evaluating Condition 0 (Baseline)...")
    acc_std, dp_std = evaluate_condition(model, val_loader, device, phys_mu, phys_sigma, 'standard')
    
    print("Evaluating Condition K1 (Vision-Dead / Phys-Only)...")
    acc_k1, dp_k1 = evaluate_condition(model, val_loader, device, phys_mu, phys_sigma, 'k1')
    
    print("Evaluating Condition K2 (Phys-Dead / Vision-Only)...")
    acc_k2, dp_k2 = evaluate_condition(model, val_loader, device, phys_mu, phys_sigma, 'k2')
    
    print("\n=== Kill-Switch Baseline Results ===")
    print(f"Condition 0 (Baseline)    | Accuracy: {acc_std*100:.2f}% | DP Gap: {dp_std:.4f}")
    print(f"Condition K1 (Phys-Only)  | Accuracy: {acc_k1*100:.2f}% | DP Gap: {dp_k1:.4f}")
    print(f"Condition K2 (Vision-Only)| Accuracy: {acc_k2*100:.2f}% | DP Gap: {dp_k2:.4f}")
    
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

if __name__ == '__main__':
    main()
