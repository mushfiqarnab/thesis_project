import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path('src').resolve()))
from data.clinical_dataloader import MultimodalClinicalDataset
from models_arch import MultimodalThreatModel
from torch.utils.data import DataLoader, Subset
import json

def get_mode_fraction(probs, tol=0.02):
    counts, edges = np.histogram(probs, bins=100, range=(0,1))
    mode_idx = np.argmax(counts)
    mode_val = (edges[mode_idx] + edges[mode_idx+1]) / 2.0
    within_tol = np.abs(probs - mode_val) <= tol
    return np.mean(within_tol)

def evaluate_checkpoint(ckpt_path, ds_path, split_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset with missing file handling
    ds = MultimodalClinicalDataset(ds_path)
    
    with open(split_path, 'r') as f:
        d = json.load(f)
        val_idx = d["val_idx"]
    # Filter val_idx to only include indices within ds length (since missing files were dropped)
    val_idx = [i for i in val_idx if i < len(ds)]
    val_ds = Subset(ds, val_idx)
    
    def simple_collate(batch):
        return {
            "img": torch.stack([b["img"] for b in batch]),
            "phys": torch.stack([b["phys"] for b in batch]),
            "y": torch.stack([b["y"] for b in batch]),
            "scar": torch.stack([b["scar"] for b in batch]),
        }
    
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=simple_collate)
    
    model = MultimodalThreatModel(
        phys_dim=ds[0]["phys"].numel(),
        vision_backbone="mobilenet_v3_small",
        fusion="cgf",
        num_classes=2
    ).to(device)
    
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
        
    model.eval()
    
    probs_list = []
    y_list = []
    scar_list = []
    
    with torch.no_grad():
        for b in val_loader:
            img = b["img"].to(device)
            phys = b["phys"].to(device)
            y = b["y"].numpy()
            scar = b["scar"].numpy()
            out = model(img=img, phys=phys, mask=None)
            p = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            probs_list.append(p)
            y_list.append(y)
            scar_list.append(scar)
            
    probs = np.concatenate(probs_list)
    y_all = np.concatenate(y_list)
    scar_all = np.concatenate(scar_list)
    
    yhat = (probs >= 0.5).astype(int)
    acc = np.mean(yhat == y_all)
    maj_acc = max(np.mean(y_all == 1), np.mean(y_all == 0))
    
    s1_mask = scar_all == 1
    s0_mask = scar_all == 0
    dp = abs(np.mean(yhat[s1_mask]) - np.mean(yhat[s0_mask])) if s1_mask.sum() and s0_mask.sum() else 0.0
    
    def eo_rates(g):
        idx = (scar_all == g)
        yy, yh = y_all[idx], yhat[idx]
        tp = np.sum((yh == 1) & (yy == 1))
        fn = np.sum((yh == 0) & (yy == 1))
        fp = np.sum((yh == 1) & (yy == 0))
        tn = np.sum((yh == 0) & (yy == 0))
        return tp / max(tp + fn, 1), fp / max(fp + tn, 1)
        
    tpr1, fpr1 = eo_rates(1)
    tpr0, fpr0 = eo_rates(0)
    eo_max = max(abs(tpr1 - tpr0), abs(fpr1 - fpr0))
    hist, edges = np.histogram(probs, bins=10, range=(0,1))
    frac_mode = get_mode_fraction(probs)
    
    print(f"--- Checkpoint: {ckpt_path} ---")
    print(f"p(class=1) min: {probs.min():.6f}, max: {probs.max():.6f}, mean: {probs.mean():.6f}, std: {probs.std():.6f}")
    print(f"Histogram (10 bins 0-1): {hist.tolist()}")
    print(f"Fraction within 0.02 of mode: {frac_mode:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Majority-class baseline: {maj_acc:.4f}")
    print(f"DP gap: {dp:.4f}")
    print(f"EO gap: {eo_max:.4f}\n")

if __name__ == "__main__":
    ckpts = [
        "outputs/lambda_sweep_ckpts/lambda_5.0_final.pth"
    ]
    ds_path = "data/csv/multimodal.csv"
    split_path = "data/csv/split_seed42_multimodal.json"
    for c in ckpts:
        if os.path.exists(c):
            try:
                evaluate_checkpoint(c, ds_path, split_path)
            except Exception as e:
                print(f"Error evaluating {c}: {e}")
