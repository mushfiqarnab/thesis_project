import sys
from pathlib import Path
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models_arch import MultimodalThreatModel

def diagnose_collapse(csv_path, ckpt_path, seed=42, val_ratio=0.2, device="cuda"):
    print("=" * 75)
    print(f"DIAGNOSTIC AUTOPSY: {Path(ckpt_path).name}")
    print(f"Dataset: {Path(csv_path).name}")
    print("=" * 75)
    
    csv_path = Path(csv_path)
    ds = MultimodalCSVDatasetWithCF(str(csv_path))
    
    # Same logic to find train/val split
    split_path = csv_path.parent / f"split_seed{seed}_{csv_path.stem}.json"
    if split_path.exists():
        with open(split_path, 'r') as f:
            d = json.load(f)
            val_idx = d["val_idx"]
            train_idx = d["train_idx"]
    else:
        print(f"Split file not found: {split_path}")
        return

    val_ds = Subset(ds, val_idx)
    
    loader = DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_samples
    )
    
    # Same Z-score logic for physics using TRAIN dataset
    X = ds.df.iloc[train_idx][ds.phys_cols].to_numpy(dtype=np.float32, copy=True)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    phys_mu = torch.tensor(mu, device=device, dtype=torch.float32).unsqueeze(0)
    phys_sigma = torch.tensor(sigma, device=device, dtype=torch.float32).unsqueeze(0)

    phys_dim = ds[0].phys.numel()
    model = MultimodalThreatModel(
        phys_dim=phys_dim,
        vision_backbone="mobilenet_v3_small",
        fusion="cgf",
        num_classes=2,
    ).to(device)

    # Safely load the state dict
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except:
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    
    probs_all, y_all, scar_all = [], [], []
    with torch.no_grad():
        for b in loader:
            img = b["img"].to(device)
            phys = b["phys"].to(device)
            y = b["y"].to(device)
            scar = b["scar"].to(device)
            mask = b["mask"].to(device)
            
            phys = (phys - phys_mu) / phys_sigma
            out = model(img, phys, mask=mask)
            p = F.softmax(out.logits, dim=1)[:, 1]
            
            probs_all.append(p.cpu().numpy())
            y_all.append(y.cpu().numpy())
            scar_all.append(scar.cpu().numpy())
            
    p1 = np.concatenate(probs_all)
    y_np = np.concatenate(y_all)
    s_np = np.concatenate(scar_all)
    
    yhat = (p1 >= 0.5).astype(int)
    acc = (yhat == y_np).mean()
    
    s1_mask = s_np == 1
    s0_mask = s_np == 0
    dp_gap = abs(yhat[s1_mask].mean() - yhat[s0_mask].mean()) if (s1_mask.sum() and s0_mask.sum()) else 0.0
    
    # Calculate Variance
    p1_var = np.var(p1)
    
    print(f"Accuracy:           {acc:.4f}")
    print(f"DP Gap (Hard Pred): {dp_gap:.4f}")
    print(f"Variance of P(y=1): {p1_var:.6f}")
    print(f"Mean P(y=1):        {p1.mean():.4f}")
    print(f"True Threat Rate:   {y_np.mean():.4f}")
    print(f"Predicted + Rate:   {yhat.mean():.4f}")
    
    print("\nConfusion Matrix Insights:")
    tp = ((yhat == 1) & (y_np == 1)).sum()
    fp = ((yhat == 1) & (y_np == 0)).sum()
    tn = ((yhat == 0) & (y_np == 0)).sum()
    fn = ((yhat == 0) & (y_np == 1)).sum()
    print(f"  TP: {tp} | FP: {fp}")
    print(f"  FN: {fn} | TN: {tn}")
    print(f"  True Pos Rate (TPR):  {tp / max(1, tp + fn):.4f}")
    print(f"  False Pos Rate (FPR): {fp / max(1, fp + tn):.4f}")
    
    # Assess Collapse
    if p1_var < 0.01:
        print("\n[CONCLUSION] FATAL SOFT COLLAPSE DETECTED.")
        print("The variance of the predicted probability is near zero. The model is ignoring the input")
        print("and applying a near-constant baseline prediction to trivially bypass the fairness penalty.")
    else:
        print("\n[CONCLUSION] Model exhibits variance in predictions. It is NOT fully collapsed.")
        
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "outputs/checkpoints/counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.pt"
    
    # Eval on the Unbiased dataset (what it was trained on)
    diagnose_collapse("data/csv/multimodal_10k_unbiased.csv", ckpt, device=device)
    
    # Eval on the Biased dataset (to see true causal generalization)
    print("\n")
    diagnose_collapse("data/csv/multimodal_10k.csv", ckpt, device=device)
