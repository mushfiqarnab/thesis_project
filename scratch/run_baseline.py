import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models_arch import MultimodalThreatModel
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scratch.dry_run import set_seed, eval_metrics_per_subject

def run_one_epoch(model, opt, loader, device, phys_mu, phys_sigma, w_dp, w_eo, w_cf):
    model.train()
    ce = nn.CrossEntropyLoss()
    for step, b in enumerate(loader):
        img, img_cf = b["img"].to(device), b["img_cf"].to(device)
        phys = (b["phys"].to(device) - phys_mu) / phys_sigma
        y, scar = b["y"].to(device), b["scar"].to(device)
        has_cf, mask = b["has_cf"].to(device).bool(), b["mask"].to(device)

        out = model(img, phys, mask=mask)
        loss = ce(out.logits, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

def run_experiment(train_csv_path, name):
    print(f"\n==================================================")
    print(f"BASELINE EXPERIMENT: {name}")
    print(f"Train CSV: {train_csv_path}")
    print(f"==================================================")
    
    with open("data/csv/folds_dry_run.json") as f:
        folds_data = json.load(f)
    f0 = folds_data["folds"][0]
    
    # We always test on the unbiased file to measure true generalizability/CF gap
    # But wait, the user said "trained once on unbiased file and once on biased file". 
    # Usually you'd evaluate on the unbiased test set to be fair, or on both. 
    # I will evaluate on the unbiased test set (test subjects) for both, since test set shouldn't have spurious correlations if we want true performance.
    # Actually, I'll just evaluate on the respective test split of the file being used, but since test subjects are isolated, it's fine.
    
    # Load train file
    u_train = pd.read_csv(train_csv_path)
    ds_train = MultimodalCSVDatasetWithCF(train_csv_path)
    
    # Load test file (unbiased always for fair eval, or use the same file? I'll use the same file for simplicity, 
    # but print metrics. Wait, if I test on biased, the CF-gap is still valid, but accuracy might be inflated by physiology.
    # Let's test on the same file the model is trained on, for the held-out test subjects.)
    u_test = u_train
    ds_test = ds_train
    
    train_idx = u_train[u_train.subject.isin(f0["train"])].index.tolist()
    val_idx = u_train[u_train.subject.isin(f0["val"])].index.tolist()
    test_idx = u_test[u_test.subject.isin(f0["test"])].index.tolist()
    
    train_phys = u_train.iloc[train_idx][["hrv", "gsr"]].to_numpy(dtype=np.float32)
    phys_mu = torch.tensor(train_phys.mean(axis=0), device="cpu")
    phys_sigma = torch.tensor(train_phys.std(axis=0).clip(min=1e-6), device="cpu")
    
    train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=64, shuffle=True, collate_fn=collate_samples)
    val_loader = DataLoader(Subset(ds_train, val_idx), batch_size=64, shuffle=False, collate_fn=collate_samples)
    test_loader = DataLoader(Subset(ds_test, test_idx), batch_size=64, shuffle=False, collate_fn=collate_samples)

    set_seed(0)
    # concat fusion, no Stiefel
    model = MultimodalThreatModel(phys_dim=2, fusion="concat", disable_stiefel=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    best_score = -1e9
    best_state = None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phys_mu = phys_mu.to(device)
    phys_sigma = phys_sigma.to(device)
    model.to(device)
    for epoch in range(1, 16):
        # no penalty (w_dp=0, w_eo=0, w_cf=0)
        run_one_epoch(model, opt, train_loader, device, phys_mu, phys_sigma, 0.0, 0.0, 0.0)
        val = eval_metrics_per_subject(model, val_loader, device, phys_mu, phys_sigma)
        
        score = val["acc"]
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    print(f"Training complete. Best Val Acc: {best_score:.4f}")
    
    # Load best model for testing
    model.load_state_dict(best_state)
    model.to(device)
    
    print("\n--- TEST SET EVALUATION (Per Subject) ---")
    model.eval()
    
    # We will evaluate per subject
    test_subjects = f0["test"]
    for subj in test_subjects:
        subj_idx = u_test[u_test.subject == subj].index.tolist()
        subj_loader = DataLoader(Subset(ds_test, subj_idx), batch_size=64, shuffle=False, collate_fn=collate_samples)
        
        met = eval_metrics_per_subject(model, subj_loader, device, phys_mu, phys_sigma)
        
        print(f"Subject {subj}:")
        print(f"  Test Acc : {met['acc']:.4f} (Majority: {met['majority_acc']:.4f})")
        print(f"  P1 Var   : {met['p1_var']:.4f}")
        print(f"  CF-Gap   : {met['cf_gap']:.4f} (Mean |delta_p|)")
        print(f"  CF-Flips : {met['cf_flip_rate']:.4f} (Hard Flips)")
        print(f"  EO TPR Gap: {met['eo_tpr_gap']:.4f}")
        print(f"  EO FPR Gap: {met['eo_fpr_gap']:.4f}")

run_experiment("data/csv/multimodal_10k_unbiased.csv", "Unbiased Data Baseline")
run_experiment("data/csv/multimodal_10k.csv", "Biased Data Baseline")
