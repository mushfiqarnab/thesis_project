import os, json, sys, subprocess, math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from src.models_arch import MultimodalThreatModel
from scratch.dry_run import set_seed

print("\n=== TRUE BIASED EVALUATION SUITE ===")
try:
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('ascii').strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"]).decode('ascii').strip()
    is_dirty = "DIRTY" if len(dirty) > 0 else "CLEAN"
except:
    git_hash, is_dirty = "unknown", "unknown"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Git Hash: {git_hash} ({is_dirty})")
print(f"Device: {device}")

def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    safe_denom = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return numerator / safe_denom

def js_divergence_stable(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(p_logits, dim=1)
    q = F.softmax(q_logits, dim=1)
    m = 0.5 * (p + q)
    m_log = m.clamp(min=1e-8).log() 
    kl_p = F.kl_div(m_log, p, reduction='none').sum(dim=1)
    kl_q = F.kl_div(m_log, q, reduction='none').sum(dim=1)
    return 0.5 * (kl_p + kl_q)

def dp_gap_prob(p1: torch.Tensor, scar: torch.Tensor) -> torch.Tensor:
    s1 = (scar == 1).float()
    s0 = (scar == 0).float()
    m1 = safe_divide((p1 * s1).sum(), s1.sum())
    m0 = safe_divide((p1 * s0).sum(), s0.sum())
    return (m1 - m0).abs()

def eo_gap_prob(p1: torch.Tensor, y: torch.Tensor, scar: torch.Tensor) -> torch.Tensor:
    y01 = y.float()
    s1 = (scar == 1).float()
    s0 = (scar == 0).float()
    tpr1 = safe_divide((p1 * y01 * s1).sum(), (y01 * s1).sum())
    fpr1 = safe_divide((p1 * (1.0 - y01) * s1).sum(), ((1.0 - y01) * s1).sum())
    tpr0 = safe_divide((p1 * y01 * s0).sum(), (y01 * s0).sum())
    fpr0 = safe_divide((p1 * (1.0 - y01) * s0).sum(), ((1.0 - y01) * s0).sum())
    return torch.max((tpr1 - tpr0).abs(), (fpr1 - fpr0).abs())

with open("data/csv/folds_dry_run.json") as f:
    folds_data = json.load(f)
f0 = folds_data["folds"][0]
test_subs = f0["test"]

u = pd.read_csv("data/csv/multimodal_10k_unbiased.csv")
b = pd.read_csv("data/csv/multimodal_10k.csv")
ds_u = MultimodalCSVDatasetWithCF("data/csv/multimodal_10k_unbiased.csv")
ds_b = MultimodalCSVDatasetWithCF("data/csv/multimodal_10k.csv")

val_idx = u[u.subject.isin(f0["val"])].index.tolist()
val_df = u.iloc[val_idx]
val_maj = max((val_df.threat == 1).mean(), (val_df.threat == 0).mean())

def eval_metrics_custom(model, loader, mu, sig):
    model.eval()
    probs_all, y_all, scar_all = [], [], []
    cf_abs_scar = 0.0
    cf_flip_scar = 0
    cf_scar_count = 0
    with torch.no_grad():
        for b_batch in loader:
            img = b_batch["img"].to(device)
            img_cf = b_batch["img_cf"].to(device)
            phys = (b_batch["phys"].to(device) - mu) / sig
            y = b_batch["y"].to(device)
            scar = b_batch["scar"].to(device)
            has_cf = b_batch["has_cf"].to(device).bool()
            mask = b_batch["mask"].to(device)
            
            out = model(img, phys, mask=mask)
            p = torch.softmax(out.logits, dim=1)[:, 1]
            
            if has_cf.any():
                out_cf = model(img_cf, phys, mask=mask)
                p_cf = torch.softmax(out_cf.logits, dim=1)[:, 1]
                
                dif = (p - p_cf).abs()
                flip = (p >= 0.5) != (p_cf >= 0.5)
                
                scar_mask = (scar == 1) & has_cf
                cf_abs_scar += dif[scar_mask].sum().item()
                cf_flip_scar += flip[scar_mask].sum().item()
                cf_scar_count += scar_mask.sum().item()
                
            probs_all.append(p.cpu().numpy())
            y_all.append(y.cpu().numpy())
            scar_all.append(scar.cpu().numpy())
            
    probs = np.concatenate(probs_all)
    y_np = np.concatenate(y_all)
    s_np = np.concatenate(scar_all)
    yhat = (probs >= 0.5).astype(int)
    
    acc = float((yhat == y_np).mean())
    majority_acc = float(max((y_np == 1).mean(), (y_np == 0).mean()))
    p1_var = float(np.var(probs))
    
    s1_mask, s0_mask = s_np == 1, s_np == 0
    dp = float(abs(yhat[s1_mask].mean() - yhat[s0_mask].mean())) if (s1_mask.sum() and s0_mask.sum()) else 0.0
    
    def eo_rates(g):
        idx = (s_np == g)
        if not idx.any(): return 0.0, 0.0, 0, 0, 0, 0
        yy, yh = y_np[idx], yhat[idx]
        tp = ((yh == 1) & (yy == 1)).sum()
        fn = ((yh == 0) & (yy == 1)).sum()
        fp = ((yh == 1) & (yy == 0)).sum()
        tn = ((yh == 0) & (yy == 0)).sum()
        return tp, fn, fp, tn
        
    tp1, fn1, fp1, tn1 = eo_rates(1)
    tp0, fn0, fp0, tn0 = eo_rates(0)
    
    tpr1 = tp1 / max(tp1 + fn1, 1)
    fpr1 = fp1 / max(fp1 + tn1, 1)
    tpr0 = tp0 / max(tp0 + fn0, 1)
    fpr0 = fp0 / max(fp0 + tn0, 1)
    
    eo_max = max(abs(tpr1 - tpr0), abs(fpr1 - fpr0))
    cf_gap = cf_abs_scar / max(cf_scar_count, 1)
    cf_flip_rate = cf_flip_scar / max(cf_scar_count, 1)
    
    return {
        "acc": acc, "dp_abs": dp, "eo_max_gap": eo_max, "cf_gap": cf_gap,
        "majority_acc": majority_acc, "p1_var": p1_var, "cf_flip_rate": cf_flip_rate,
        "tp1": tp1, "fn1": fn1, "fp1": fp1, "tn1": tn1, "tpr1": tpr1, "fpr1": fpr1,
        "tp0": tp0, "fn0": fn0, "fp0": fp0, "tn0": tn0, "tpr0": tpr0, "fpr0": fpr0,
        "tpr_gap": abs(tpr1 - tpr0), "fpr_gap": abs(fpr1 - fpr0)
    }

def train_eval_config(config_name, fusion, disable_stiefel, penalty_on):
    print(f"\n==========================================================")
    print(f"CONFIGURATION: {config_name}")
    print(f"Fusion: {fusion}, Stiefel Disabled: {disable_stiefel}, Penalty ON: {penalty_on}")
    print(f"==========================================================")
    
    train_idx = b[b.subject.isin(f0["train"])].index.tolist()
    val_idx = b[b.subject.isin(f0["val"])].index.tolist()
    
    train_phys = b.iloc[train_idx][["hrv", "gsr"]].to_numpy(dtype=np.float32)
    phys_mu = torch.tensor(train_phys.mean(axis=0), device=device)
    phys_sigma = torch.tensor(train_phys.std(axis=0).clip(min=1e-6), device=device)
    
    train_loader = DataLoader(Subset(ds_b, train_idx), batch_size=64, shuffle=True, collate_fn=collate_samples)
    val_loader = DataLoader(Subset(ds_b, val_idx), batch_size=64, shuffle=False, collate_fn=collate_samples)
    
    set_seed(0)
    model = MultimodalThreatModel(phys_dim=2, fusion=fusion, disable_stiefel=disable_stiefel).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    # Use scheduler for fair comparison if penalties are on
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    ce = nn.CrossEntropyLoss()
    
    w_dp, w_eo, w_cf = 1.0, 1.0, 0.2
    lam_cf = 1.0 if penalty_on else 0.0
    lam_dp = 0.5 if penalty_on else 0.0
    lam_eo = 0.5 if penalty_on else 0.0
    lam_gate = 0.05 if penalty_on else 0.0
    
    best_score = -1e9
    best_state = None
    
    for epoch in range(1, 16):
        model.train()
        
        # Annealing logic from train_cgf_fair.py
        anneal_factor = 0.0
        warmup, duration = 2, 8
        if epoch > warmup:
            if epoch >= warmup + duration:
                anneal_factor = 1.0
            else:
                progress = (epoch - warmup) / duration
                anneal_factor = 0.5 * (1 - math.cos(math.pi * progress))
                
        cur_lam_dp = lam_dp * anneal_factor
        cur_lam_eo = lam_eo * anneal_factor
        
        for step, batch in enumerate(train_loader):
            img, img_cf = batch["img"].to(device), batch["img_cf"].to(device)
            phys = (batch["phys"].to(device) - phys_mu) / phys_sigma
            y, scar = batch["y"].to(device), batch["scar"].to(device)
            has_cf, mask = batch["has_cf"].to(device).bool(), batch["mask"].to(device)
            
            out = model(img, phys, mask=mask)
            loss_task = ce(out.logits, y)
            
            loss_cf = torch.tensor(0.0, device=device)
            if lam_cf > 0 and has_cf.any():
                out_cf = model(img_cf, phys, mask=mask)
                js = js_divergence_stable(out.logits, out_cf.logits)
                loss_cf = js[has_cf].mean()
                
            loss_gate = torch.tensor(0.0, device=device)
            if lam_gate > 0 and out.gate is not None and out.focus is not None:
                focus = torch.log1p(out.focus.clamp(min=0.0, max=1e3))
                loss_gate = (out.gate * focus).mean()
                
            p1 = F.softmax(out.logits, dim=1)[:, 1]
            loss_dp = dp_gap_prob(p1, scar)
            loss_eo = eo_gap_prob(p1, y, scar)
            
            loss = loss_task + lam_cf * loss_cf + lam_gate * loss_gate + cur_lam_dp * loss_dp + cur_lam_eo * loss_eo
            
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            
        scheduler.step()
        
        val = eval_metrics_custom(model, val_loader, phys_mu, phys_sigma)
        is_degenerate = val["p1_var"] < 0.005
        is_below_baseline = val["acc"] <= (val["majority_acc"] + 0.01)
        
        # Uses standard selection rule: score = Acc - w_eo * eo_max - w_cf * cf_gap
        score = val["acc"] - w_eo * val["eo_max_gap"] - w_cf * val["cf_gap"]
        
        if is_degenerate or is_below_baseline:
            score = -1e9
            
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    print(f"Training complete. Best Validation Score: {best_score:.4f}")
    if best_score == -1e9:
        print("WARNING: Model collapsed in all epochs. Using last epoch for evaluation.")
    else:
        model.load_state_dict(best_state)
        
    model.eval()
    
    for eval_name, test_df, ds_test in [("BIASED TEST ROWS", b, ds_b), ("UNBIASED TEST ROWS", u, ds_u)]:
        print(f"\n--- EVALUATION ON {eval_name} ---")
        for subj in test_subs:
            idx = test_df[test_df.subject == subj].index.tolist()
            subj_loader = DataLoader(Subset(ds_test, idx), batch_size=128, shuffle=False, collate_fn=collate_samples)
            met = eval_metrics_custom(model, subj_loader, phys_mu, phys_sigma)
            
            print(f"Subject {subj}:")
            print(f"  Accuracy: {met['acc']:.4f} (Majority: {met['majority_acc']:.4f})")
            print(f"  CF-Gap (Scar Rows Only): {met['cf_gap']:.4f} | Hard-Flip Rate: {met['cf_flip_rate']:.4f}")
            print(f"  EO TPR Gap: {met['tpr_gap']:.4f} | EO FPR Gap: {met['fpr_gap']:.4f}")
            print(f"  Cells -> Scar=1: TP={met['tp1']} FN={met['fn1']} FP={met['fp1']} TN={met['tn1']}")
            print(f"  Cells -> Scar=0: TP={met['tp0']} FN={met['fn0']} FP={met['fp0']} TN={met['tn0']}")

# Config 1: Concat fusion, no penalty, no Stiefel
train_eval_config("1. Concat / No Penalty / No Stiefel", fusion="concat", disable_stiefel=True, penalty_on=False)

# Config 2: CGF fusion, penalty on, no Stiefel
train_eval_config("2. CGF / Penalty ON / No Stiefel", fusion="cgf", disable_stiefel=True, penalty_on=True)

# Config 3: CGF fusion, penalty on, Stiefel on
train_eval_config("3. CGF / Penalty ON / Stiefel ON", fusion="cgf", disable_stiefel=False, penalty_on=True)
