"""
thesis_diagnostics_strict.py
=====================================================================
Final zero-compromise empirical extraction for thesis pp33.
Features:
  1. Strict Subject-Leakage-Free Validation.
  2. Dynamic Training-Set Z-Score Normalization for Physiologic Data.
  3. Equalized Odds (EO) Extraction (TPR / FPR disaggregated).
  4. Stratified Bootstrap 95% CI on Soft DP Gap.
  5. CGF Gate Attenuation Statistics (Welch t-test + Cohen d).
"""
from __future__ import annotations
import sys
import json
import os
from pathlib import Path

# Fix relative imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy import stats
import pandas as pd

from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models_arch import MultimodalThreatModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNBIASED_CSV = Path("data/csv/multimodal_10k_unbiased.csv")
BIASED_CSV = Path("data/csv/multimodal_10k.csv")
SPLIT_FILE = Path("data/csv/multimodal_10k_strict_split_seed42.json")
CKPT_DIR   = Path("outputs/checkpoints")

# The 6 new strict-split checkpoints
CKPTS = {
    "Baseline":        ("counterfactual_concat_js_mobilenet_v3_small_multimodal_10k_unbiased_best_strict_baseline.pt", "concat", "1"),
    "Naive-Concat":    ("counterfactual_concat_js_mobilenet_v3_small_multimodal_10k_unbiased_best_strict_concat.pt", "concat", "0"),
    "V4-Penalty":      ("counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_strict_v4.pt", "cgf", "1"),
    "Flagship-Stiefel":("counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_strict_stiefel.pt", "cgf", "0"),
    "Pure-CF":         ("counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_strict_purecf.pt", "cgf", "0"),
    "Stiefel-ZeroLoss":("counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_strict_stiefel_zeroloss.pt", "cgf", "0"),
}

BATCH = 128
WORKERS = 0

# --- 1. DATA PREP & NORMALIZATION FIX ---
def get_normalization_stats():
    """Extracts mean/std exclusively from the unbiased train split"""
    print(f"Loading Split: {SPLIT_FILE}")
    with open(SPLIT_FILE, 'r') as f:
        split = json.load(f)
    train_idx = split['train_idx']
    
    ds = MultimodalCSVDatasetWithCF(str(UNBIASED_CSV))
    train_phys_raw = ds.df.iloc[train_idx][ds.phys_cols].to_numpy(dtype=np.float32)
    
    phys_mu = torch.tensor(train_phys_raw.mean(axis=0), device=DEVICE)
    phys_std = torch.tensor(train_phys_raw.std(axis=0).clip(min=1e-6), device=DEVICE)
    
    return phys_mu, phys_std

def load_biased_eval_loader():
    """Loads the biased evaluation set"""
    ds = MultimodalCSVDatasetWithCF(str(BIASED_CSV))
    # Note: We evaluate on the whole biased set, or just the val split of the unbiased set.
    # To measure algorithmic bias generalization, we evaluate on the BIASED dataset.
    return DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS, collate_fn=collate_samples)

# --- 2. SECURE CHECKPOINT LOADER ---
def load_model_securely(ckpt_name: str, fusion: str, disable_stiefel: str) -> torch.nn.Module:
    os.environ["EQUITAS_DISABLE_STIEFEL"] = disable_stiefel
    model = MultimodalThreatModel(phys_dim=2, fusion=fusion, vision_backbone="mobilenet_v3_small")
    
    ckpt_path = CKPT_DIR / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing Checkpoint: {ckpt_path}")
        
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
        
    cleaned = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model.to(DEVICE)

# --- 3. INFERENCE ---
@torch.no_grad()
def run_inference(model, loader, phys_mu, phys_std, collect_gates=False):
    probs, labels, scars, gates = [], [], [], []
    for batch in loader:
        img   = batch["img"].to(DEVICE)
        raw_phys = batch["phys"].to(DEVICE)
        
        # APPLY THE MISSING NORMALIZATION
        phys = (raw_phys - phys_mu) / phys_std
        
        mask  = batch["mask"].to(DEVICE)
        y     = batch["y"].long()
        scar  = batch["scar"].long()

        out = model(img, phys, mask)
        p1  = F.softmax(out.logits, dim=1)[:, 1].cpu()

        probs.append(p1)
        labels.append(y)
        scars.append(scar)
        if collect_gates and out.gate is not None:
            gates.append(out.gate.squeeze(1).cpu())
            
    p_np = torch.cat(probs).numpy()
    y_np = torch.cat(labels).numpy()
    s_np = torch.cat(scars).numpy()
    g_np = torch.cat(gates).numpy() if collect_gates and len(gates)>0 else None
    return p_np, y_np, s_np, g_np

# --- 4. AUDITS ---
def audit_equalized_odds(name, p_np, y_np, s_np):
    yhat = (p_np >= 0.5).astype(int)
    
    def eo(g):
        idx = (s_np == g)
        yy, yh = y_np[idx], yhat[idx]
        tp = ((yh == 1) & (yy == 1)).sum()
        fn = ((yh == 0) & (yy == 1)).sum()
        fp = ((yh == 1) & (yy == 0)).sum()
        tn = ((yh == 0) & (yy == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        return tpr, fpr, tp, tn, fp, fn

    tpr1, fpr1, tp1, tn1, fp1, fn1 = eo(1)
    tpr0, fpr0, tp0, tn0, fp0, fn0 = eo(0)
    
    acc = (yhat == y_np).mean()
    dp_hard = abs(yhat[s_np==1].mean() - yhat[s_np==0].mean())
    dp_soft = abs(p_np[s_np==1].mean() - p_np[s_np==0].mean())
    
    print(f"\n[{name}]")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Hard DP Gap : {dp_hard:.4f} | Soft DP Gap: {dp_soft:.4f}")
    print(f"  Scar=1 (Biased)   -> TPR: {tpr1:.4f}  | FPR: {fpr1:.4f}  | N={s_np.sum()}")
    print(f"  Scar=0 (Unbiased) -> TPR: {tpr0:.4f}  | FPR: {fpr0:.4f}  | N={(s_np==0).sum()}")
    
    return dp_soft

def audit_bootstrap_dp(models_p, s_np, n_iters=10000):
    print(f"\n{'='*70}\n AUDIT: STRATIFIED BOOTSTRAP 95% CI (Soft DP Gap)\n{'='*70}")
    
    s1_idx = np.where(s_np == 1)[0]
    s0_idx = np.where(s_np == 0)[0]
    n1, n0 = len(s1_idx), len(s0_idx)
    
    results = {m: [] for m in models_p}
    
    for _ in range(n_iters):
        b1 = np.random.choice(s1_idx, n1, replace=True)
        b0 = np.random.choice(s0_idx, n0, replace=True)
        
        for name, p_np in models_p.items():
            dp = abs(p_np[b1].mean() - p_np[b0].mean())
            results[name].append(dp)
            
    print(f"  {n_iters:,} iterations | Stratified resampling | Soft probabilities")
    print(f"  {str('Model').ljust(20)} {'DP Mean':>10} {'CI Lower':>12} {'CI Upper':>12}")
    print("  " + "-"*55)
    
    for name, dps in results.items():
        arr = np.array(dps)
        mean_val = arr.mean()
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print(f"  {name.ljust(20)} {mean_val:10.4f} {lo:12.4f} {hi:12.4f}")

def audit_gate_stats(name, g_np, s_np):
    if g_np is None:
        return
    print(f"\n{'='*70}\n AUDIT: CGF GATE ATTENUATION ({name})\n{'='*70}")
    g1 = g_np[s_np == 1]
    g0 = g_np[s_np == 0]
    
    t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False)
    
    n1, n0 = len(g1), len(g0)
    var1, var0 = np.var(g1, ddof=1), np.var(g0, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))
    cohen_d = (np.mean(g1) - np.mean(g0)) / pooled_sd

    print(f"  Gate (scar=1) : mean={np.mean(g1):.4f}  std={np.std(g1):.4f}  n={n1}")
    print(f"  Gate (scar=0) : mean={np.mean(g0):.4f}  std={np.std(g0):.4f}  n={n0}")
    print(f"  Absolute Delta: {abs(np.mean(g1) - np.mean(g0)):.4f}")
    print(f"  Welch t-stat  : {t_stat:.4f}")
    print(f"  p-value       : {p_val:.6e}")
    print(f"  Cohen d       : {cohen_d:.4f}")

def main():
    print("======================================================================")
    print(" EQUITAS-MITL: FINAL DIAGNOSTIC EXTRACTION (ZERO-LEAKAGE + NORM FIXED)")
    print("======================================================================")
    
    phys_mu, phys_std = get_normalization_stats()
    print(f"Train Phys Mean: {phys_mu.cpu().numpy()}")
    print(f"Train Phys Std : {phys_std.cpu().numpy()}")
    
    loader = load_biased_eval_loader()
    
    models_p = {}
    master_s = None
    
    for name, (ckpt, fusion, disable_stiefel) in CKPTS.items():
        try:
            model = load_model_securely(ckpt, fusion, disable_stiefel)
            p_np, y_np, s_np, g_np = run_inference(model, loader, phys_mu, phys_std, collect_gates=(fusion=="cgf"))
            
            models_p[name] = p_np
            master_s = s_np
            audit_equalized_odds(name, p_np, y_np, s_np)
            
            if name == "Flagship-Stiefel" and g_np is not None:
                audit_gate_stats(name, g_np, s_np)
        except FileNotFoundError:
            print(f"  [SKIPPED] Checkpoint not yet generated: {name}")
        except Exception as e:
            print(f"  [FATAL ERROR] Audit failed for {name}: {e}")
            raise
            
    if models_p:
        audit_bootstrap_dp(models_p, master_s)

if __name__ == "__main__":
    main()
