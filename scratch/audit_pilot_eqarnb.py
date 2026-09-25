import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(Path('src').resolve()))
from models_arch import MultimodalThreatModel

# Reuse dataset definition for Sham audit
from train_pilot_eqarnb import UBFC_CF_Dataset, jsd

def get_youden_threshold(y_true, probs):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_j, best_t = -1, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = tpr + tnr - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t, best_j

def main():
    out_dir = Path('outputs/pilot_checkpoints')
    regimes = ['ext', 'mod', 'rnd']
    results = {}
    
    # ---------------------------------------------------------
    # ACTION 2: CALIBRATED FAIRNESS AUDIT
    # ---------------------------------------------------------
    print("Executing Calibrated Fairness Audit...")
    for r in regimes:
        df_preds = pd.read_csv(out_dir / f"oof_preds_{r}.csv")
        y_true = df_preds['y_true'].values
        probs = df_preds['prob'].values
        a_true = df_preds['a_true'].values
        
        best_t, best_j = get_youden_threshold(y_true, probs)
        preds = (probs >= best_t).astype(int)
        
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
        
        # Demographic Parity Gap
        # dp_gap = | P(pred=1 | A=1) - P(pred=1 | A=0) |
        p_a1 = np.mean(preds[a_true == 1]) if sum(a_true == 1) > 0 else 0
        p_a0 = np.mean(preds[a_true == 0]) if sum(a_true == 0) > 0 else 0
        dp_gap = abs(p_a1 - p_a0)
        
        results[f"Regime_{r.upper()}"] = {
            "Optimal_Threshold": float(best_t),
            "Max_J_Statistic": float(best_j),
            "Accuracy": float(acc),
            "Precision": float(prec),
            "AUC-ROC": float(auc),
            "DP_Gap": float(dp_gap)
        }
        
    # ---------------------------------------------------------
    # ACTION 3: CAUSAL INVARIANCE VERIFICATION (Sham-Edit Audit)
    # ---------------------------------------------------------
    print("Executing Causal Invariance Sham Audit on R_extreme model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalThreatModel(phys_dim=2, vision_backbone="mobilenet_v3_small", fusion='cgf', num_classes=2).to(device)
    model.load_state_dict(torch.load(out_dir / "eqarnb_ext_fold1.pt", map_location=device, weights_only=True))
    model.eval()
    
    # Load Fold 1 Test Data (subject s1)
    df_data = pd.read_csv('data/ubfc_multimodal_processed/ubfc_multimodal_regimes.csv')
    test_df = df_data[df_data['subject'] == 's1']
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_loader = DataLoader(UBFC_CF_Dataset(test_df, 'ext', transform), batch_size=16, shuffle=False)
    
    js_true, js_sham = [], []
    with torch.no_grad():
        for img_clean, img_scar, img_sham, phys, _, _ in test_loader:
            img_clean, img_scar, img_sham = img_clean.to(device), img_scar.to(device), img_sham.to(device)
            phys = phys.to(device)
            
            p_clean = F.softmax(model(img_clean, phys).logits, dim=1)
            p_scar = F.softmax(model(img_scar, phys).logits, dim=1)
            p_sham = F.softmax(model(img_sham, phys).logits, dim=1)
            
            js_true.append(jsd(p_clean, p_scar).item())
            js_sham.append(jsd(p_clean, p_sham).item())
            
    mean_js_true = np.mean(js_true)
    mean_js_sham = np.mean(js_sham)
    ratio = mean_js_sham / mean_js_true if mean_js_true > 0 else 1.0
    
    results["Causal_Invariance_Audit"] = {
        "D_true (Semantic Scar)": float(mean_js_true),
        "D_sham (Chin Blur)": float(mean_js_sham),
        "D_sham / D_true Ratio": float(ratio),
        "Verdict": "PASS" if ratio <= 0.10 else "FAIL"
    }

    with open('scratch/audit_pilot_eqarnb_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- FINAL PILOT INTEGRATION METRICS ---")
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    main()
