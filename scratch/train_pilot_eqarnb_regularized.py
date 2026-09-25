import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path('src').resolve()))
from models_arch import MultimodalThreatModel
from train_pilot_eqarnb import UBFC_CF_Dataset

def jsd_stable(p, q):
    p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
    q = torch.clamp(q, 1e-7, 1.0 - 1e-7)
    m = 0.5 * (p + q)
    return 0.5 * F.kl_div(m.log(), p, reduction='batchmean') + 0.5 * F.kl_div(m.log(), q, reduction='batchmean')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/ubfc_multimodal_processed/ubfc_multimodal_regimes.csv')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    pretrained_backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    logo = LeaveOneGroupOut()
    groups = df['subject'].values
    
    # HYPERPARAMETERS
    lambda_cf = 0.5
    lambda_g = 0.25 # L1 penalty to prevent over-trust of vision
    
    print("\n--- REGULARIZED PROBE: REGIME EXT ONLY (rho=0.85) ---", flush=True)
    print(f"Applying Loss: L_ce + {lambda_cf} * L_cf + {lambda_g} * L_gate", flush=True)
    
    fold_idx = 1
    all_y_true, all_probs = [], []
    all_gates = []
    
    for train_idx, test_idx in logo.split(np.zeros(len(df)), df['threat'].values, groups):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        train_loader = DataLoader(UBFC_CF_Dataset(train_df, 'ext', transform), batch_size=32, shuffle=True)
        test_loader = DataLoader(UBFC_CF_Dataset(test_df, 'ext', transform), batch_size=32, shuffle=False)
        
        model = MultimodalThreatModel(phys_dim=2, vision_backbone="mobilenet_v3_small", fusion='cgf', num_classes=2, freeze_vision=True).to(device)
        try:
            model.vision.backbone.load_state_dict(pretrained_backbone.state_dict(), strict=False)
        except: pass
            
        for param in model.vision.parameters():
            param.requires_grad = False
        for param in model.fuse.parameters():
            param.requires_grad = True
        for param in model.phys_mlp.parameters():
            param.requires_grad = True
            
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-3)
        
        model.train()
        for epoch in range(5):
            for img_clean, img_scar, _, phys, y, _ in train_loader:
                img_clean, img_scar = img_clean.to(device), img_scar.to(device)
                phys, y = phys.to(device), y.to(device)
                
                optimizer.zero_grad()
                out_scar = model(img_scar, phys)
                out_clean = model(img_clean, phys)
                
                loss_ce = F.cross_entropy(out_scar.logits, y)
                loss_cf = jsd_stable(F.softmax(out_clean.logits, dim=1), F.softmax(out_scar.logits, dim=1))
                loss_gate = out_scar.gate.mean() # L1 Regularization on the gate
                
                # The patched loss function
                loss = loss_ce + (lambda_cf * loss_cf) + (lambda_g * loss_gate)
                
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for _, img_scar, _, phys, y, _ in test_loader:
                img_scar, phys = img_scar.to(device), phys.to(device)
                out = model(img_scar, phys)
                probs = F.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
                gates = out.gate.cpu().numpy().flatten()
                
                all_probs.extend(probs)
                all_y_true.extend(y.numpy())
                all_gates.extend(gates)
        
        fold_idx += 1
        
    auc = roc_auc_score(all_y_true, all_probs)
    mean_gate = np.mean(all_gates)
    
    print(f"\n[RESULT] Regularized Probe AUC: {auc:.4f}", flush=True)
    print(f"[RESULT] Regularized Mean Gate: {mean_gate:.4f}", flush=True)

if __name__ == '__main__':
    main()
