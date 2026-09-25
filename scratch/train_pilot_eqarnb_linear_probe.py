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
import json

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
    
    regimes = ['ext', 'mod', 'rnd']
    logo = LeaveOneGroupOut()
    groups = df['subject'].values
    final_aucs = {}
    
    for r in regimes:
        print(f"\n--- Linear Probe Training EQARNB Regime: {r.upper()} ---")
        fold_idx = 1
        all_y_true, all_probs = [], []
        
        for train_idx, test_idx in logo.split(np.zeros(len(df)), df['threat'].values, groups):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            train_loader = DataLoader(UBFC_CF_Dataset(train_df, r, transform), batch_size=16, shuffle=True)
            test_loader = DataLoader(UBFC_CF_Dataset(test_df, r, transform), batch_size=16, shuffle=False)
            
            # Instantiate with freeze_vision=True
            model = MultimodalThreatModel(
                phys_dim=2, 
                vision_backbone="mobilenet_v3_small", 
                fusion='cgf', 
                num_classes=2,
                freeze_vision=True
            ).to(device)
            
            try:
                model.vision.backbone.load_state_dict(pretrained_backbone.state_dict(), strict=False)
            except:
                pass
                
            # Filter optimizer to only use requires_grad params
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
            
            model.train()
            # 10 epochs for linear probe convergence
            for epoch in range(10):
                for img_clean, img_scar, _, phys, y, _ in train_loader:
                    img_clean, img_scar = img_clean.to(device), img_scar.to(device)
                    phys, y = phys.to(device), y.to(device)
                    
                    optimizer.zero_grad()
                    out_scar = model(img_scar, phys)
                    loss_cls = F.cross_entropy(out_scar.logits, y)
                    
                    out_clean = model(img_clean, phys)
                    p_scar = F.softmax(out_scar.logits, dim=1)
                    p_clean = F.softmax(out_clean.logits, dim=1)
                    
                    loss_cf = jsd_stable(p_clean, p_scar)
                    loss = loss_cls + (0.5 * loss_cf)
                    
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                for img_clean, img_scar, _, phys, y, _ in test_loader:
                    img_scar, phys = img_scar.to(device), phys.to(device)
                    out = model(img_scar, phys)
                    probs = F.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
                    all_probs.extend(probs)
                    all_y_true.extend(y.numpy())
            
            print(f"Fold {fold_idx}/7 completed.")
            fold_idx += 1
            
        auc = roc_auc_score(all_y_true, all_probs)
        final_aucs[f"Regime_{r.upper()}"] = {"AUC-ROC": float(auc)}
        print(f"Regime {r.upper()} Linear Probe OOF AUC: {auc:.4f}")

    print("\n--- LINEAR PROBE CONVERGENCE RESULTS ---")
    print(json.dumps(final_aucs, indent=4))
    with open('scratch/linear_probe_aucs.json', 'w') as f:
        json.dump(final_aucs, f, indent=4)

if __name__ == '__main__':
    main()
