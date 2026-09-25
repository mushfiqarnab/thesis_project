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
    
    print("\n--- FAST PROBE: REGIME EXT ONLY (rho=0.85) ---", flush=True)
    fold_idx = 1
    all_y_true, all_probs = [], []
    
    for train_idx, test_idx in logo.split(np.zeros(len(df)), df['threat'].values, groups):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        train_loader = DataLoader(UBFC_CF_Dataset(train_df, 'ext', transform), batch_size=32, shuffle=True)
        test_loader = DataLoader(UBFC_CF_Dataset(test_df, 'ext', transform), batch_size=32, shuffle=False)
        
        # Instantiate with freeze_vision=True
        model = MultimodalThreatModel(phys_dim=2, vision_backbone="mobilenet_v3_small", fusion='cgf', num_classes=2, freeze_vision=True).to(device)
        try:
            model.vision.backbone.load_state_dict(pretrained_backbone.state_dict(), strict=False)
        except: pass
            
        # Instead of finding specific module names, just rely on freeze_vision=True setting requires_grad=False internally
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-3)
        
        model.train()
        for epoch in range(5):
            for img_clean, img_scar, _, phys, y, _ in train_loader:
                img_clean, img_scar = img_clean.to(device), img_scar.to(device)
                phys, y = phys.to(device), y.to(device)
                
                optimizer.zero_grad()
                out_scar = model(img_scar, phys)
                loss = F.cross_entropy(out_scar.logits, y) + 0.5 * jsd_stable(F.softmax(model(img_clean, phys).logits, dim=1), F.softmax(out_scar.logits, dim=1))
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for _, img_scar, _, phys, y, _ in test_loader:
                img_scar, phys = img_scar.to(device), phys.to(device)
                probs = F.softmax(model(img_scar, phys).logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_y_true.extend(y.numpy())
        
        print(f"Fold {fold_idx}/7 completed.", flush=True)
        fold_idx += 1
        
    auc = roc_auc_score(all_y_true, all_probs)
    print(f"\n[RESULT] Regime EXT Fast Probe AUC: {auc:.4f}", flush=True)

if __name__ == '__main__':
    main()
