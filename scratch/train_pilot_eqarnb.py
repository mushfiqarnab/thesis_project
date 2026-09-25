import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from PIL import Image

import sys
sys.path.insert(0, str(Path('src').resolve()))
from models_arch import MultimodalThreatModel

class UBFC_CF_Dataset(Dataset):
    def __init__(self, df, regime_prefix, transform=None):
        self.df = df.reset_index(drop=True)
        self.regime_prefix = regime_prefix
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_clean = Image.open(row['clean_path']).convert('RGB')
        img_scar = Image.open(row[f"{self.regime_prefix}_path"]).convert('RGB')
        img_sham = Image.open(row['sham_path']).convert('RGB')
        
        if self.transform:
            img_clean = self.transform(img_clean)
            img_scar = self.transform(img_scar)
            img_sham = self.transform(img_sham)
            
        phys = torch.tensor([row['hrv'], row['gsr']], dtype=torch.float32)
        y = torch.tensor(row['threat'], dtype=torch.long)
        a = torch.tensor(row[f"{self.regime_prefix}_scar"], dtype=torch.float32)
        
        return img_clean, img_scar, img_sham, phys, y, a

def jsd(p, q):
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
    
    out_dir = Path('outputs/pilot_checkpoints')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    regimes = ['ext', 'mod', 'rnd']
    logo = LeaveOneGroupOut()
    groups = df['subject'].values
    
    oof_predictions = {r: [] for r in regimes}
    
    for r in regimes:
        print(f"\n--- Training EQARNB Regime: {r.upper()} ---")
        fold_idx = 1
        
        for train_idx, test_idx in logo.split(np.zeros(len(df)), df['threat'].values, groups):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            train_loader = DataLoader(UBFC_CF_Dataset(train_df, r, transform), batch_size=16, shuffle=True)
            test_loader = DataLoader(UBFC_CF_Dataset(test_df, r, transform), batch_size=16, shuffle=False)
            
            model = MultimodalThreatModel(phys_dim=2, vision_backbone="mobilenet_v3_small", fusion='cgf', num_classes=2).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Fast 1-epoch pilot training
            model.train()
            for img_clean, img_scar, _, phys, y, _ in train_loader:
                img_clean, img_scar = img_clean.to(device), img_scar.to(device)
                phys, y = phys.to(device), y.to(device)
                
                optimizer.zero_grad()
                out_scar = model(img_scar, phys)
                loss_cls = F.cross_entropy(out_scar.logits, y)
                
                out_clean = model(img_clean, phys)
                p_scar = F.softmax(out_scar.logits, dim=1)
                p_clean = F.softmax(out_clean.logits, dim=1)
                
                loss_cf = jsd(p_clean, p_scar)
                loss = loss_cls + (0.5 * loss_cf)
                
                loss.backward()
                optimizer.step()
                
            # OOF Inference
            model.eval()
            with torch.no_grad():
                for img_clean, img_scar, img_sham, phys, y, a in test_loader:
                    img_scar, phys = img_scar.to(device), phys.to(device)
                    out = model(img_scar, phys)
                    probs = F.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
                    
                    for i in range(len(y)):
                        oof_predictions[r].append({
                            'y_true': y[i].item(),
                            'prob': probs[i],
                            'a_true': a[i].item(),
                            'fold': fold_idx
                        })
            
            # Save Fold 1 model for the Causal Sham Audit later
            if fold_idx == 1:
                torch.save(model.state_dict(), out_dir / f"eqarnb_{r}_fold1.pt")
                
            print(f"Fold {fold_idx}/7 completed.")
            fold_idx += 1
            
        pd.DataFrame(oof_predictions[r]).to_csv(out_dir / f"oof_preds_{r}.csv", index=False)
        
    print("\nPilot training complete. OOF predictions and checkpoints saved.")

if __name__ == '__main__':
    main()
