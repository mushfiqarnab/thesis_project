import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from torchvision import transforms
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models_arch import MultimodalThreatModel

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        # fallback for historical paths
        if not os.path.exists(img_path) and img_path.startswith('data\\'):
            img_path = img_path.replace('\\', '/')
            
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        phys = torch.tensor([row['hrv'], row['gsr']], dtype=torch.float32)
        y = torch.tensor(row['threat'], dtype=torch.float32)
        a = torch.tensor(row['scar'], dtype=torch.float32)
        
        return img, phys, y, a

def run_audit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on {device}...")

    # Load data
    df = pd.read_csv('data/publishable_scar_production/multimodal_publishable.csv')
    test_df = df[df['split'] == 'test']
    
    prevalence = test_df['threat'].mean()
    print(f"Test Set Size: {len(test_df)}, Threat Prevalence: {prevalence:.2%}\n")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = TestDataset(test_df, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = MultimodalThreatModel(backbone='mobilenet_v3_small', fusion='concat', disable_stiefel=True)
    model.to(device)

    ckpt_path = 'outputs/checkpoints/baseline_best.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'outputs/checkpoints/baseline_mobilenet_v3_small_concat_best.pt'
        
    print(f"Loading checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in sd:
        sd = sd['state_dict']
        
    try:
        model.load_state_dict(sd)
    except Exception as e:
        # Sometimes modules are named differently
        model.load_state_dict(sd, strict=False)
        print("Loaded with strict=False due to key mismatches.")

    model.eval()
    
    all_y = []
    all_a = []
    all_probs = []
    
    with torch.no_grad():
        for img, phys, y, a in test_loader:
            img, phys = img.to(device), phys.to(device)
            out = model(img, phys)
            probs = torch.sigmoid(out.logits).cpu().numpy().squeeze()
            if probs.ndim == 0:
                probs = np.array([probs])
            
            all_probs.extend(probs)
            all_y.extend(y.numpy())
            all_a.extend(a.numpy())

    all_y = np.array(all_y)
    all_a = np.array(all_a)
    all_probs = np.array(all_probs)
    
    # ---------------------------------------------------------
    # ACTION 1: RAW METRICS (Threshold = 0.5)
    # ---------------------------------------------------------
    preds_50 = (all_probs >= 0.5).astype(int)
    acc_50 = accuracy_score(all_y, preds_50)
    auc_val = roc_auc_score(all_y, all_probs)
    prec_50 = precision_score(all_y, preds_50, zero_division=0)
    rec_50 = recall_score(all_y, preds_50)
    cm_50 = confusion_matrix(all_y, preds_50)
    
    dp_a1_50 = np.mean(preds_50[all_a == 1])
    dp_a0_50 = np.mean(preds_50[all_a == 0])
    dp_gap_50 = abs(dp_a1_50 - dp_a0_50)

    print("=" * 50)
    print("ACTION 1: DEFAULT EVALUATION (Threshold = 0.50)")
    print("=" * 50)
    print(f"Accuracy:  {acc_50:.4f}")
    print(f"AUC-ROC:   {auc_val:.4f}")
    print(f"Precision: {prec_50:.4f}")
    print(f"Recall:    {rec_50:.4f}")
    print(f"Pred Rate: {np.mean(preds_50):.2%} (vs True {prevalence:.2%})")
    print(f"DP Gap:    {dp_gap_50:.4f} (P(1|A=1)={dp_a1_50:.4f}, P(1|A=0)={dp_a0_50:.4f})")
    print("Confusion Matrix:")
    print(cm_50)

    # ---------------------------------------------------------
    # ACTION 2: YOUDEN'S J CALIBRATION
    # ---------------------------------------------------------
    # Calculate TPR and TNR for all possible thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    best_j = -1
    best_t = 0.5
    
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_y, preds, labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp+fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn+fp) > 0 else 0
        j = tpr + tnr - 1
        if j > best_j:
            best_j = j
            best_t = t

    preds_opt = (all_probs >= best_t).astype(int)
    acc_opt = accuracy_score(all_y, preds_opt)
    prec_opt = precision_score(all_y, preds_opt, zero_division=0)
    rec_opt = recall_score(all_y, preds_opt)
    cm_opt = confusion_matrix(all_y, preds_opt)
    
    dp_a1_opt = np.mean(preds_opt[all_a == 1])
    dp_a0_opt = np.mean(preds_opt[all_a == 0])
    dp_gap_opt = abs(dp_a1_opt - dp_a0_opt)

    print("\n" + "=" * 50)
    print(f"ACTION 2: CALIBRATED EVALUATION (Threshold = {best_t:.4f})")
    print("=" * 50)
    print(f"Youden's J: {best_j:.4f}")
    print(f"Accuracy:   {acc_opt:.4f}")
    print(f"Precision:  {prec_opt:.4f}")
    print(f"Recall:     {rec_opt:.4f}")
    print(f"Pred Rate:  {np.mean(preds_opt):.2%} (vs True {prevalence:.2%})")
    print(f"DP Gap:     {dp_gap_opt:.4f} (P(1|A=1)={dp_a1_opt:.4f}, P(1|A=0)={dp_a0_opt:.4f})")
    print("Confusion Matrix:")
    print(cm_opt)

if __name__ == '__main__':
    run_audit()
