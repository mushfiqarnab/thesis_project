import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneGroupOut
from PIL import Image

class UBFCVisionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['clean_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(row['threat'], dtype=torch.long)
        return img, y

def train_k2_fold(train_df, test_df, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(UBFCVisionDataset(train_df, transform), batch_size=32, shuffle=True)
    test_loader = DataLoader(UBFCVisionDataset(test_df, transform), batch_size=32, shuffle=False)
    
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    # Fast 1 epoch train for baseline mapping
    for img, y in train_loader:
        img, y = img.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    all_y, all_probs = [], []
    with torch.no_grad():
        for img, y in test_loader:
            img = img.to(device)
            out = model(img)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.numpy())
            
    all_y = np.array(all_y)
    preds = (np.array(all_probs) >= 0.5).astype(int)
    
    acc = accuracy_score(all_y, preds)
    auc = roc_auc_score(all_y, all_probs) if len(np.unique(all_y)) > 1 else 0.5
    prec = precision_score(all_y, preds, zero_division=0)
    rec = recall_score(all_y, preds, zero_division=0)
    return acc, auc, prec, rec

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/ubfc_multimodal_processed/ubfc_multimodal_regimes.csv')
    
    # We want strict LOSO using the subject column
    logo = LeaveOneGroupOut()
    X_dummy = np.zeros(len(df))
    y = df['threat'].values
    groups = df['subject'].values
    
    k1_metrics = {'acc': [], 'auc': [], 'prec': [], 'rec': []}
    k2_metrics = {'acc': [], 'auc': [], 'prec': [], 'rec': []}
    
    print("Initiating 7-Fold LOSO Cross-Validation on Pilot Cohort...")
    
    fold_idx = 1
    for train_idx, test_idx in logo.split(X_dummy, y, groups):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # --- K1: Physiology-Only ---
        X_train_phys = train_df[['hrv', 'gsr']].values
        y_train = train_df['threat'].values
        X_test_phys = test_df[['hrv', 'gsr']].values
        y_test = test_df['threat'].values
        
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
        mlp.fit(X_train_phys, y_train)
        k1_probs = mlp.predict_proba(X_test_phys)[:, 1]
        k1_preds = (k1_probs >= 0.5).astype(int)
        
        k1_metrics['acc'].append(accuracy_score(y_test, k1_preds))
        k1_metrics['auc'].append(roc_auc_score(y_test, k1_probs) if len(np.unique(y_test)) > 1 else 0.5)
        k1_metrics['prec'].append(precision_score(y_test, k1_preds, zero_division=0))
        k1_metrics['rec'].append(recall_score(y_test, k1_preds, zero_division=0))
        
        # --- K2: Vision-Only ---
        acc_v, auc_v, prec_v, rec_v = train_k2_fold(train_df, test_df, device)
        k2_metrics['acc'].append(acc_v)
        k2_metrics['auc'].append(auc_v)
        k2_metrics['prec'].append(prec_v)
        k2_metrics['rec'].append(rec_v)
        
        print(f"Completed Fold {fold_idx}/7 (Test Subject: {groups[test_idx[0]]})...")
        fold_idx += 1

    def summarize(metrics):
        return {
            "Mean_Accuracy": float(np.mean(metrics['acc'])),
            "Mean_AUC": float(np.mean(metrics['auc'])),
            "Mean_Precision": float(np.mean(metrics['prec'])),
            "Mean_Recall": float(np.mean(metrics['rec']))
        }

    results = {
        "K1_Physiology_Baseline (LOSO)": summarize(k1_metrics),
        "K2_Vision_Baseline (LOSO)": summarize(k2_metrics)
    }
    
    with open('scratch/pilot_baselines_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- PILOT N=7 KILL-SWITCH BASELINES ESTABLISHED ---")
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    main()
