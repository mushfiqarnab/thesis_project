import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
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
    
    # Fast 1 epoch train for baseline ceiling mapping (In production: 20 epochs)
    model.train()
    for img, y in train_loader:
        img, y = img.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    all_y = []
    all_probs = []
    with torch.no_grad():
        for img, y in test_loader:
            img = img.to(device)
            out = model(img)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.numpy())
            
    acc = accuracy_score(all_y, (np.array(all_probs) >= 0.5).astype(int))
    auc = roc_auc_score(all_y, all_probs) if len(np.unique(all_y)) > 1 else 0.5
    return acc, auc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/ubfc_multimodal_processed/ubfc_multimodal_regimes.csv')
    
    k1_accs, k1_aucs = [], []
    k2_accs, k2_aucs = [], []
    
    print("Initiating 5-Fold Cross-Validation for Kill-Switch Baselines...")
    
    for fold in sorted(df['fold'].unique()):
        train_df = df[df['fold'] != fold]
        test_df = df[df['fold'] == fold]
        
        # --- K1: Physiology-Only (MLP) ---
        X_train = train_df[['hrv', 'gsr']].values
        y_train = train_df['threat'].values
        X_test = test_df[['hrv', 'gsr']].values
        y_test = test_df['threat'].values
        
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
        mlp.fit(X_train, y_train)
        k1_probs = mlp.predict_proba(X_test)[:, 1]
        k1_preds = (k1_probs >= 0.5).astype(int)
        
        k1_accs.append(accuracy_score(y_test, k1_preds))
        k1_aucs.append(roc_auc_score(y_test, k1_probs) if len(np.unique(y_test)) > 1 else 0.5)
        
        # --- K2: Vision-Only (MobileNet-V3) ---
        # Train vision backbone exclusively on clean frames
        acc_v, auc_v = train_k2_fold(train_df, test_df, device)
        k2_accs.append(acc_v)
        k2_aucs.append(auc_v)
        print(f"Completed Fold {fold}/4...")

    results = {
        "K1_Physiology_Baseline": {
            "Mean_Accuracy": float(np.mean(k1_accs)),
            "Std_Accuracy": float(np.std(k1_accs)),
            "Mean_AUC": float(np.mean(k1_aucs)),
            "Std_AUC": float(np.std(k1_aucs))
        },
        "K2_Vision_Baseline": {
            "Mean_Accuracy": float(np.mean(k2_accs)),
            "Std_Accuracy": float(np.std(k2_accs)),
            "Mean_AUC": float(np.mean(k2_aucs)),
            "Std_AUC": float(np.std(k2_aucs))
        }
    }
    
    with open('scratch/k_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- KILL-SWITCH BASELINES ESTABLISHED ---")
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    main()
