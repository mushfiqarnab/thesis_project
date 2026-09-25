import os
import sys
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, accuracy_score, precision_score
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path('src').resolve()))
from models_arch import MultimodalThreatModel

class PublishableDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        if not os.path.exists(img_path) and img_path.startswith('data\\'):
            img_path = img_path.replace('\\', '/')
            
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        phys = torch.tensor([row['hrv'], row['gsr']], dtype=torch.float32)
        y = torch.tensor(row['threat'], dtype=torch.long) # num_classes=2 usually implies long targets
        a = torch.tensor(row['scar'], dtype=torch.float32)
        
        return img, phys, y, a

def evaluate_model(model_path, fusion_type, loader, device):
    print(f"Evaluating {model_path}...")
    model = MultimodalThreatModel(
        phys_dim=2,
        vision_backbone="mobilenet_v3_small",
        fusion=fusion_type,
        num_classes=2
    ).to(device)
    
    # Load state
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    try:
        model.load_state_dict(state)
    except Exception as e:
        model.load_state_dict(state, strict=False)
        print("Loaded with strict=False")

    model.eval()
    
    all_y = []
    all_probs = []
    
    with torch.no_grad():
        for img, phys, y, a in loader:
            img, phys = img.to(device), phys.to(device)
            out = model(img, phys)
            
            # Since num_classes=2, use softmax and take prob of class 1
            probs = F.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            
            all_probs.extend(probs)
            all_y.extend(y.numpy())
            
    all_y = np.array(all_y)
    all_probs = np.array(all_probs)
    
    # Metrics @ 0.5
    preds_50 = (all_probs >= 0.5).astype(int)
    tn50, fp50, fn50, tp50 = confusion_matrix(all_y, preds_50, labels=[0,1]).ravel()
    auc_val = roc_auc_score(all_y, all_probs)
    brier = brier_score_loss(all_y, all_probs)
    acc_50 = accuracy_score(all_y, preds_50)
    prec_50 = precision_score(all_y, preds_50, zero_division=0)
    
    # Youden's J Optimization
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
    tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(all_y, preds_opt, labels=[0,1]).ravel()
    acc_opt = accuracy_score(all_y, preds_opt)
    prec_opt = precision_score(all_y, preds_opt, zero_division=0)
    
    return {
        "ROC-AUC": float(auc_val),
        "Brier Score": float(brier),
        "Threshold_0.5": {
            "Accuracy": float(acc_50),
            "Precision": float(prec_50),
            "TP": int(tp50), "TN": int(tn50), "FP": int(fp50), "FN": int(fn50)
        },
        "Youden_Optimized": {
            "Optimal_Threshold": float(best_t),
            "Max_J_Statistic": float(best_j),
            "Accuracy": float(acc_opt),
            "Precision": float(prec_opt),
            "TP": int(tp_opt), "TN": int(tn_opt), "FP": int(fp_opt), "FN": int(fn_opt)
        }
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/publishable_scar_production/multimodal_publishable.csv')
    test_df = df[df['split'] == 'test']
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = PublishableDataset(test_df, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    results = {}
    
    # Checkpoint paths
    ckpt_base1 = 'outputs/checkpoints/baseline_best.pt'
    ckpt_base2 = 'outputs/checkpoints/baseline_mobilenet_v3_small_concat_best.pt'
    
    # Try to find a counterfactual checkpoint
    ckpt_cf = 'outputs/checkpoints/counterfactual_concat_js_mobilenet_v3_small_multimodal_10k_unbiased_best.pt'
    if not os.path.exists(ckpt_cf):
        ckpt_cf = 'outputs/checkpoints/counterfactual_concat_js_mobilenet_v3_small_multimodal_publishable_best_baseline_production.pt'
        
    for name, path, fusion in [
        ("Baseline_A (baseline_best.pt)", ckpt_base1, 'concat'),
        ("Baseline_B (baseline_mobilenet_v3_small_concat_best.pt)", ckpt_base2, 'concat'),
        ("Counterfactual", ckpt_cf, 'concat')
    ]:
        if os.path.exists(path):
            results[name] = evaluate_model(path, fusion, test_loader, device)
        else:
            results[name] = {"error": f"File not found: {path}"}
            
    with open('scratch_eval_definitive_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    main()
