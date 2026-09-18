import os, json, sys, subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from src.models_arch import MultimodalThreatModel
from scratch.dry_run import set_seed, eval_metrics_per_subject

# 1. Boilerplate / Setup
print("\n=== RIGOROUS DIAGNOSTICS SUITE ===")
try:
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('ascii').strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"]).decode('ascii').strip()
    is_dirty = "DIRTY" if len(dirty) > 0 else "CLEAN"
except:
    git_hash, is_dirty = "unknown", "unknown"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Git Hash: {git_hash} ({is_dirty})")
print(f"Device: {device}")

# Fold 0 definition
with open("data/csv/folds_dry_run.json") as f:
    folds_data = json.load(f)
f0 = folds_data["folds"][0]
test_subs = f0["test"]

# Data loading
u = pd.read_csv("data/csv/multimodal_10k_unbiased.csv")
b = pd.read_csv("data/csv/multimodal_10k.csv")
ds_u = MultimodalCSVDatasetWithCF("data/csv/multimodal_10k_unbiased.csv")
ds_b = MultimodalCSVDatasetWithCF("data/csv/multimodal_10k.csv")

# 5. Selection Rule / Baselines
print("\n=== VAL SELECTION RULES ===")
# Compute Val majority for unbiased (which is used for selection)
val_idx = u[u.subject.isin(f0["val"])].index.tolist()
val_df = u.iloc[val_idx]
val_maj = max((val_df.threat == 1).mean(), (val_df.threat == 0).mean())
print(f"Val Majority (Fold 0): {val_maj:.4f}")
print("Selection Rule: score = Acc - w_eo * max(TPR_gap, FPR_gap) - w_cf * CF_gap")
print("Guard: Reject if Var < 0.005 or Acc <= (Majority + 0.01)")

def get_subset_loader(ds, df, subjects):
    idx = df[df.subject.isin(subjects)].index.tolist()
    return DataLoader(Subset(ds, idx), batch_size=64, shuffle=False, collate_fn=collate_samples)

def train_baseline(df, ds, name, disable_stiefel=True):
    print(f"\n--- Training {name} Baseline ---")
    train_idx = df[df.subject.isin(f0["train"])].index.tolist()
    val_idx = df[df.subject.isin(f0["val"])].index.tolist()
    
    train_phys = df.iloc[train_idx][["hrv", "gsr"]].to_numpy(dtype=np.float32)
    phys_mu = torch.tensor(train_phys.mean(axis=0), device=device)
    phys_sigma = torch.tensor(train_phys.std(axis=0).clip(min=1e-6), device=device)
    
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=64, shuffle=True, collate_fn=collate_samples)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=64, shuffle=False, collate_fn=collate_samples)
    
    set_seed(0)
    model = MultimodalThreatModel(phys_dim=2, fusion="concat", disable_stiefel=disable_stiefel).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ce = nn.CrossEntropyLoss()
    
    best_score = -1e9
    best_state = None
    
    stiefel_devs = []
    
    for epoch in range(1, 16):
        model.train()
        for b_batch in train_loader:
            img = b_batch["img"].to(device)
            phys = (b_batch["phys"].to(device) - phys_mu) / phys_sigma
            y = b_batch["y"].to(device)
            mask = b_batch["mask"].to(device)
            out = model(img, phys, mask=mask)
            loss = ce(out.logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            
        # Log Stiefel Dev (Task 6)
        if not disable_stiefel:
            with torch.no_grad():
                W = model.fuse.cls[0].get_stiefel_weight()
                I = torch.eye(W.size(0), device=W.device)
                dev = torch.linalg.matrix_norm(W @ W.T - I, ord='fro').item()
                stiefel_devs.append(dev)
                
        val = eval_metrics_per_subject(model, val_loader, device, phys_mu, phys_sigma)
        score = val["acc"]
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if not disable_stiefel:
        print(f"Stiefel deviations over epochs: {[round(d, 6) for d in stiefel_devs]}")
        
    model.load_state_dict(best_state)
    model.eval()
    return model, phys_mu, phys_sigma

# Train standard baselines
model_u, mu_u, sig_u = train_baseline(u, ds_u, "Unbiased", disable_stiefel=True)
model_b, mu_b, sig_b = train_baseline(b, ds_b, "Biased", disable_stiefel=True)

# Train a Stiefel baseline just to log the deviations (Task 6)
_, _, _ = train_baseline(b, ds_b, "Biased (Stiefel ON)", disable_stiefel=False)

# 1. CF Handling on Scar Rows
print("\n=== TASK 1: CF-Gap on Scar Rows Only ===")
test_idx = u[u.subject.isin(test_subs)].index.tolist()
loader_test = DataLoader(Subset(ds_u, test_idx), batch_size=128, shuffle=False, collate_fn=collate_samples)

def eval_cf_isolated(model, loader, mu, sig):
    cf_abs_all = 0.0
    cf_flip_all = 0
    cf_abs_scar = 0.0
    cf_flip_scar = 0
    n_cf = 0
    n_total = 0
    with torch.no_grad():
        for batch in loader:
            img, img_cf = batch["img"].to(device), batch["img_cf"].to(device)
            phys = (batch["phys"].to(device) - mu) / sig
            mask = batch["mask"].to(device)
            scar = batch["scar"]
            has_cf = batch["has_cf"]
            
            p = torch.softmax(model(img, phys, mask=mask).logits, dim=1)[:, 1]
            p_cf = torch.softmax(model(img_cf, phys, mask=mask).logits, dim=1)[:, 1]
            dif = (p - p_cf).abs()
            flip = (p >= 0.5) != (p_cf >= 0.5)
            
            # Overall (how the script did it previously)
            cf_abs_all += dif[has_cf].sum().item()
            cf_flip_all += flip[has_cf].sum().item()
            n_total += has_cf.sum().item()
            
            # Scar rows only
            scar_mask = (scar == 1) & has_cf
            cf_abs_scar += dif[scar_mask].sum().item()
            cf_flip_scar += flip[scar_mask].sum().item()
            n_cf += scar_mask.sum().item()
            
    print(f"Total Rows with CF Generated: {n_total} (includes scar rows and whatever else triggered it)")
    print(f"Scar Rows with CF: {n_cf}")
    if n_total > 0:
        print(f"  All-Row CF-Gap: {cf_abs_all/n_total:.4f} | Flips: {cf_flip_all/n_total:.4f}")
    if n_cf > 0:
        print(f"  Scar-Row CF-Gap: {cf_abs_scar/n_cf:.4f} | Flips: {cf_flip_scar/n_cf:.4f}")
        
    # Save a sample grid of 4
    import torchvision
    to_pil = torchvision.transforms.ToPILImage()
    inv_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inv_std = [1/0.229, 1/0.224, 1/0.225]
    denorm = torchvision.transforms.Normalize(mean=inv_mean, std=inv_std)
    
    # Just take the last batch's first 4 scar samples
    mask_scar = (batch["scar"] == 1) & batch["has_cf"]
    indices = torch.where(mask_scar)[0][:4]
    
    if len(indices) > 0:
        imgs = [to_pil(denorm(batch["img"][i])) for i in indices]
        masks = [to_pil(batch["mask"][i]) for i in indices]
        cfs = [to_pil(denorm(batch["img_cf"][i])) for i in indices]
        
        from PIL import Image
        grid = Image.new('RGB', (224 * 3, 224 * len(indices)))
        for row, (im, mk, cf) in enumerate(zip(imgs, masks, cfs)):
            grid.paste(im, (0, row * 224))
            grid.paste(mk.convert('RGB'), (224, row * 224))
            grid.paste(cf, (448, row * 224))
        grid.save("scratch/task1_cf_samples.jpg")
        print("Saved 4 side-by-side images to scratch/task1_cf_samples.jpg")

print("Biased Model CF Gaps (Unbiased Test Rows):")
eval_cf_isolated(model_b, loader_test, mu_b, sig_b)

# 2. Blur Control
print("\n=== TASK 2: Blur Control (Borrowed Masks) ===")
# Take clean images (scar==0) and blur them using masks from scar==1
clean_idx = u[(u.subject.isin(test_subs)) & (u.scar == 0)].index.tolist()
scar_idx = u[(u.subject.isin(test_subs)) & (u.scar == 1)].index.tolist()
clean_loader = DataLoader(Subset(ds_u, clean_idx), batch_size=len(clean_idx), shuffle=False, collate_fn=collate_samples)
scar_loader = DataLoader(Subset(ds_u, scar_idx), batch_size=len(clean_idx), shuffle=True, collate_fn=collate_samples) # shuffle to get random masks

c_batch = next(iter(clean_loader))
s_batch = next(iter(scar_loader))

# Blur them manually using the logic from dataset
from src.dataset_fair import remove_scar_pil
import torchvision.transforms as T
from PIL import Image

def borrowed_blur(c_img_tensor, s_mask_tensor):
    to_pil = T.ToPILImage()
    to_tensor = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    blurred_tensors = []
    
    # Denormalize
    inv_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inv_std = [1/0.229, 1/0.224, 1/0.225]
    denorm = T.Normalize(mean=inv_mean, std=inv_std)
    
    for i in range(len(c_img_tensor)):
        c_pil = to_pil(denorm(c_img_tensor[i]))
        m_pil = to_pil(s_mask_tensor[i])
        blurred_pil = remove_scar_pil(c_pil, m_pil, blur_radius=6.0, alpha=0.85)
        blurred_tensors.append(to_tensor(blurred_pil))
    return torch.stack(blurred_tensors)

print("Applying borrowed blur masks to clean images...")
blur_control_img = borrowed_blur(c_batch["img"], s_batch["mask"]).to(device)
orig_img = c_batch["img"].to(device)
phys = (c_batch["phys"].to(device) - mu_b) / sig_b

with torch.no_grad():
    p_orig = torch.softmax(model_b(orig_img, phys, mask=torch.zeros_like(s_batch["mask"]).to(device)).logits, dim=1)[:, 1]
    p_blur = torch.softmax(model_b(blur_control_img, phys, mask=s_batch["mask"].to(device)).logits, dim=1)[:, 1]
    dif = (p_orig - p_blur).abs().mean().item()
    print(f"Control |delta_p| (Biased Model on Clean Images with Borrowed Blur): {dif:.4f}")

# 3. Component Accuracies (Majority, Scar-only, Phys-only, Vision-only) per subject, grouped by Label x Scar
print("\n=== TASK 3: Component Accuracies ===")
def worst_group_acc(y, pred, scar):
    g_accs = []
    for yy in [0, 1]:
        for ss in [0, 1]:
            mask = (y == yy) & (scar == ss)
            if mask.sum() > 0:
                g_accs.append((pred[mask] == y[mask]).mean())
    return min(g_accs) if g_accs else 0.0

test_df = u[u.subject.isin(test_subs)]

# Phys only (Train on fold 0 train)
train_df = u[u.subject.isin(f0["train"])]
X_tr = train_df[["hrv", "gsr"]].to_numpy()
y_tr = train_df["threat"].to_numpy()
lr = LogisticRegression().fit(X_tr, y_tr)
gbm = HistGradientBoostingClassifier().fit(X_tr, y_tr)

for subj in test_subs:
    print(f"\nSubject {subj}:")
    mask = test_df.subject == subj
    sdf = test_df[mask]
    y_s = sdf["threat"].to_numpy()
    scar_s = sdf["scar"].to_numpy()
    
    # Majority
    p_maj = np.zeros_like(y_s) + int(y_tr.mean() > 0.5)
    acc_maj = (p_maj == y_s).mean()
    wg_maj = worst_group_acc(y_s, p_maj, scar_s)
    print(f"  Majority   -> Acc: {acc_maj:.4f} | WGA: {wg_maj:.4f}")
    
    # Scar only (if scar==1 predict 1)
    p_scar = scar_s.copy()
    acc_scar = (p_scar == y_s).mean()
    wg_scar = worst_group_acc(y_s, p_scar, scar_s)
    print(f"  Scar-Only  -> Acc: {acc_scar:.4f} | WGA: {wg_scar:.4f}")
    
    # Phys only (LR)
    p_lr = lr.predict(sdf[["hrv", "gsr"]].to_numpy())
    acc_lr = (p_lr == y_s).mean()
    wg_lr = worst_group_acc(y_s, p_lr, scar_s)
    print(f"  Phys (LR)  -> Acc: {acc_lr:.4f} | WGA: {wg_lr:.4f}")
    
    # Phys only (GBM)
    p_gbm = gbm.predict(sdf[["hrv", "gsr"]].to_numpy())
    acc_gbm = (p_gbm == y_s).mean()
    wg_gbm = worst_group_acc(y_s, p_gbm, scar_s)
    print(f"  Phys (GBM) -> Acc: {acc_gbm:.4f} | WGA: {wg_gbm:.4f}")

# 4. Cross Evaluate Baselines
print("\n=== TASK 4: Cross Evaluation ===")

def eval_model(model, name, df, ds, csv_name, mu, sig):
    print(f"\nModel: {name} | Eval CSV: {csv_name}")
    for subj in test_subs:
        idx = df[df.subject == subj].index.tolist()
        loader = DataLoader(Subset(ds, idx), batch_size=64, shuffle=False, collate_fn=collate_samples)
        
        preds, ys, scars = [], [], []
        with torch.no_grad():
            for b_batch in loader:
                img = b_batch["img"].to(device)
                phys = (b_batch["phys"].to(device) - mu) / sig
                mask = b_batch["mask"].to(device)
                p = torch.softmax(model(img, phys, mask=mask).logits, dim=1)[:, 1]
                preds.append((p >= 0.5).cpu().numpy().astype(int))
                ys.append(b_batch["y"].numpy())
                scars.append(b_batch["scar"].numpy())
                
        preds = np.concatenate(preds)
        ys = np.concatenate(ys)
        scars = np.concatenate(scars)
        
        acc = (preds == ys).mean()
        wga = worst_group_acc(ys, preds, scars)
        
        # EO counts
        def eo_stats(g):
            m = (scars == g)
            tp = ((preds[m] == 1) & (ys[m] == 1)).sum()
            fn = ((preds[m] == 0) & (ys[m] == 1)).sum()
            fp = ((preds[m] == 1) & (ys[m] == 0)).sum()
            tn = ((preds[m] == 0) & (ys[m] == 0)).sum()
            return tp, fn, fp, tn
            
        tp1, fn1, fp1, tn1 = eo_stats(1)
        tp0, fn0, fp0, tn0 = eo_stats(0)
        
        tpr1 = tp1 / max(tp1 + fn1, 1)
        tpr0 = tp0 / max(tp0 + fn0, 1)
        fpr1 = fp1 / max(fp1 + tn1, 1)
        fpr0 = fp0 / max(fp0 + tn0, 1)
        
        print(f"  Subject {subj}: Acc: {acc:.4f} | WGA: {wga:.4f}")
        print(f"    Scar=1 -> TP:{tp1} FN:{fn1} FP:{fp1} TN:{tn1} (TPR:{tpr1:.2f}, FPR:{fpr1:.2f})")
        print(f"    Scar=0 -> TP:{tp0} FN:{fn0} FP:{fp0} TN:{tn0} (TPR:{tpr0:.2f}, FPR:{fpr0:.2f})")
        print(f"    Gaps   -> TPR Gap: {abs(tpr1-tpr0):.4f} | FPR Gap: {abs(fpr1-fpr0):.4f}")

eval_model(model_u, "Unbiased", u, ds_u, "Unbiased CSV", mu_u, sig_u)
eval_model(model_u, "Unbiased", b, ds_b, "Biased CSV", mu_u, sig_u)
eval_model(model_b, "Biased", u, ds_u, "Unbiased CSV", mu_b, sig_b)
eval_model(model_b, "Biased", b, ds_b, "Biased CSV", mu_b, sig_b)
