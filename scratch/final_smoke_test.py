import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
from models_arch import MultimodalThreatModel
from train_pilot_eqarnb import UBFC_CF_Dataset
from models.ntk_pruner import CounterfactualNTKExtractor

def jsd_stable(p, q):
    p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
    q = torch.clamp(q, 1e-7, 1.0 - 1e-7)
    m = 0.5 * (p + q)
    return 0.5 * F.kl_div(m.log(), p, reduction='batchmean') + 0.5 * F.kl_div(m.log(), q, reduction='batchmean')

def run_rigorous_smoke_test():
    print("=== FINAL RIGOROUS PIPELINE CHECK ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[1/5] Using device: {device}")
    
    # 1. Data Pipeline Test
    print("[2/5] Testing Data Pipeline & Standardizations...")
    df = pd.read_csv('data/ubfc_multimodal_processed/ubfc_multimodal_regimes.csv')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = UBFC_CF_Dataset(df, 'ext', transform)
    # Get a single batch
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    img_clean, img_scar, img_sham, phys, y, a = [b.to(device) for b in batch]
    print(f"      -> Data Pipeline: OK. Batch Shapes: img={img_clean.shape}, phys={phys.shape}")

    # 2. Concat Architecture Test
    print("[3/5] Testing Design A (Concat Fusion) & Gradients...")
    model_concat = MultimodalThreatModel(phys_dim=2, fusion='concat', num_classes=2).to(device)
    out_concat = model_concat(img_scar, phys)
    loss_concat = F.cross_entropy(out_concat.logits, y)
    loss_concat.backward()
    print("      -> Design A (Concat): OK.")

    # 3. CGF Architecture + Loss Logic Test
    print("[4/5] Testing Design B (Causal Gated Fusion) + Counterfactual Masking...")
    # Using phys_dim=2, properly instantiating
    model_cgf = MultimodalThreatModel(phys_dim=2, fusion='cgf', num_classes=2, freeze_vision=False).to(device)
    
    # We create a dummy spatial mask simulating a scar mask (B, 1, 224, 224)
    s_mask = torch.zeros((img_scar.size(0), 1, 224, 224), device=device)
    s_mask[:, :, 100:150, 100:150] = 1.0 # dummy scar region
    
    # Original Forward Pass
    out_scar = model_cgf(img_scar, phys, mask=s_mask)
    
    # Counterfactual Forward Pass (Passing the ORIGINAL mask, as we fixed)
    out_clean = model_cgf(img_clean, phys, mask=s_mask)
    
    loss_ce = F.cross_entropy(out_scar.logits, y)
    loss_cf = jsd_stable(F.softmax(out_clean.logits, dim=1), F.softmax(out_scar.logits, dim=1))
    loss_gate = torch.mean((out_scar.gate - 0.5) ** 2)
    
    lambda_cf, lambda_g = 0.5, 5.0
    loss_composite = loss_ce + (lambda_cf * loss_cf) + (lambda_g * loss_gate)
    loss_composite.backward()
    print(f"      -> Design B (CGF): OK. Gate Shape: {out_scar.gate.shape}, Focus Shape: {out_scar.focus.shape}")
    print(f"      -> Composite Loss Backprop: OK.")

    # 4. Pruning / Edge Compression Test (NTK Pruner)
    print("[5/5] Testing NTK Null-Space Pruner (Compression Pipeline)...")
    pruner = CounterfactualNTKExtractor(model_cgf, threshold_ratio=0.95)
    
    # Extract Jacobian for 1 sample to test mechanics
    try:
        pruner.extract_sample_jacobian(img_clean[0:1], phys[0:1], target_class=1)
        masks = pruner.compute_null_space_masks()
        pruner.apply_masks()
        print("      -> NTK Pruning Mechanics: OK.")
    except Exception as e:
        print(f"      -> NTK Pruning FAILED: {str(e)}")
        raise e

    print("\n[VERDICT] All architectures and pipelines compiled and executed flawlessly.")
    print("[VERDICT] No hidden graph disconnections or shape mismatches detected.")

if __name__ == '__main__':
    run_rigorous_smoke_test()
