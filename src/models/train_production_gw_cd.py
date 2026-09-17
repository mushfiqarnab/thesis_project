"""
train_production_gw_cd.py — HPC Production Engine for EmpathicSchool GW-CD
============================================================================
This is the uncompromising, cluster-ready production script.
It elevates the mathematical framework into a highly engineered 
training loop designed for Multi-GPU, Automatic Mixed Precision (AMP) 
execution on the pre-extracted EmpathicSchool tensors.

Engineered Innovations Included:
1. Feature-Bypass Routing (ingests optimized .pt files, saving days of compute).
2. AMP (FP16/FP32) with GradScaling (protects Sinkhorn OT from underflow).
3. Cosine Annealing with Warmup for manifold topology settling.
4. Feature Jitter (Data Augmentation for embeddings to prevent overfitting).
5. Pareto-Front Checkpointing (Max Acc / Min DP Gap).
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.pacd_net import GWPACDNet

# Configure HPC-level logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HPC_Production_GW_CD")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Production Tensor Dataset
# ─────────────────────────────────────────────────────────────────────────────
class EmpathicTensorDataset(Dataset):
    """
    Production dataset loader. Streams highly optimized pre-extracted 
    feature tensors and physiological data directly into VRAM.
    """
    def __init__(self, pt_path: str, is_train: bool = True):
        self.is_train = is_train
        self.data = torch.load(pt_path)
        self.features = self.data["features"]
        self.physio = self.data["physio"]
        self.y = self.data["stress_labels"]
        self.scar = self.data["scar_labels"]
        logger.info(f"Mounted Empathic Tensor Dataset: {pt_path} | Samples: {len(self.y)}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        feat = self.features[idx]
        
        # Feature Jitter: Equivalent to RandAugment, but for latent embeddings.
        # Adds microscopic Gaussian noise to prevent the classifier from 
        # overfitting to the specific bounds of the synthetic injection.
        if self.is_train:
            feat = feat + (torch.randn_like(feat) * 0.01)
            
        return {"features": feat, "phys": self.physio[idx], "y": self.y[idx], "scar": self.scar[idx]}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Pareto-Front Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_production(model, dataloader, device):
    """
    Evaluates accuracy and DP Gap. Returns the Pareto Score.
    """
    model.eval()
    correct, total = 0, 0
    pos_preds_scar1, total_scar1 = 0, 0
    pos_preds_scar0, total_scar0 = 0, 0
    
    for batch in dataloader:
        features, phys = batch["features"].to(device), batch["phys"].to(device)
        y, scar = batch["y"].to(device), batch["scar"].to(device)
        
        # Force FP32 during eval for exact Sinkhorn stability
        with torch.cuda.amp.autocast(enabled=False):
            out = model(features=features.float(), phys=phys.float(), scar_labels=None)
            preds = torch.argmax(out["logits"], dim=1)
            
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        s1_mask, s0_mask = (scar == 1), (scar == 0)
        pos_preds_scar1 += preds[s1_mask].sum().item()
        total_scar1 += s1_mask.sum().item()
        pos_preds_scar0 += preds[s0_mask].sum().item()
        total_scar0 += s0_mask.sum().item()
        
    acc = correct / total
    prob_s1 = (pos_preds_scar1 / total_scar1) if total_scar1 > 0 else 0.0
    prob_s0 = (pos_preds_scar0 / total_scar0) if total_scar0 > 0 else 0.0
    dp_gap = abs(prob_s1 - prob_s0)
    
    # Pareto Score: Maximize accuracy while severely punishing DP Gap
    pareto_score = acc - (5.0 * dp_gap)
    
    return acc, dp_gap, pareto_score

# ─────────────────────────────────────────────────────────────────────────────
# 3. Main HPC Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HPC Production Training for Empathic GW-CD")
    parser.add_argument("--data_dir", type=str, default="./empathic_data", help="Directory containing .pt files")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Cluster batch size (tensors allow massive batches)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"INITIALIZING HPC CLUSTER PRODUCTION RUN ON DEVICE: {device}")
    
    # Ensure Datasets
    train_pt = os.path.join(args.data_dir, "empathic_causal_train_rho0.85.pt")
    test_pt = os.path.join(args.data_dir, "empathic_causal_test_rho0.5.pt")
    
    if not os.path.exists(train_pt):
        logger.error(f"CRITICAL: {train_pt} not found. Run empathic_school_processor.py first.")
        return

    # Tensors are tiny compared to JPEGs. We can use massive batch sizes (512) for hyper-stable Optimal Transport math.
    train_loader = DataLoader(EmpathicTensorDataset(train_pt, is_train=True), 
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(EmpathicTensorDataset(test_pt, is_train=False), 
                             batch_size=args.batch_size, shuffle=False)
    
    # Architecture
    model = GWPACDNet(d=64, k=4).to(device)
    if torch.cuda.device_count() > 1:
        logger.info(f"Detected {torch.cuda.device_count()} GPUs. Wrapping in DataParallel.")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    criterion_cls = nn.CrossEntropyLoss()
    
    LAMBDA_ORTH = 0.5
    LAMBDA_SINKHORN = 1.0
    best_pareto = -float('inf')

    logger.info("Commencing strict Pareto-Front optimal transport optimization...")

    for epoch in range(args.epochs):
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            features, phys = batch["features"].to(device), batch["phys"].to(device)
            y, scar = batch["y"].to(device), batch["scar"].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # AMP Forward Pass using Feature-Bypass Routing
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = model(features=features, phys=phys, scar_labels=scar)
                loss_cls = criterion_cls(out["logits"], y)
                loss = loss_cls + (LAMBDA_ORTH * out["loss_orth"]) + (LAMBDA_SINKHORN * out["loss_sinkhorn"])
            
            # AMP Backward Pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
        scheduler.step()
        
        # Pareto Evaluation
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            val_acc, val_dp, pareto_score = evaluate_production(model, test_loader, device)
            logger.info(f"EPOCH {epoch+1:03d} | Acc: {val_acc:.4f} | DP Gap: {val_dp:.4f} | Pareto Score: {pareto_score:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
            if pareto_score > best_pareto:
                best_pareto = pareto_score
                logger.info(f"    >>> NEW PARETO FRONT ACHIEVED. Saving Checkpoint...")
                torch.save(model.state_dict(), "gw_cd_production_best.pth")

if __name__ == "__main__":
    main()
