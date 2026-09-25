import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path('src').resolve()))
from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models_arch import MultimodalThreatModel

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

def get_10_samples(dataset):
    idx_1 = []
    idx_0 = []
    for i in range(len(dataset)):
        sample = dataset[i]
        y = sample.y if hasattr(sample, 'y') else sample['y']
        y = y.item() if hasattr(y, 'item') else y
        if y == 1 and len(idx_1) < 5:
            idx_1.append(i)
        elif y == 0 and len(idx_0) < 5:
            idx_0.append(i)
        if len(idx_1) == 5 and len(idx_0) == 5:
            break
    return idx_1 + idx_0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    csv_path = "data/csv/multimodal.csv"
    ds = MultimodalCSVDatasetWithCF(csv_path, drop_nan_rows=True, verbose=False)
    
    indices = get_10_samples(ds)
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=10, shuffle=False, collate_fn=collate_samples)
    
    phys_dim = ds[0].phys.numel()
    model = MultimodalThreatModel(
        phys_dim=phys_dim,
        vision_backbone="mobilenet_v3_small",
        fusion="cgf",
        num_classes=2
    ).to(device)
    
    model.train()
    # Disable dropout for this strict overfit test
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
            
    # Disable data augmentation by overriding dataset transforms (if any). The dataset_fair doesn't seem to apply random augmentations by default unless specified, but we can bypass it if needed.
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    grad_norms = {"v_proj": [], "p_proj": []}
    
    def get_grad_norm(name):
        def hook(grad):
            norm = grad.norm().item()
            grad_norms[name].append(norm)
        return hook

    # Register hooks
    for name, param in model.named_parameters():
        if "fuse.v_proj.weight" in name:
            param.register_hook(get_grad_norm("v_proj"))
        elif "fuse.p_proj.weight" in name:
            param.register_hook(get_grad_norm("p_proj"))

    print("Starting Micro-Batch Overfit Test...")
    for epoch in range(1, 101):
        for batch in loader:
            img = batch["img"].to(device)
            phys = batch["phys"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)
            
            optimizer.zero_grad()
            out = model(img, phys, mask=mask)
            logits = out.logits
            loss = criterion(logits, y)
            loss.backward()
            
            if epoch == 1 or epoch == 50:
                print(f"Epoch {epoch} Gradient Norms:")
                print(f"  v_proj: {grad_norms['v_proj'][-1]:.6f}" if grad_norms['v_proj'] else "  v_proj: 0.000000")
                print(f"  p_proj: {grad_norms['p_proj'][-1]:.6f}" if grad_norms['p_proj'] else "  p_proj: 0.000000")
                
            optimizer.step()
            
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean().item()
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
