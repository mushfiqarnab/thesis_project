import sys
sys.path.insert(0, "src")

import numpy as np
import torch
from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models import MultimodalThreatModel
from torch.utils.data import DataLoader

def main():
    ds = MultimodalCSVDatasetWithCF("data/csv/multimodal.csv")
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_samples)

    b = next(iter(dl))
    model = MultimodalThreatModel(
        phys_dim=b["phys"].shape[1],
        vision_backbone="mobilenet_v3_small",
        fusion="cgf",
        num_classes=2,
    ).eval()

    with torch.no_grad():
        out = model(b["img"], b["phys"], mask=b["mask"])
        focus = out.focus.detach().cpu().numpy().reshape(-1)
        gate = out.gate.detach().cpu().numpy().reshape(-1)

    mask_sums = b["mask"].sum(dim=(1,2,3)).cpu().numpy()

    print("mask_sum: mean/min/max =", float(mask_sums.mean()), float(mask_sums.min()), float(mask_sums.max()))
    print("focus:    mean/min/max =", float(focus.mean()), float(focus.min()), float(focus.max()))
    print("gate:     mean/min/max =", float(gate.mean()), float(gate.min()), float(gate.max()))

if __name__ == "__main__":
    main()
