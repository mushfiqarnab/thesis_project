import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MultimodalCSVDataset
from models import MultiModalViT


def compute_fairness(pred, y, scar):
    pred_pos = (pred == 1)

    # Demographic parity gap
    dp1 = pred_pos[scar == 1].mean() if np.any(scar == 1) else 0.0
    dp0 = pred_pos[scar == 0].mean() if np.any(scar == 0) else 0.0
    dp_gap = abs(dp1 - dp0)

    # Equalized odds: TPR & FPR gaps
    def rate(mask_group):
        return pred_pos[mask_group].mean() if np.any(mask_group) else 0.0

    tpr1 = rate((y == 1) & (scar == 1))
    tpr0 = rate((y == 1) & (scar == 0))
    fpr1 = rate((y == 0) & (scar == 1))
    fpr0 = rate((y == 0) & (scar == 0))

    return dp_gap, abs(tpr1 - tpr0), abs(fpr1 - fpr0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MultimodalCSVDataset("data/csv/multimodal.csv", img_size=224)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    model = MultiModalViT(num_classes=2).to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/baseline_best.pt", map_location=device))
    model.eval()

    all_pred, all_y, all_scar = [], [], []

    with torch.no_grad():
        for img, phys, y, scar in loader:
            img, phys = img.to(device), phys.to(device)
            logits = model(img, phys)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

            all_pred.append(pred)
            all_y.append(y.numpy())
            all_scar.append(scar.numpy())

    pred = np.concatenate(all_pred)
    y = np.concatenate(all_y)
    scar = np.concatenate(all_scar)

    acc = (pred == y).mean()
    dp_gap, tpr_gap, fpr_gap = compute_fairness(pred, y, scar)

    print("Accuracy:", acc)
    print("Demographic Parity gap:", dp_gap)
    print("Equalized Odds TPR gap:", tpr_gap)
    print("Equalized Odds FPR gap:", fpr_gap)


if __name__ == "__main__":
    main()
