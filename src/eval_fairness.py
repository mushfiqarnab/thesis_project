from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from dataset_fair import MultimodalCSVDatasetWithMask
from models import MultiModalViT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "csv" / "multimodal.csv"
CKPT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "baseline_best.pt"


def collate_keep_mask(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    phys = torch.stack([b[1] for b in batch], dim=0)
    y = torch.stack([b[2] for b in batch], dim=0)
    scar = torch.stack([b[3] for b in batch], dim=0)
    mask_paths = [b[4] for b in batch]
    return imgs, phys, y, scar, mask_paths


def dp_gap(pred, scar):
    pred1 = (pred == 1)
    s1 = (scar == 1)
    s0 = (scar == 0)
    p1_s1 = pred1[s1].float().mean().item() if s1.any() else 0.0
    p1_s0 = pred1[s0].float().mean().item() if s0.any() else 0.0
    return abs(p1_s1 - p1_s0), p1_s1, p1_s0


def eo_gap(pred, y, scar):
    s1 = (scar == 1)
    s0 = (scar == 0)

    def rates(mask):
        yy = y[mask]
        pp = pred[mask]
        pos = (yy == 1)
        neg = (yy == 0)
        tpr = ((pp == 1) & pos).float().sum() / (pos.float().sum() + 1e-9)
        fpr = ((pp == 1) & neg).float().sum() / (neg.float().sum() + 1e-9)
        return tpr.item(), fpr.item()

    tpr1, fpr1 = rates(s1) if s1.any() else (0.0, 0.0)
    tpr0, fpr0 = rates(s0) if s0.any() else (0.0, 0.0)
    gap = max(abs(tpr1 - tpr0), abs(fpr1 - fpr0))
    return gap, (tpr1, fpr1), (tpr0, fpr0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = MultimodalCSVDatasetWithMask(str(CSV_PATH), img_size=224)

    total = len(dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size
    _, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_keep_mask)

    model = MultiModalViT(num_classes=2).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    correct = 0
    total_n = 0
    all_pred, all_y, all_scar = [], [], []

    with torch.no_grad():
        for img, phys, y, scar, mask_paths in loader:
            img, phys, y = img.to(device), phys.to(device), y.to(device)
            logits = model(img, phys)
            pred = torch.argmax(logits, dim=1).cpu()

            correct += (pred == y.cpu()).sum().item()
            total_n += y.size(0)

            all_pred.append(pred)
            all_y.append(y.cpu())
            all_scar.append(scar.cpu().long())

    pred = torch.cat(all_pred)
    y = torch.cat(all_y)
    scar = torch.cat(all_scar)

    acc = correct / total_n
    dpg, p1s1, p1s0 = dp_gap(pred, scar)
    eog, (tpr1, fpr1), (tpr0, fpr0) = eo_gap(pred, y, scar)

    print("\nBaseline fairness ✅")
    print("Val Accuracy:", round(acc, 4))
    print("DP gap:", round(dpg, 4), "| P(pred=1|scar=1)=", round(p1s1, 4), " P(pred=1|scar=0)=", round(p1s0, 4))
    print("EO gap:", round(eog, 4))
    print("  scar=1: TPR=", round(tpr1, 4), "FPR=", round(fpr1, 4))
    print("  scar=0: TPR=", round(tpr0, 4), "FPR=", round(fpr0, 4))


if __name__ == "__main__":
    main()
