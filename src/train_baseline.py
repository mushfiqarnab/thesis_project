from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import MultimodalCSVDataset
from models import MultiModalViT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "csv" / "multimodal.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = MultimodalCSVDataset(str(CSV_PATH), img_size=224)

    total = len(dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    model = MultiModalViT(num_classes=2, freeze_vit=False).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    epochs = 5

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        correct = 0
        total_n = 0

        for img, phys, y, scar in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            img, phys, y = img.to(device), phys.to(device), y.to(device)

            logits = model(img, phys)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * img.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total_n += y.size(0)

        train_loss /= total_n
        train_acc = correct / total_n

        # ---- val ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total_n = 0

        with torch.no_grad():
            for img, phys, y, scar in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                img, phys, y = img.to(device), phys.to(device), y.to(device)

                logits = model(img, phys)
                loss = criterion(logits, y)

                val_loss += loss.item() * img.size(0)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == y).sum().item()
                total_n += y.size(0)

        val_loss /= total_n
        val_acc = correct / total_n

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = OUT_DIR / "baseline_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print("Saved best checkpoint:", ckpt_path)

    print("\nDONE ✅ Best val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
