from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import Image, ImageFilter
import torchvision.transforms as T

from dataset_fair import MultimodalCSVDatasetWithPaths
from models import MultiModalViT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "csv" / "multimodal.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collate_keep_paths(batch):
    # batch: list of (image_path, phys, y, scar, mask_path)
    image_paths = [b[0] for b in batch]
    phys = torch.stack([b[1] for b in batch], dim=0)
    y = torch.stack([b[2] for b in batch], dim=0)
    scar = torch.stack([b[3] for b in batch], dim=0)
    mask_paths = [b[4] for b in batch]
    return image_paths, phys, y, scar, mask_paths


def remove_scar_pil(img_pil: Image.Image, mask_pil: Image.Image, blur_radius=6.0, alpha=0.85):
    """
    Counterfactual generation: replace scar region with blurred skin using mask.
    """
    img = img_pil.convert("RGB")
    blur = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    img_np = torch.from_numpy(__import__("numpy").array(img)).float()
    blur_np = torch.from_numpy(__import__("numpy").array(blur)).float()
    m = torch.from_numpy(__import__("numpy").array(mask_pil.convert("L"))).float() / 255.0

    # soften mask edges
    m = torch.clamp(m, 0, 1).unsqueeze(-1)

    out = img_np * (1 - alpha * m) + blur_np * (alpha * m)
    out = torch.clamp(out, 0, 255).byte().numpy()
    return Image.fromarray(out)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # same transform for original and counterfactual
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    dataset = MultimodalCSVDatasetWithPaths(str(CSV_PATH))

    total = len(dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_keep_paths)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_keep_paths)

    model = MultiModalViT(num_classes=2).to(device)

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    # fairness weight (tune)
    lambda_cf = 1.0

    best_val = 0.0
    epochs = 5

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        correct = 0

        for image_paths, phys, y, scar, mask_paths in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            phys, y = phys.to(device), y.to(device)

            # Load and transform original images
            imgs = []
            imgs_cf = []
            cf_mask = []  # which samples have CF
            for i, p in enumerate(image_paths):
                img_pil = Image.open(p).convert("RGB")
                img_t = transform(img_pil)
                imgs.append(img_t)

                # build counterfactual only if scar==1 and mask exists
                if float(scar[i].item()) == 1.0 and mask_paths[i] and mask_paths[i].lower() != "nan":
                    try:
                        mask_pil = Image.open(mask_paths[i]).convert("L")
                        mask_pil = mask_pil.resize(img_pil.size)
                        img_cf_pil = remove_scar_pil(img_pil, mask_pil, blur_radius=6.0, alpha=0.85)
                        img_cf_t = transform(img_cf_pil)
                        imgs_cf.append(img_cf_t)
                        cf_mask.append(True)
                    except Exception:
                        cf_mask.append(False)
                else:
                    cf_mask.append(False)

            img = torch.stack(imgs, dim=0).to(device)

            logits = model(img, phys)
            loss_ce = ce(logits, y)

            # Counterfactual invariance loss on logits (only where CF exists)
            loss_cf = torch.tensor(0.0, device=device)
            if any(cf_mask):
                idxs = [j for j, ok in enumerate(cf_mask) if ok]
                # build matched phys for cf samples
                phys_cf = phys[idxs]

                # stack cf images in same order
                img_cf = torch.stack([imgs_cf[k] for k in range(len(idxs))], dim=0).to(device)

                logits_cf = model(img_cf, phys_cf)

                # match original logits subset
                logits_sub = logits[idxs]

                # invariance: make predictions similar
                loss_cf = F.mse_loss(logits_sub, logits_cf)

            loss = loss_ce + lambda_cf * loss_cf

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * img.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total_n += y.size(0)

        train_loss = total_loss / total_n
        train_acc = correct / total_n

        # --- validation (accuracy only) ---
        model.eval()
        correct = 0
        total_n = 0
        vloss = 0.0
        with torch.no_grad():
            for image_paths, phys, y, scar, mask_paths in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                phys, y = phys.to(device), y.to(device)
                imgs = []
                for p in image_paths:
                    img_pil = Image.open(p).convert("RGB")
                    imgs.append(transform(img_pil))
                img = torch.stack(imgs, dim=0).to(device)

                logits = model(img, phys)
                loss = ce(logits, y)
                vloss += loss.item() * img.size(0)

                pred = torch.argmax(logits, dim=1)
                correct += (pred == y).sum().item()
                total_n += y.size(0)

        vloss /= total_n
        vacc = correct / total_n

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} | val_loss={vloss:.4f} val_acc={vacc:.3f} | last_cf_loss={loss_cf.item():.6f}")

        if vacc > best_val:
            best_val = vacc
            ckpt = OUT_DIR / "counterfactual_fair_best.pt"
            torch.save(model.state_dict(), ckpt)
            print("Saved best checkpoint:", ckpt)

    print("\nDONE ✅ Best val_acc:", best_val)


if __name__ == "__main__":
    main()
