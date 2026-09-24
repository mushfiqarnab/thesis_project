from __future__ import annotations

from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models_arch import MultimodalThreatModel, count_trainable_params


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "csv" / "multimodal_10k_unbiased.csv"

OUT_CKPT = PROJECT_ROOT / "outputs" / "checkpoints"
OUT_REP = PROJECT_ROOT / "outputs" / "reports"
OUT_CKPT.mkdir(parents=True, exist_ok=True)
OUT_REP.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_or_load_split(csv_path: Path, n: int, seed: int = 42, val_ratio: float = 0.2):
    # Use dataset-specific split file name (matches project convention)
    csv_stem = csv_path.stem
    split_path = csv_path.parent / f"split_seed{seed}_{csv_stem}.json"
    
    if split_path.exists():
        d = json.loads(split_path.read_text(encoding="utf-8"))
        return d["train_idx"], d["val_idx"]

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = int(val_ratio * n)
    val_idx = idx[:val_n].tolist()
    train_idx = idx[val_n:].tolist()

    split_path.write_text(json.dumps({"seed": seed, "val_ratio": val_ratio,
                                     "train_idx": train_idx, "val_idx": val_idx}, indent=2),
                          encoding="utf-8")
    return train_idx, val_idx


@torch.no_grad()
def eval_acc(model, loader, device, camera_off=False):
    model.eval()
    correct, total = 0, 0
    for b in loader:
        img = b["img"].to(device)
        if camera_off:
            img = torch.zeros_like(img)
        phys = b["phys"].to(device)
        y = b["y"].to(device)
        out = model(img, phys)
        pred = out.logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(CSV_PATH))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vision_backbone", type=str, default="mobilenet_v3_small")
    parser.add_argument("--fusion", type=str, default="concat")
    parser.add_argument("--camera_off", action="store_true", help="Zero out vision input to test physiology-only")
    parser.add_argument("--suffix", type=str, default="baseline")
    args = parser.parse_args()

    seed = args.seed
    vision_backbone = args.vision_backbone
    fusion = args.fusion
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    csv_path = Path(args.csv)
    num_workers = 0  # Windows-safe; increase on Linux if stable
    # ---------------------------------

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ds = MultimodalCSVDatasetWithCF(str(csv_path))
    train_idx, val_idx = make_or_load_split(csv_path, len(ds), seed=seed)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_samples)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, collate_fn=collate_samples)

    phys_dim = ds[0].phys.numel()
    model = MultimodalThreatModel(
        phys_dim=phys_dim,
        vision_backbone=vision_backbone,
        fusion=fusion,
        num_classes=2,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    best = -1.0
    best_ckpt = OUT_CKPT / f"{args.suffix}_{vision_backbone}_{fusion}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for b in pbar:
            img = b["img"].to(device)
            if args.camera_off:
                img = torch.zeros_like(img)
            phys = b["phys"].to(device)
            y = b["y"].to(device)

            out = model(img, phys)
            loss = ce(out.logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        acc = eval_acc(model, val_loader, device, camera_off=args.camera_off)
        print(f"Epoch {epoch}: val_acc={acc:.4f}")

        if acc > best:
            best = acc
            torch.save(model.state_dict(), best_ckpt)
            print("Saved:", best_ckpt)

    report = {
        "design": args.suffix,
        "camera_off": args.camera_off,
        "seed": seed,
        "vision_backbone": vision_backbone,
        "fusion": fusion,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "best_val_acc": best,
        "params_trainable": count_trainable_params(model),
        "checkpoint": str(best_ckpt),
        "csv_path": str(CSV_PATH),
    }
    report_file = OUT_REP / f"train_{args.suffix}_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved report:", report_file)


if __name__ == "__main__":
    main()
