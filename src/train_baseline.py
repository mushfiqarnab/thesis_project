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
from models import MultimodalThreatModel, count_trainable_params


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "csv" / "multimodal.csv"

OUT_CKPT = PROJECT_ROOT / "outputs" / "checkpoints"
OUT_REP = PROJECT_ROOT / "outputs" / "reports"
OUT_CKPT.mkdir(parents=True, exist_ok=True)
OUT_REP.mkdir(parents=True, exist_ok=True)

SPLIT_PATH = PROJECT_ROOT / "data" / "csv" / "split_seed42.json"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_or_load_split(n: int, seed: int = 42, val_ratio: float = 0.2):
    if SPLIT_PATH.exists():
        d = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        return d["train_idx"], d["val_idx"]

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = int(val_ratio * n)
    val_idx = idx[:val_n].tolist()
    train_idx = idx[val_n:].tolist()

    SPLIT_PATH.write_text(json.dumps({"seed": seed, "val_ratio": val_ratio,
                                     "train_idx": train_idx, "val_idx": val_idx}, indent=2),
                          encoding="utf-8")
    return train_idx, val_idx


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for b in loader:
        img = b["img"].to(device)
        phys = b["phys"].to(device)
        y = b["y"].to(device)
        out = model(img, phys)
        pred = out.logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    # ---- config (edit only here) ----
    seed = 42
    vision_backbone = "mobilenet_v3_small"   # change to "vit_b_16" for reference baseline
    fusion = "concat"
    epochs = 10
    batch_size = 32
    lr = 2e-4
    num_workers = 0  # Windows-safe; increase on Linux if stable
    # ---------------------------------

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ds = MultimodalCSVDatasetWithCF(str(CSV_PATH))
    train_idx, val_idx = make_or_load_split(len(ds), seed=seed)

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
    best_ckpt = OUT_CKPT / f"baseline_{vision_backbone}_{fusion}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for b in pbar:
            img = b["img"].to(device)
            phys = b["phys"].to(device)
            y = b["y"].to(device)

            out = model(img, phys)
            loss = ce(out.logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        acc = eval_acc(model, val_loader, device)
        print(f"Epoch {epoch}: val_acc={acc:.4f}")

        if acc > best:
            best = acc
            torch.save(model.state_dict(), best_ckpt)
            print("Saved:", best_ckpt)

    report = {
        "design": "A",
        "seed": seed,
        "vision_backbone": vision_backbone,
        "fusion": fusion,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "best_val_acc": best,
        "params_trainable": count_trainable_params(model),
        "checkpoint": str(best_ckpt),
        "split_path": str(SPLIT_PATH),
    }
    (OUT_REP / "train_baseline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved report:", OUT_REP / "train_baseline_report.json")


if __name__ == "__main__":
    main()
