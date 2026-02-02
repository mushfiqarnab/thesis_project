from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from models import MultimodalThreatModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "csv" / "multimodal.csv"
SPLIT_PATH = PROJECT_ROOT / "data" / "csv" / "split_seed42.json"

OUT_DIR = PROJECT_ROOT / "outputs" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dp_gap_signed(yhat: np.ndarray, scar: np.ndarray) -> float:
    p1 = yhat[scar == 1].mean() if (scar == 1).any() else 0.0
    p0 = yhat[scar == 0].mean() if (scar == 0).any() else 0.0
    return float(p1 - p0)


def eo_gaps(yhat: np.ndarray, y: np.ndarray, scar: np.ndarray) -> dict:
    def rates(g):
        idx = (scar == g)
        if not idx.any():
            return 0.0, 0.0
        yy = y[idx]
        yh = yhat[idx]
        tp = ((yh == 1) & (yy == 1)).sum()
        fn = ((yh == 0) & (yy == 1)).sum()
        fp = ((yh == 1) & (yy == 0)).sum()
        tn = ((yh == 0) & (yy == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        return float(tpr), float(fpr)

    tpr1, fpr1 = rates(1)
    tpr0, fpr0 = rates(0)
    return {
        "tpr1": tpr1, "tpr0": tpr0,
        "fpr1": fpr1, "fpr0": fpr0,
        "tpr_gap": float(tpr1 - tpr0),
        "fpr_gap": float(fpr1 - fpr0),
        "eo_max_gap": float(max(abs(tpr1 - tpr0), abs(fpr1 - fpr0))),
    }


@torch.no_grad()
def main():
    # ---- config (edit only here to match your checkpoint) ----
    vision_backbone = "mobilenet_v3_small"
    fusion = "cgf"  # "concat" for baseline, "cgf" for fair model
    ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"counterfactual_{fusion}_js_{vision_backbone}_best.pt"
    out_json = OUT_DIR / "fairness_report.json"
    batch_size = 64
    num_workers = 0
    # ---------------------------------------------------------

    if not SPLIT_PATH.exists():
        raise FileNotFoundError("split_seed42.json not found. Train once to generate it.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MultimodalCSVDatasetWithCF(str(CSV_PATH))

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    val_ds = Subset(ds, split["val_idx"])

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, collate_fn=collate_samples)

    phys_dim = ds[0].phys.numel()
    model = MultimodalThreatModel(
        phys_dim=phys_dim,
        vision_backbone=vision_backbone,
        fusion=fusion,
        num_classes=2,
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    probs_all, y_all, scar_all = [], [], []
    cf_gap_list = []
    gate_list, focus_list = [], []

    for b in loader:
        img = b["img"].to(device)
        img_cf = b["img_cf"].to(device)
        phys = b["phys"].to(device)
        y = b["y"].cpu().numpy()
        scar = b["scar"].cpu().numpy()
        has_cf = b["has_cf"].cpu().numpy().astype(bool)
        mask = b["mask"].to(device)

        out = model(img, phys, mask=mask)
        p = F.softmax(out.logits, dim=1)[:, 1].detach().cpu().numpy()  # P(threat=1)

        if has_cf.any():
            out_cf = model(img_cf, phys, mask=mask)
            p_cf = F.softmax(out_cf.logits, dim=1)[:, 1].detach().cpu().numpy()
            cf_gap_list.extend(np.abs(p[has_cf] - p_cf[has_cf]).tolist())

        if out.gate is not None:
            gate_list.extend(out.gate.detach().cpu().numpy().reshape(-1).tolist())
        if out.focus is not None:
            focus_list.extend(out.focus.detach().cpu().numpy().reshape(-1).tolist())

        probs_all.append(p)
        y_all.append(y)
        scar_all.append(scar)

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)
    scar_all = np.concatenate(scar_all)

    yhat = (probs_all >= 0.5).astype(int)

    acc = float((yhat == y_all).mean())
    dp_s = dp_gap_signed(yhat, scar_all)
    eo = eo_gaps(yhat, y_all, scar_all)
    cf_gap = float(np.mean(cf_gap_list)) if len(cf_gap_list) else 0.0

    report = {
        "checkpoint": str(ckpt_path),
        "fusion": fusion,
        "vision_backbone": vision_backbone,
        "n_val": int(len(y_all)),
        "acc": acc,
        "dp_gap_signed": dp_s,
        "dp_gap_abs": float(abs(dp_s)),
        "eo": eo,
        "cf_prob_gap_mean_abs": cf_gap,
        "gate_mean": float(np.mean(gate_list)) if gate_list else None,
        "focus_mean": float(np.mean(focus_list)) if focus_list else None,
        "cf_samples": int(len(cf_gap_list)),
    }

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
