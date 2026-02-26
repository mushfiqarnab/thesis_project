from __future__ import annotations

from pathlib import Path
import argparse
import json
import torch
import torch.nn as nn
import torchvision.models as tvm
from types import SimpleNamespace

from dataset_fair import MultimodalCSVDatasetWithCF
from models import MultimodalThreatModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Export a dynamic-quantized model artifact (Linear -> int8).")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--fusion", type=str, required=True, choices=["concat", "cgf"])
    p.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "vit_b_16"])
    p.add_argument("--csv", type=str, default=str(PROJECT_ROOT / "data" / "csv" / "multimodal.csv"))
    p.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "checkpoints"))
    return p.parse_args()


def load_raw_state_dict(ckpt_path: Path) -> dict:
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Invalid checkpoint format.")
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def is_legacy_checkpoint(state: dict) -> bool:
    keys = list(state.keys())
    has_vit = any(k.startswith("vit.") for k in keys)
    has_phys_old = any(k.startswith("phys_net.") for k in keys)
    has_cls_old = any(k.startswith("classifier.") for k in keys)
    return has_vit and (has_phys_old or has_cls_old)


class LegacyViTConcatModel(nn.Module):
    def __init__(self, phys_in_dim: int = 2, phys_emb: int = 32, num_classes: int = 2):
        super().__init__()
        try:
            vit = tvm.vit_b_16(weights=None)
        except TypeError:
            vit = tvm.vit_b_16(pretrained=False)
        if hasattr(vit, "heads"):
            vit.heads = nn.Identity()
        else:
            vit.head = nn.Identity()
        self.vit = vit
        self.phys_net = nn.Sequential(
            nn.Linear(phys_in_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, phys_emb), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + phys_emb, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, img: torch.Tensor, phys: torch.Tensor, mask=None):
        v = self.vit(img)
        p = self.phys_net(phys)
        logits = self.classifier(torch.cat([v, p], dim=1))
        return SimpleNamespace(logits=logits, gate=None, focus=None)


def maybe_quantize_dynamic(model: nn.Module) -> nn.Module:
    try:
        import torch.ao.quantization as tq
        return tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    except Exception:
        return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def main():
    args = parse_args()
    ckpt = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = load_raw_state_dict(ckpt)
    legacy = is_legacy_checkpoint(state)

    ds = MultimodalCSVDatasetWithCF(args.csv)
    phys_dim = int(ds[0].phys.numel())

    if legacy:
        model = LegacyViTConcatModel(phys_in_dim=phys_dim, phys_emb=32, num_classes=2).cpu()
        model.load_state_dict(state, strict=True)
        family = "legacy_vit_concat_phys32"
    else:
        model = MultimodalThreatModel(phys_dim=phys_dim, vision_backbone=args.backbone, fusion=args.fusion, num_classes=2).cpu()
        model.load_state_dict(state, strict=True)
        family = "current_multimodal"

    model.eval()
    model_q = maybe_quantize_dynamic(model).cpu().eval()

    out_pt = out_dir / f"{ckpt.stem}_qdyn_{family}.pt"
    torch.save(model_q.state_dict(), out_pt)

    report = {
        "in_ckpt": str(ckpt),
        "out_ckpt": str(out_pt),
        "legacy": legacy,
        "family": family,
        "in_size_mb": float(ckpt.stat().st_size) / (1024 ** 2),
        "out_size_mb": float(out_pt.stat().st_size) / (1024 ** 2),
        "note": "This is a quantized state_dict; load by constructing the same model then applying quantize_dynamic before load_state_dict().",
    }

    out_json = out_dir / f"{ckpt.stem}_qdyn_{family}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", out_pt)


if __name__ == "__main__":
    main()
