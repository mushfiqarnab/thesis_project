from __future__ import annotations

from pathlib import Path
import argparse
import json
import time
from types import SimpleNamespace
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as tvm

from dataset_fair import MultimodalCSVDatasetWithCF

from models import MultimodalThreatModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Edge metrics benchmark: size, CPU latency, throughput, RAM.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--fusion", type=str, required=True, choices=["concat", "cgf"])
    p.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "vit_b_16"])
    p.add_argument("--csv", type=str, default=str(PROJECT_ROOT / "data" / "csv" / "multimodal.csv"))
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quantize_dynamic", action="store_true")
    p.add_argument("--out", type=str, default="")
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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def safe_rss_mb() -> float | None:
    try:
        import psutil  # optional
        proc = psutil.Process()
        return float(proc.memory_info().rss) / (1024 ** 2)
    except Exception:
        return None


@torch.no_grad()
def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    torch.set_num_threads(int(args.threads))
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # Build one real sample input (CPU edge-style)
    ds = MultimodalCSVDatasetWithCF(args.csv)
    s = ds[0]
    img = s.img.unsqueeze(0).cpu()
    phys = s.phys.unsqueeze(0).cpu()
    mask = s.mask.unsqueeze(0).cpu()

    state = load_raw_state_dict(ckpt_path)
    legacy = is_legacy_checkpoint(state)
    phys_dim = int(phys.shape[1])

    if legacy:
        model = LegacyViTConcatModel(phys_in_dim=phys_dim, phys_emb=32, num_classes=2).cpu()
        model.load_state_dict(state, strict=True)
        model_family = "legacy_vit_concat_phys32"
    else:
        model = MultimodalThreatModel(phys_dim=phys_dim, vision_backbone=args.backbone, fusion=args.fusion, num_classes=2).cpu()
        model.load_state_dict(state, strict=True)
        model_family = "current_multimodal"

    model.eval()

    if args.quantize_dynamic:
        model = maybe_quantize_dynamic(model)
        model.eval()

    # Warmup
    for _ in range(int(args.warmup)):
        _ = model(img, phys, mask=mask).logits

    rss_before = safe_rss_mb()

    # Benchmark
    times = []
    for _ in range(int(args.iters)):
        t0 = time.perf_counter()
        _ = model(img, phys, mask=mask).logits
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    rss_after = safe_rss_mb()

    ms = np.array(times, dtype=np.float64)
    latency_mean = float(ms.mean())
    latency_p50 = float(np.percentile(ms, 50))
    latency_p95 = float(np.percentile(ms, 95))
    throughput_fps = float(1000.0 / latency_mean) if latency_mean > 0 else 0.0

    size_mb = float(ckpt_path.stat().st_size) / (1024 ** 2)

    report = {
        "ckpt": str(ckpt_path),
        "model_family": model_family,
        "quantize_dynamic": bool(args.quantize_dynamic),
        "ckpt_size_mb": size_mb,
        "params": int(count_params(model)),
        "cpu_threads": int(args.threads),
        "latency_ms_mean": latency_mean,
        "latency_ms_p50": latency_p50,
        "latency_ms_p95": latency_p95,
        "throughput_fps_est": throughput_fps,
        "rss_mb_before": rss_before,
        "rss_mb_after": rss_after,
        "rss_mb_delta": (None if (rss_before is None or rss_after is None) else float(rss_after - rss_before)),
    }

    out_path = Path(args.out) if args.out else (PROJECT_ROOT / "outputs" / "results" / f"edge_{Path(args.ckpt).stem}_{'qdyn' if args.quantize_dynamic else 'fp32'}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
