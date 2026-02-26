from __future__ import annotations

from pathlib import Path
import argparse
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from models import MultimodalThreatModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (PROJECT_ROOT / pp)


def load_state_dict_safely(ckpt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Loads a checkpoint robustly across PyTorch versions and checkpoint formats:
      - prefers weights_only=True when supported
      - unwraps {"state_dict": ...} checkpoints
      - strips 'module.' prefix from DataParallel
    Returns a plain state_dict.
    """
    try:
        obj = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(str(ckpt_path), map_location=device)

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]

    if not isinstance(obj, dict):
        raise ValueError(f"Checkpoint at {ckpt_path} did not resolve to a state_dict dict.")

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        # sometimes people save under "model." prefix
        if k.startswith("model."):
            k = k[len("model.") :]
        cleaned[k] = v
    return cleaned


def infer_phys_dim_from_state_dict(state: Dict[str, torch.Tensor]) -> int:
    """
    Infers phys_dim from a checkpoint by looking for a physiology Linear layer weight.
    Works even if your CSV uses 2-feature phys or 4-feature WESAD phys.
    """
    candidates: List[Tuple[int, str]] = []

    for k, v in state.items():
        if not torch.is_tensor(v) or v.ndim != 2:
            continue
        k_low = k.lower()
        # Prefer keys that clearly belong to physiology subnetwork
        if "phys" in k_low and k_low.endswith(".weight"):
            in_dim = int(v.shape[1])
            if 1 <= in_dim <= 64:  # physiology input is small
                candidates.append((in_dim, k))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][0]

    # Fallback: pick the smallest "reasonable" Linear input dim (avoid mobilenet/vit large dims)
    for k, v in state.items():
        if not torch.is_tensor(v) or v.ndim != 2:
            continue
        in_dim = int(v.shape[1])
        out_dim = int(v.shape[0])
        if 1 <= in_dim <= 64 and 8 <= out_dim <= 512:
            candidates.append((in_dim, k))

    if not candidates:
        raise ValueError("Could not infer phys_dim from checkpoint. No suitable Linear weights found.")

    candidates.sort(key=lambda x: x[0])
    return candidates[0][0]


def should_prune_module(name: str, module: nn.Module, prune_vision: bool) -> bool:
    """
    By default prune only the non-vision linear layers (phys + fusion/classifier).
    This is safer than pruning the backbone (mobilenet).
    """
    if not isinstance(module, nn.Linear):
        return False

    n = name.lower()

    # If user explicitly wants vision pruning too, allow everything
    if prune_vision:
        return True

    # Skip common vision/backbone namespaces
    if any(tok in n for tok in ["vision", "backbone", "mobilenet", "vit"]):
        return False

    # Keep prunes focused where we intended: phys + fusion/classifier head
    if any(tok in n for tok in ["phys", "classifier", "head", "fusion", "cgf", "gate"]):
        return True

    # If naming is different in your codebase, still prune small heads:
    # heuristic: prune only if layer is not huge
    # (this avoids accidentally pruning backbone linears)
    try:
        in_f = int(module.in_features)
        out_f = int(module.out_features)
        if in_f <= 1024 and out_f <= 1024:
            return True
    except Exception:
        pass

    return False


def main():
    ap = argparse.ArgumentParser("Prune non-vision Linear layers of a trained checkpoint (thesis pipeline).")
    ap.add_argument("--ckpt", type=str, required=True, help="Input checkpoint path (.pt state_dict)")
    ap.add_argument("--out", type=str, required=True, help="Output pruned checkpoint path (.pt state_dict)")
    ap.add_argument("--fusion", type=str, required=True, choices=["concat", "cgf"])
    ap.add_argument("--backbone", type=str, required=True, choices=["mobilenet_v3_small"])
    ap.add_argument("--amount", type=float, default=0.3, help="Fraction to prune in selected Linear layers")
    ap.add_argument("--prune_vision", action="store_true", help="If set, also prune vision backbone Linear layers")
    args = ap.parse_args()

    ckpt_path = _resolve_path(args.ckpt)
    out_path = _resolve_path(args.out)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if ckpt_path.resolve() == out_path.resolve():
        raise ValueError("Refusing to overwrite input checkpoint. Use a different --out path.")

    device = torch.device("cpu")

    # Load checkpoint safely + infer phys_dim
    state = load_state_dict_safely(ckpt_path, device)
    phys_dim = infer_phys_dim_from_state_dict(state)

    model = MultimodalThreatModel(
        phys_dim=phys_dim,
        vision_backbone=args.backbone,
        fusion=args.fusion,
        num_classes=2,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    # Prune selected Linear layers
    pruned_layers = 0
    for name, m in model.named_modules():
        if should_prune_module(name, m, prune_vision=args.prune_vision):
            prune.l1_unstructured(m, name="weight", amount=float(args.amount))
            pruned_layers += 1

    # Make pruning permanent
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and hasattr(m, "weight_orig"):
            prune.remove(m, "weight")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))

    print(f"[Prune] ckpt_in={ckpt_path}")
    print(f"[Prune] inferred_phys_dim={phys_dim} fusion={args.fusion} backbone={args.backbone}")
    print(f"[Prune] amount={args.amount} prune_vision={args.prune_vision} pruned_linear_layers={pruned_layers}")
    print(f"[Prune] Saved pruned state_dict to: {out_path}")


if __name__ == "__main__":
    main()
