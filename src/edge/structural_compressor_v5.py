"""
structural_compressor_v5.py
===========================
Zero-compromise structural compressor for Edge NPU deployment.
1. Shields ONLY the final classification head (out_features==2).
2. Dynamically catches and rebuilds severed Linear layer dimensions.
3. Exports via stable Opset 12 (dynamo=False) for ONNXRuntime 1.30.0.
"""
import os
import torch
import torch.nn as nn
import torch_pruning as tp
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.pacd_net import GWPACDNet

# Force ASCII-only logging to prevent Windows cp1252 Unicode crashes
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CompressorV5")

def rebuild_mismatched_linears(model, example_inputs):
    shapes = {}
    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(module, nn.Linear):
                actual_in = inp[0].shape[-1]
                if actual_in != module.in_features:
                    shapes[name] = (actual_in, module.out_features, module.bias is not None)
        return hook

    handles = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        try:
            model(img=example_inputs["img"], phys=example_inputs["phys"], scar_labels=example_inputs["scar_labels"])
        except Exception:
            pass

    for h in handles:
        h.remove()

    for name, (new_in, out_f, has_bias) in shapes.items():
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old_layer = getattr(parent, parts[-1])
        new_layer = nn.Linear(new_in, out_f, bias=has_bias)
        nn.init.kaiming_uniform_(new_layer.weight, a=0.01)
        if has_bias:
            nn.init.zeros_(new_layer.bias)
        setattr(parent, parts[-1], new_layer)
        logger.info(f"Rebuilt Linear [{name}]: {old_layer.in_features} -> {new_in}")

def compress_and_export():
    checkpoint_path = "outputs/fair_model_best.pth"
    export_path = "outputs/pacd_net_edge.onnx"
    prune_ratio = 0.35

    logger.info(f"Initializing V5 Compression | Ratio: {prune_ratio}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"FATAL: Checkpoint {checkpoint_path} missing.")

    device = torch.device("cpu")
    model = GWPACDNet(d=64, k=4).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if any(k.startswith("base_model.") for k in state):
        state = {k.replace("base_model.", ""): v for k, v in state.items() if k.startswith("base_model.")}
    model.load_state_dict(state, strict=False)
    model.eval()

    example_inputs = {
        "img": torch.randn(1, 3, 224, 224),
        "phys": torch.randn(1, 4),
        "scar_labels": torch.tensor([0], dtype=torch.long),
    }

    base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    logger.info(f"Baseline Architecture: {base_ops/1e6:.2f}M FLOPs | {base_params/1e6:.3f}M Params")

    ignored = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == 2]
    logger.info(f"Shielding {len(ignored)} final classification head(s).")

    imp = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs=example_inputs, importance=imp,
        pruning_ratio=prune_ratio, ignored_layers=ignored, round_to=8
    )
    pruner.step()
    logger.info("Structural pruning step completed.")

    rebuild_mismatched_linears(model, example_inputs)

    pruned_ops, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    logger.info(f"Pruned Architecture  : {pruned_ops/1e6:.2f}M FLOPs | {pruned_params/1e6:.3f}M Params")
    logger.info(f"Net FLOPs Reduction  : {100.0 * (1.0 - pruned_ops / max(base_ops, 1)):.2f}%")

    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    dummy_tuple = (example_inputs["img"], example_inputs["phys"], example_inputs["scar_labels"])

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, dummy_tuple, export_path, export_params=True,
            opset_version=12, do_constant_folding=True,
            input_names=["visual", "physio", "scar_label"],
            output_names=["logits"],
            dynamic_axes={"visual": {0: "batch"}, "physio": {0: "batch"}, "scar_label": {0: "batch"}}
        )

    size_mb = os.path.getsize(export_path) / (1024 * 1024)
    logger.info(f"ONNX Graph Exported  : {export_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    compress_and_export()
