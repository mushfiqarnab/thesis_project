import os
import torch
import torch_pruning as tp
import logging
from src.models.pacd_net import GWPACDNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StructuralCompressorV4")

def compress_and_export(checkpoint_path="gw_cd_production_best.pth", export_path="outputs/pacd_net_edge.onnx", prune_ratio=0.35):
    logger.info(f"Initiating V4 Dimension-Shielded Structural Pruning (Ratio: {prune_ratio})")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"FATAL: Checkpoint {checkpoint_path} not found.")

    device = torch.device("cpu")
    
    # 1. Reconstruct Base Model
    model = GWPACDNet(d=64, k=4).to(device)
    
    # Safely load wrapper dictionary and map to base backbone
    state_dict = torch.load(checkpoint_path, map_location=device)
    base_state_dict = {k.replace('base_model.', ''): v for k, v in state_dict.items() if k.startswith('base_model.')}
    
    model.load_state_dict(base_state_dict, strict=False)
    model.eval()

    # 2. Configure Dummy Inputs for DepGraph Tracing
    dummy_visual = torch.randn(1, 3, 224, 224)
    dummy_physio = torch.randn(1, 4)
    dummy_scar = torch.tensor([0], dtype=torch.long)
    example_inputs = {"img": dummy_visual, "phys": dummy_physio, "scar_labels": dummy_scar}

    # Record baseline Ops and Parameters using v1.6.0+ standard
    base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    logger.info(f"Baseline Architecture -> Ops (MACs): {base_ops / 1e6:.2f}M, Params: {base_params / 1e6:.2f}M")

    # 3. V4 Dimension-Based Shielding: Protect Linear heads and non-standard tensor shapes, 
    # while allowing Conv2d + BatchNorm2d backbone pairs to prune fluidly.
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            ignored_layers.append(m)
        elif hasattr(m, "weight") and m.weight is not None and m.weight.dim() not in [2, 4]:
            ignored_layers.append(m)

    logger.info(f"V4 Shield Active: Shielded {len(ignored_layers)} custom/linear modules from DepGraph traversal.")

    # 4. Initialize Pruner with 8-channel hardware memory alignment
    imp = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=imp,
        pruning_ratio=prune_ratio,
        ignored_layers=ignored_layers,
        round_to=8 
    )

    # Execute physical tensor shrinkage safely
    pruner.step()
    
    pruned_ops, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    logger.info(f"Pruned Architecture   -> Ops (MACs): {pruned_ops / 1e6:.2f}M, Params: {pruned_params / 1e6:.2f}M")
    logger.info(f"Real Hardware Compression: {100.0 * (1.0 - pruned_ops/base_ops):.2f}% FLOPs reduction.")

    # 5. Export to ONNX with Stable Opset (17) utilizing onnxscript
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        (dummy_visual, dummy_physio, dummy_scar),
        export_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['visual', 'physio', 'scar_label'],
        output_names=['logits'],
        dynamic_axes={
            'visual': {0: 'batch_size'}, 
            'physio': {0: 'batch_size'},
            'scar_label': {0: 'batch_size'}
        }
    )
    
    logger.info(f"SUCCESS: Hardware-ready ONNX graph exported to {export_path}")
    logger.info(f"Payload size: {os.path.getsize(export_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    compress_and_export()
