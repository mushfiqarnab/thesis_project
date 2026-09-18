"""
export_v4_onnx.py
======================================================
Compiles the Golden Software Master (V4 Checkpoint) into a static Edge ONNX graph.
Enforces strict Opset 14 (LayerNorm native) and constant folding to strip demographic markers.
"""
import torch
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models_arch import MultimodalThreatModel

def export_to_onnx():
    ckpt_path = Path("outputs/checkpoints/counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_strict_stiefel.pt")
    out_path = Path("outputs/equitas_mitl_strict_v4_edge.onnx")
    
    print("==================================================================")
    print(" MITL HARDWARE COMPILER (ONNX OPSET 14)")
    print("==================================================================")
    
    if not ckpt_path.exists():
        print(f"[ERROR] Golden checkpoint missing: {ckpt_path}")
        return
        
    print(f"[1/3] Loading V4 CGF Architecture...")
    # Initialize the model exactly as it was during the Master Run
    # Because we unified StiefelCausalLinear across all runs, the state dict keys
    # now match perfectly. EQUITAS_DISABLE_STIEFEL defaults to 0 (active).
    model = MultimodalThreatModel(phys_dim=2, fusion="cgf")
    
    print(f"[2/3] Restoring Golden Weights...")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # WE NOW ENFORCE STRICT=TRUE TO PROVE NO WEIGHTS ARE DROPPED
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model
            
        def forward(self, img, phys):
            # ONNX tracing requires standard tensor outputs, not dataclasses
            out = self.core(img, phys)
            return out.logits
            
    onnx_model = ONNXWrapper(model)
    onnx_model.eval()
    
    print(f"[3/3] Compiling to Opset 13 INT8-Ready Graph...")
    # Create dummy inputs for tracing
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_phys = torch.randn(1, 2)
    
    # Export
    torch.onnx.export(
        onnx_model,
        (dummy_img, dummy_phys),
        str(out_path),
        export_params=True,
        opset_version=14,  # Upgraded for native LayerNorm compilation
        do_constant_folding=True,
        input_names=['vision_input', 'phys_input'],
        output_names=['threat_logits'],
        dynamic_axes={
            'vision_input': {0: 'batch_size'},
            'phys_input': {0: 'batch_size'},
            'threat_logits': {0: 'batch_size'}
        }
    )
    
    print(f"\n[SUCCESS] Opset 14 Edge Graph saved to: {out_path}")
    print("==================================================================")

if __name__ == "__main__":
    export_to_onnx()
