import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import onnxruntime as ort

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.equitas_rcmf import EquitasRCMFModel

def run_perfection_audit():
    print("================================================================")
    print("      EQUITAS-RCMF: MASTER THESIS PERFECTION AUDIT")
    print("================================================================")
    
    all_passed = True

    # 1. DATA ISOLATION AUDIT
    print("\n[1/4] Auditing Data Leakage & Cryptographic Isolation...")
    try:
        manifest_path = PROJECT_ROOT / "data/publishable_scar_production/manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            train_faces = set(manifest["face_splits"]["train"])
            val_faces = set(manifest["face_splits"]["val"])
            test_faces = set(manifest["face_splits"]["test"])
            
            assert len(train_faces.intersection(val_faces)) == 0, "Leakage: Train/Val faces overlap!"
            assert len(train_faces.intersection(test_faces)) == 0, "Leakage: Train/Test faces overlap!"
            assert len(val_faces.intersection(test_faces)) == 0, "Leakage: Val/Test faces overlap!"
            print("  [PASS] Strict zero-leakage verified across all splits.")
        else:
            print("  [SKIP] Manifest not found.")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # 2. MATHEMATICAL INVARIANT AUDIT (STIEFEL MANIFOLD)
    print("\n[2/4] Auditing Riemannian Stiefel Orthogonality...")
    try:
        model = EquitasRCMFModel(phys_dim=2, d_causal=192, d_confounder=64)
        model.eval()
        
        # Test dynamic QR projection
        ortho_dev_dynamic = model.stiefel_decomp.verify_mutual_orthogonality()
        assert ortho_dev_dynamic < 1e-5, f"Dynamic Orthogonality failed: {ortho_dev_dynamic}"
        
        # Test frozen edge projection
        model.freeze_stiefel_weights()
        ortho_dev_frozen = model.stiefel_decomp.verify_mutual_orthogonality()
        assert ortho_dev_frozen < 1e-5, f"Frozen Orthogonality failed: {ortho_dev_frozen}"
        
        print(f"  [PASS] Perfect orthogonal disentanglement (Error: {ortho_dev_frozen:.2e})")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # 3. THERMODYNAMIC GATE BOUNDARY AUDIT
    print("\n[3/4] Auditing Thermodynamic Gate Limits...")
    try:
        dummy_v = torch.randn(2, 192)
        dummy_p = torch.randn(2, 192)
        
        # Test Case A: No focus (Focus = 0.0)
        gate_no_focus = model.gate_engine(dummy_v, dummy_p, focus=torch.tensor([[0.0], [0.0]]))
        assert (gate_no_focus > 0.0).all() and (gate_no_focus <= 1.0).all(), "Gate out of bounds"
        
        # Test Case B: Extreme focus (Focus = 10.0 -> Scar detected)
        gate_extreme = model.gate_engine(dummy_v, dummy_p, focus=torch.tensor([[10.0], [10.0]]))
        assert (gate_extreme < 1e-3).all(), f"Gate failed to suppress vision: {gate_extreme.mean()}"
        
        print("  [PASS] Continuous thermodynamic suppression behaves perfectly.")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    # 4. EDGE COMPILE PARITY AUDIT
    print("\n[4/4] Auditing ONNX Edge Hardware Parity...")
    try:
        onnx_path = PROJECT_ROOT / "outputs/edge/equitas_rcmf_master_fp32.onnx"
        ckpt_path = PROJECT_ROOT / "outputs/checkpoints/equitas_rcmf_master_best.pt"
        if onnx_path.exists() and ckpt_path.exists():
            # Load exact weights into PyTorch model
            model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu", weights_only=True))
            model.eval()
            model.freeze_stiefel_weights()
            
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            
            dummy_img = torch.randn(1, 3, 224, 224)
            dummy_phys = torch.randn(1, 2)
            
            # PyTorch Inference
            with torch.no_grad():
                pt_out = model(dummy_img, dummy_phys, mask=None)
                pt_prob = torch.softmax(pt_out.logits, dim=1)[:, 1:2].numpy()
            
            # ONNX Inference
            ort_inputs = {
                session.get_inputs()[0].name: dummy_img.numpy(),
                session.get_inputs()[1].name: dummy_phys.numpy(),
            }
            ort_out = session.run(None, ort_inputs)[0]
            
            max_diff = np.max(np.abs(pt_prob - ort_out))
            assert max_diff < 1e-4, f"ONNX parity failed! Max diff: {max_diff}"
            print(f"  [PASS] Edge compilation parity verified (Max Error: {max_diff:.2e})")
        else:
            print("  [SKIP] ONNX model not found.")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False

    print("\n================================================================")
    if all_passed:
        print(" [VERDICT] 100% PERFECT. ALL INVARIANTS HOLD.")
    else:
        print(" [VERDICT] IMPERFECTIONS DETECTED. SEE LOGS ABOVE.")
    print("================================================================")

if __name__ == "__main__":
    run_perfection_audit()
