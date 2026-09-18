import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cgrn_arch import CGRN

def verify_training_alignment():
    print("[AUDIT] Instantiating Phase-4 CGRN...")
    model = CGRN(v_dim=576, p_dim=64, num_classes=2)
    
    print("[AUDIT] Simulating training loop forward pass...")
    dummy_v = torch.randn(8, 576)
    dummy_p = torch.randn(8, 64)
    
    # Simulate what train_cgf_fair.py expects
    print("[AUDIT] train_cgrn_phase4.py expects: out = model(dummy_v, dummy_p)")
    try:
        out = model(dummy_v, dummy_p) 
        
        print(f"[AUDIT] Model output type: {type(out)}")
        
        # New expected extraction
        logits = out['logits']
        phi_v = out['phi_v']
        trust_score = out['trust_score']
        
        print("[SUCCESS] Pipeline structural alignment is perfect. No missing keys.")
        print(f"[MATHEMATICS] Logits shape: {logits.shape}, HSIC Kernel shape: {phi_v.shape}")
        
    except Exception as e:
        print(f"\n[FATAL PIPELINE FRACTURE DETECTED]")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    verify_training_alignment()
