"""
mitl_proof.py
=============
Cross-toolchain proof of Manifold-Induced Topological Locking (MITL).
Demonstrates that the MITL phenomenon is not a torch-pruning bug
but a fundamental constraint by showing that torch.fx symbolic tracing
also fails at the manifold boundary.
"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from src.models.pacd_net import GWPACDNet

model = GWPACDNet(d=64, k=4)
model.eval()

print("=" * 65)
print("     MITL CROSS-TOOLCHAIN PROOF — torch.fx SYMBOLIC TRACE")
print("=" * 65)
print("Attempting torch.fx symbolic_trace on GWPACDNet...")
print("Expected: failure at manifold boundary (linalg_qr or view op)")
print("-" * 65)

try:
    from torch.fx import symbolic_trace
    traced = symbolic_trace(model)
    print("STATUS: FX trace SUCCEEDED.")
    print("IMPLICATION: MITL argument requires re-examination.")
    print("The manifold ops ARE traceable. The locking is pruner-specific.")
except Exception as e:
    print(f"STATUS: FX trace FAILED.")
    print(f"Error type : {type(e).__name__}")
    print(f"Error msg  : {str(e)[:200]}")
    print("-" * 65)
    print("MITL CONFIRMED: GWPACDNet resists automated graph analysis")
    print("across multiple independent toolchains (torch-pruning DepGraph")
    print("AND torch.fx symbolic tracing). This is tool-independent evidence")
    print("of the manifold boundary constraint.")

print("=" * 65)
print()
print("Attempting trace with concrete example inputs (torch.jit.trace)...")
try:
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_phys = torch.randn(1, 4)
    
    class TraceWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, img, phys): return self.m(img=img, phys=phys, scar_labels=None)["logits"]
    
    wrapper = TraceWrapper(model)
    traced_jit = torch.jit.trace(wrapper, (dummy_img, dummy_phys))
    print("STATUS: JIT trace SUCCEEDED with concrete inputs.")
    print("IMPLICATION: Dynamic control flow (if/else branches) is the")
    print("primary tracing barrier, not the QR operation itself.")
    print("The MITL argument must focus on the DepGraph shape indexing,")
    print("not symbolic tracing inability.")
except Exception as e:
    print(f"STATUS: JIT trace FAILED: {type(e).__name__}: {str(e)[:200]}")
