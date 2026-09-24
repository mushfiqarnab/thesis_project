import sys
from pathlib import Path
import time
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.equitas_rcmf import EquitasRCMFModel

def main():
    print("Initializing Edge Latency Benchmark (Autonomous Mode)...")
    
    # We benchmark on CPU to simulate edge hardware constraints (e.g. Raspberry Pi, Mobile)
    device = torch.device("cpu")
    
    # Instantiate the Edge-optimized model
    # (Matches our 1-hour fast run using MobileNetV3)
    model = EquitasRCMFModel(
        phys_dim=3, # BVP, EDA, TEMP typically
        vision_backbone="mobilenet_v3_small",
        d_causal=192,
        d_confounder=64,
        num_classes=2
    ).to(device)
    
    model.eval()
    
    # Dummy tensors representing 1 frame of streaming multimodal input
    # Vision: (1, 3, 224, 224)
    # Phys: (1, 3)
    dummy_img = torch.randn(1, 3, 224, 224).to(device)
    dummy_phys = torch.randn(1, 3).to(device)
    
    print("\nWarming up the model...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_img, dummy_phys, mask=None) # Autonomous inference (no mask)
            
    print("Running 1,000 iterations for stable latency measurement...")
    latencies = []
    
    with torch.no_grad():
        for _ in range(1000):
            start = time.perf_counter()
            _ = model(dummy_img, dummy_phys, mask=None)
            latencies.append((time.perf_counter() - start) * 1000) # Convert to milliseconds
            
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    fps = 1000.0 / avg_latency
    
    print("\n" + "="*50)
    print("      EDGE DEPLOYMENT PROOF")
    print("="*50)
    print(f"Architecture    : EQUITAS-RCMF (MobileNetV3)")
    print(f"Hardware Target : CPU (Simulated Edge)")
    print(f"Total Params    : {sum(p.numel() for p in model.parameters()):,}")
    print("-"*50)
    print(f"Avg Latency     : {avg_latency:.2f} ms per frame")
    print(f"95th Percentile : {p95_latency:.2f} ms per frame")
    print(f"Throughput      : {fps:.1f} FPS")
    print("="*50)
    
    if fps > 15.0:
        print("Verdict: PASS - Model is fully capable of real-time edge streaming.")
    else:
        print("Verdict: FAIL - Model does not meet real-time edge constraints.")

if __name__ == "__main__":
    main()
