import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path('src').resolve()))
from models.stiefel_causal_layer import StiefelCausalLinear

def simulate_8bit_quantization(tensor):
    """Simulates dynamic 8-bit quantization rounding behavior."""
    if tensor.abs().max() == 0:
        return tensor
    scale = tensor.abs().max() / 127.0
    return torch.round(tensor / scale) * scale

def stress_test_stiefel_layer():
    print("==================================================")
    print(" STIEFEL MANIFOLD CAUSAL LAYER: EDGE STRESS TEST  ")
    print("==================================================")
    
    in_dim, out_dim = 128, 64
    layer = StiefelCausalLinear(in_dim, out_dim, ns_iterations=3)
    
    # 1. Baseline FP32 Check
    W_fp32 = layer.get_stiefel_weight()
    I = torch.eye(out_dim)
    dev_fp32 = torch.linalg.matrix_norm(torch.matmul(W_fp32, W_fp32.t()) - I, ord='fro').item()
    print(f"[*] Baseline (FP32) Orthogonality Deviation: {dev_fp32:.6e} (Stable)")
    
    # 2. Mimicking QDyn (8-bit) compounding accumulation in Newton-Schulz
    print("\n[*] Injecting QDyn (8-bit) accumulation shifts into Newton-Schulz iterations...")
    
    # We will override the iteration locally to simulate edge hardware precision constraints
    W_raw = layer.weight_raw.clone().detach()
    
    # Pre-condition scaling
    norm = torch.linalg.matrix_norm(W_raw, ord='fro') + 1e-8
    scale = norm / (out_dim ** 0.5)
    Q = W_raw / scale
    
    I_gpu = torch.eye(out_dim)
    
    for i in range(1, 11): # Push iterations to see if it explodes under noise
        # Standard step
        Q_Q_T = torch.matmul(Q, Q.t())
        
        # INJECT EDGE HARDWARE QUANTIZATION NOISE TO MATMUL RESULT
        Q_Q_T = simulate_8bit_quantization(Q_Q_T) 
        
        step = 1.5 * I_gpu - 0.5 * Q_Q_T
        Q = torch.matmul(step, Q)
        
        # INJECT EDGE HARDWARE QUANTIZATION NOISE TO STATE
        Q = simulate_8bit_quantization(Q)
        
        dev = torch.linalg.matrix_norm(torch.matmul(Q, Q.t()) - I_gpu, ord='fro').item()
        nan_flag = "DETECTED" if torch.isnan(Q).any() else "Safe"
        print(f"    -> Iteration {i}: Deviation = {dev:.4f} | NaN Status: {nan_flag}")
        
        if torch.isnan(Q).any() or dev > 1e4:
            print("\n[!] CATASTROPHIC FAILURE: Newton-Schulz iteration mathematically exploded under 8-bit precision.")
            print("    Gradients are obliterated. Edge devices will output NaN.")
            break
            
    print("==================================================")

if __name__ == "__main__":
    stress_test_stiefel_layer()
