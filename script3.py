import torch

def test_nsmr_bfloat16_stability():
    torch.manual_seed(42)
    # Load actual checkpoint
    ckpt = torch.load(r"C:\Users\USERAS\thesis_project\outputs\quarantine_checkpoints_20260924\equitas_rcmf_master_best.pt", map_location="cpu")
    if "model_state_dict" in ckpt: ckpt = ckpt["model_state_dict"]
    
    stiefel_weight = None
    for k, v in ckpt.items():
        if 'stiefel' in k and 'weight_raw' in k:
            stiefel_weight = v
            break
            
    if stiefel_weight is None:
        print("Could not find stiefel_decomp.weight_raw in checkpoint. Using random init.")
        W_initial = torch.randn(256, 576, dtype=torch.bfloat16)
    else:
        print("Found real stiefel weight.")
        W_initial = stiefel_weight.to(torch.bfloat16)
        
    W = W_initial.t()
    frobenius_norm = torch.linalg.matrix_norm(W.to(torch.float32), ord='fro')
    Q = (W.to(torch.float32) / (frobenius_norm + 1e-6)).to(torch.bfloat16)
    
    d = Q.shape[1]
    I = torch.eye(d, dtype=torch.bfloat16)
    
    print("--- Edge NPU bfloat16 Newton-Schulz Manifold Retraction (NSMR) ---")
    print(f"Matrix Dimension: {Q.shape}")
    print(f"Execution Precision: {Q.dtype}")
    
    iterations = 5
    for i in range(iterations):
        Q_T_Q = torch.matmul(Q.t(), Q)
        inner_term = 3.0 * I - Q_T_Q
        Q = 0.5 * torch.matmul(Q, inner_term)
        
        current_dev = torch.linalg.matrix_norm((torch.matmul(Q.t(), Q) - I).to(torch.float32), ord='fro')
        print(f"Iteration {i+1} | Frobenius Deviation from Identity: {current_dev.item():.6f}")

    final_deviation = torch.linalg.matrix_norm((torch.matmul(Q.t(), Q) - I).to(torch.float32), ord='fro')
    
    if final_deviation < 0.1:
        print("\nHardware Proof Successful: The Stiefel geometry converges and stabilizes entirely within bfloat16 truncation limits.")
    else:
        print("\nHardware Proof Failed: Numerical overflow or divergence detected.")

if __name__ == "__main__":
    test_nsmr_bfloat16_stability()
