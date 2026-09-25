"""Diagnose and fix the NSMR scaling — find correct initialization."""
import torch

torch.manual_seed(42)
print("=== DIAGNOSING NSMR SCALING ISSUE ===")
print()

# Original script used Frobenius scaling -> singular values ~0.074 (too small)
# NSMR requires singular values NEAR 1.0 (spectral radius of Q^T Q - I must be < 1)

W_tall = torch.randn(576, 256, dtype=torch.float32)
torch.manual_seed(42)
W_tall = torch.randn(576, 256, dtype=torch.float32)

fn = torch.linalg.matrix_norm(W_tall, ord='fro')
Q_frob = W_tall / (fn + 1e-6)
sv_frob = torch.linalg.svdvals(Q_frob)
print("FROBENIUS NORM SCALING:")
print(f"  Frobenius norm of W: {fn.item():.2f}")
print(f"  Singular values of Q: min={sv_frob.min():.4f}, max={sv_frob.max():.4f}, mean={sv_frob.mean():.4f}")
ev_frob = sv_frob**2
print(f"  Q^T Q eigenvalues:    min={ev_frob.min():.6f}, max={ev_frob.max():.6f}")
print(f"  Spectral radius of (Q^T Q - I): {(ev_frob - 1).abs().max():.6f}")
dev_frob = torch.linalg.matrix_norm(Q_frob.t() @ Q_frob - torch.eye(256), ord='fro').item()
print(f"  Initial Frobenius deviation: {dev_frob:.4f}  <-- this is too large for NSMR to converge in 5 iters")

print()
print("SPECTRAL NORM SCALING (correct approach for NSMR):")
torch.manual_seed(42)
W_tall = torch.randn(576, 256, dtype=torch.float32)
spec_norm = torch.linalg.svdvals(W_tall).max()
Q_spec = W_tall / (spec_norm + 1e-6)
sv_spec = torch.linalg.svdvals(Q_spec)
print(f"  Spectral norm of W: {spec_norm.item():.4f}")
print(f"  Singular values of Q: min={sv_spec.min():.4f}, max={sv_spec.max():.4f}, mean={sv_spec.mean():.4f}")
ev_spec = sv_spec**2
print(f"  Q^T Q eigenvalues:    min={ev_spec.min():.6f}, max={ev_spec.max():.6f}")
print(f"  Spectral radius of (Q^T Q - I): {(ev_spec - 1).abs().max():.6f}")
dev_spec = torch.linalg.matrix_norm(Q_spec.t() @ Q_spec - torch.eye(256), ord='fro').item()
print(f"  Initial Frobenius deviation: {dev_spec:.4f}")

print()
print("=== RUNNING NSMR WITH SPECTRAL NORM SCALING IN bfloat16 ===")
torch.manual_seed(42)
W_bf = torch.randn(576, 256, dtype=torch.bfloat16)
spec_norm_bf = torch.linalg.svdvals(W_bf.to(torch.float32)).max()
Q = (W_bf.to(torch.float32) / (spec_norm_bf + 1e-6)).to(torch.bfloat16)
I = torch.eye(256, dtype=torch.bfloat16)

dev0 = torch.linalg.matrix_norm((Q.t() @ Q - I).to(torch.float32), ord='fro').item()
print(f"Initial deviation (bfloat16, spectral scaled): {dev0:.6f}")
print()

for i in range(5):
    QtQ = torch.matmul(Q.t(), Q)
    Q = 0.5 * torch.matmul(Q, 3.0 * I - QtQ)
    dev = torch.linalg.matrix_norm((Q.t() @ Q - I).to(torch.float32), ord='fro').item()
    print(f"  Iteration {i+1}: Frobenius deviation = {dev:.8f}")

final = torch.linalg.matrix_norm((Q.t() @ Q - I).to(torch.float32), ord='fro').item()
print()
print(f"FINAL Frobenius deviation: {final:.8f}")
verdict = "PASS" if final < 0.1 else "FAIL"
print(f"RESULT: {verdict} (threshold < 0.1)")

print()
print("=== ROOT CAUSE DIAGNOSIS ===")
print("The defense document uses Frobenius norm scaling.")
print("NSMR convergence requires spectral norm scaling (max singular value).")
print("The Frobenius norm is ~sqrt(n) * spectral_norm, which over-shrinks singular values.")
print("When singular values << 1, Q^T Q << I, deviation is large, 5 iters insufficient.")
print("Spectral norm scaling guarantees all singular values <= 1 -> rho(Q^T Q - I) <= 1 -> converges.")
