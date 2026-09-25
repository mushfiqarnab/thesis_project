"""Script 3: bfloat16 NSMR Stability Proof + Geometry Constant Analysis"""
import torch

# ============================================================
# SCRIPT 3: bfloat16 Newton-Schulz Manifold Retraction Proof
# ============================================================
print("=" * 60)
print("SCRIPT 3: bfloat16 NSMR STABILITY PROOF")
print("=" * 60)

torch.manual_seed(42)
W_initial = torch.randn(256, 576, dtype=torch.bfloat16)
W = W_initial.t()  # (576, 256) tall matrix for column orthogonality

frobenius_norm = torch.linalg.matrix_norm(W.to(torch.float32), ord='fro')
Q = (W.to(torch.float32) / (frobenius_norm + 1e-6)).to(torch.bfloat16)
I = torch.eye(256, dtype=torch.bfloat16)

print(f"Matrix shape: {Q.shape}, dtype: {Q.dtype}")
print(f"Frobenius norm of initial W: {frobenius_norm.item():.4f}")

dev0 = torch.linalg.matrix_norm((torch.matmul(Q.t(), Q) - I).to(torch.float32), ord='fro').item()
print(f"Initial Q^T Q deviation from I: {dev0:.6f}")
print()

for i in range(5):
    Q_T_Q = torch.matmul(Q.t(), Q)
    inner = 3.0 * I - Q_T_Q
    Q = 0.5 * torch.matmul(Q, inner)
    dev = torch.linalg.matrix_norm((torch.matmul(Q.t(), Q) - I).to(torch.float32), ord='fro').item()
    print(f"  Iteration {i+1}: Frobenius deviation = {dev:.8f}")

final = torch.linalg.matrix_norm((torch.matmul(Q.t(), Q) - I).to(torch.float32), ord='fro').item()
print()
print(f"FINAL Frobenius deviation: {final:.8f}")
verdict = "PASS" if final < 0.1 else "FAIL"
print(f"RESULT: {verdict} (threshold < 0.1)")

# ============================================================
# GEOMETRY CONSTANT RESOLUTION
# ============================================================
print()
print("=" * 60)
print("GEOMETRY CONSTANT FORENSIC ANALYSIS")
print("=" * 60)

print()
print("ACTUAL CODE (preprocess_video_mediapipe.py, line 68):")
print("  iod = distance(kps[0], kps[1])  # BlazeFace eye centers")
print("  base_size = iod * 2.590073")
print("  MARGIN = 1.5")
print("  side = int(base_size * MARGIN)")
print("  => effective w = iod * 2.590073 * 1.5 = iod * 3.885110")

code_constant = 2.590073
margin = 1.5
effective = code_constant * margin
print(f"\nEffective w/IOD ratio from code: {effective:.6f}")
print(f"Geometry audit measured mean:    3.882400")
print(f"Match? {abs(effective - 3.8824) < 0.005}")

print()
print("DEFENSE DOCUMENT CLAIMS:")
print("  w = IOD * 3.566283 * 1.5 = IOD * 5.349425")
print("  Constant 3.566283 derived from 468-landmark bizygomatic/canthus-midpoint ratio")

print()
print("ROOT CAUSE OF DISCREPANCY:")
print("  Pipeline uses: mp.solutions.face_DETECTION (BlazeFace, 6 keypoints)")
print("  Defense doc uses: mp.solutions.face_MESH (468 landmarks)")
print("  BlazeFace kps[0]=right_eye_CENTER, kps[1]=left_eye_CENTER (direct centers)")
print("  Face Mesh: eye center = midpoint of medial+lateral canthi (not same as BlazeFace)")
print("  BlazeFace IOD != Face Mesh IOD => different calibration constant")
print(f"\n  BlazeFace constant: 2.590073")
print(f"  Face Mesh constant: 3.566283")
print(f"  Ratio: {3.566283 / 2.590073:.6f} (Face Mesh IOD is ~{3.566283/2.590073:.2f}x larger than BlazeFace IOD)")
print()
print("CONCLUSION: The 3.566283 constant is derived for Face Mesh landmarks.")
print("The actual pipeline uses BlazeFace with constant 2.590073.")
print("Both are geometrically principled, but they are DIFFERENT constants.")
print("The defense document describes a DIFFERENT implementation than what ran.")
