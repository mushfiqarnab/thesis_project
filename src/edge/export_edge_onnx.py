import torch
import torch.nn as nn
from pathlib import Path
import sys

# Assume models_arch is available in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from models_arch import MultimodalThreatModel, CGRN
except ImportError:
    pass # Handled for scaffolding

class StaticMahalanobisReferee(nn.Module):
    """
    O(1) Edge-Native Gating Mechanism.
    Replaces dynamic SVD/Covariance calculations with a static Mahalanobis projection.
    This projects the incoming embedding into the pre-computed topological space of the training set.
    """
    def __init__(self, mu_train: torch.Tensor, p_matrix: torch.Tensor, threshold: float = 0.5):
        super().__init__()
        # Register as buffers so they are part of the state_dict but not updated by gradients
        self.register_buffer('mu', mu_train)
        self.register_buffer('P', p_matrix)  # P = Sigma^{-1/2}
        self.threshold = threshold

    def forward(self, v_raw: torch.Tensor, v_debiased: torch.Tensor) -> torch.Tensor:
        # 1. Center the debiased feature
        centered = v_debiased - self.mu
        
        # 2. Project into Mahalanobis space (O(1) Matrix Multiplication)
        projected = torch.matmul(centered, self.P)
        
        # 3. L2 Norm in projected space is exactly the Mahalanobis distance
        mahalanobis_dist = torch.norm(projected, p=2, dim=1, keepdim=True)
        
        # 4. Exponential gating based on static topological deviation
        trust_score = torch.exp(-mahalanobis_dist / self.threshold)
        return trust_score

class EdgeCGRN(nn.Module):
    """
    The wrapper model that replaces the dynamic CGRN with the static Edge-Native components.
    """
    def __init__(self, core_model: nn.Module, mu_train: torch.Tensor, p_matrix: torch.Tensor):
        super().__init__()
        self.v_proj = core_model.v_proj
        self.p_proj = core_model.p_proj
        self.debiaser = core_model.debiaser # Retains the static omega/bias for HSIC RFF
        self.referee = StaticMahalanobisReferee(mu_train, p_matrix)
        self.cls = core_model.cls

    def forward(self, v: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        v_raw = self.v_proj(v)
        p_emb = self.p_proj(p)
        
        # Euclidean HSIC projection (static linear/cos ops, natively supported in Opset 14)
        v_debiased, _ = self.debiaser(v_raw)
        
        # NOTE: SVD Retraction is explicitly stripped here. The offline model learned the 
        # Grassmannian mapping, so the weights in `debiaser` are already topologically aligned.
        # Enforcing SVD at inference is mathematically redundant and impossible on the NPU.
        
        trust_score = self.referee(v_raw, v_debiased)
        fused = (trust_score * v_debiased) + ((1.0 - trust_score) * p_emb)
        return self.cls(fused)

def compile_edge_artifact(ckpt_path: Path, calibration_dataloader, out_path: Path):
    """
    1. Loads the offline Riemannian model.
    2. Calibrates the Mahalanobis topological matrix via a forward pass on training data.
    3. Fuses the matrix into the EdgeCGRN.
    4. Compiles strictly to ONNX Opset 14.
    """
    print("[SYSTEM] Initiating Asymmetric Edge Compilation...")
    
    # 1. Load Model (Scaffolded)
    # core_model = torch.load(ckpt_path)
    core_model = CGRN(v_dim=576, p_dim=64) # Placeholder for actual loaded weights
    core_model.eval()
    
    # 2. Extract Topological Covariance Offline
    all_debiased = []
    with torch.no_grad():
        # In practice, iterate over calibration_dataloader
        # Here we simulate with a dummy calibration batch
        dummy_v = torch.randn(100, 576)
        v_raw = core_model.v_proj(dummy_v)
        v_deb, _ = core_model.debiaser(v_raw)
        all_debiased.append(v_deb)
        
    V_train = torch.cat(all_debiased, dim=0)
    mu_train = V_train.mean(dim=0, keepdim=True)
    
    # Compute true covariance
    centered = V_train - mu_train
    Sigma = torch.matmul(centered.T, centered) / (V_train.size(0) - 1)
    
    # Compute inverse square root (P-matrix) via SVD ONCE offline
    U, S, V = torch.svd(Sigma + torch.eye(Sigma.size(0)) * 1e-5)
    P_matrix = torch.matmul(U, torch.matmul(torch.diag(1.0 / torch.sqrt(S)), V.T))
    
    print(f"[MATHEMATICS] Mahalanobis Projection Matrix computed. Condition Number: {S[0]/S[-1]:.4f}")
    
    # 3. Fuse into static edge graph
    edge_model = EdgeCGRN(core_model, mu_train, P_matrix)
    edge_model.eval()
    
    # 4. Export to ONNX Opset 14
    dummy_vision = torch.randn(1, 576)
    dummy_phys = torch.randn(1, 64)
    
    torch.onnx.export(
        edge_model,
        (dummy_vision, dummy_phys),
        str(out_path),
        export_params=True,
        opset_version=14,  # Crucial for native LayerNorm support inside the architecture
        do_constant_folding=True,
        input_names=['vision_input', 'phys_input'],
        output_names=['threat_logits'],
        dynamic_axes={
            'vision_input': {0: 'batch_size'},
            'phys_input': {0: 'batch_size'},
            'threat_logits': {0: 'batch_size'}
        }
    )
    print(f"[SUCCESS] Edge Graph baked and locked. Saved to: {out_path}")

if __name__ == "__main__":
    compile_edge_artifact(Path("dummy.pt"), None, Path("outputs/strict_edge_cgrn.onnx"))
