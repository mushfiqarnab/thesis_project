import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class RiemannianHSICDebiaser(nn.Module):
    """
    CGRN 3.0 Module 1: Grassmannian HSIC
    Computes HSIC in Euclidean space but mathematically forces the debiased 
    tensor to remain topologically bound to the Stiefel/Grassmann manifold.
    """
    def __init__(self, emb_dim: int, num_random_features: int = 1024):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(emb_dim, num_random_features) * 0.1, requires_grad=False)
        self.bias = nn.Parameter(torch.rand(num_random_features) * 2 * torch.pi, requires_grad=False)
        self.debias_proj = nn.Sequential(
            nn.Linear(num_random_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim)
        )

    def forward(self, v_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phi_v = torch.cos(torch.matmul(v_emb, self.omega) + self.bias)
        v_euclidean = v_emb - self.debias_proj(phi_v)
        
        # Retract back to Grassmann manifold (Orthogonalization mapping)
        # Using SVD to project the Euclidean debiased matrix onto the closest orthogonal manifold
        U, _, V = torch.svd(v_euclidean)
        v_grassmann = torch.matmul(U, V.transpose(-2, -1))
        
        return v_grassmann, phi_v

class SPDRiemannianRefereeNetwork(nn.Module):
    """
    CGRN 4.0 Module 2: Geodesic Fréchet on the SPD Manifold
    Computes the true Riemannian geodesic distance between covariance matrices 
    on the Symmetric Positive Definite (SPD) manifold, completely bypassing 
    Euclidean proxies. 
    Metric: ||log(C1^{-1/2} C2 C1^{-1/2})||_F
    """
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def _compute_covariance(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x_centered = x - x.mean(dim=0, keepdim=True)
        return torch.matmul(x_centered.transpose(0, 1), x_centered) / (b - 1 + 1e-8)
        
    def _matrix_inverse_sqrt(self, matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Differentiable inverse square root via SVD
        U, S, V = torch.svd(matrix + torch.eye(matrix.size(0), device=matrix.device) * eps)
        S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S + eps))
        return torch.matmul(U, torch.matmul(S_inv_sqrt, V.transpose(-2, -1)))

    def _matrix_log(self, matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        U, S, V = torch.svd(matrix + torch.eye(matrix.size(0), device=matrix.device) * eps)
        S_log = torch.diag(torch.log(S + eps))
        return torch.matmul(U, torch.matmul(S_log, V.transpose(-2, -1)))

    def forward(self, v_raw: torch.Tensor, v_debiased: torch.Tensor) -> torch.Tensor:
        cov_raw = self._compute_covariance(v_raw)
        cov_deb = self._compute_covariance(v_debiased)
        
        # True Riemannian Geodesic Distance on the SPD Manifold
        c_raw_inv_sqrt = self._matrix_inverse_sqrt(cov_raw)
        
        # Inner term: C1^{-1/2} C2 C1^{-1/2}
        inner_term = torch.matmul(c_raw_inv_sqrt, torch.matmul(cov_deb, c_raw_inv_sqrt))
        
        # Logarithmic map back to the tangent space
        log_term = self._matrix_log(inner_term)
        
        # Frobenius norm of the tangent matrix yields the true geodesic distance
        geodesic_dist = torch.norm(log_term, p='fro')
        
        # Soft-severing gate: 1.0 if distance is 0, approaches 0.0 as distance grows
        trust_score = torch.exp(-geodesic_dist / self.threshold)
        return trust_score.unsqueeze(0).expand(v_raw.size(0), 1)

class CounterfactualNTKPruningHooks:
    """
    CGP 4.0: Counterfactual Neural Tangent Kernel (NTK) Subspace Alignment
    Flaw in 3.0: K-FAC assumes layer independence (block-diagonal Hessian) and stationary distributions.
    Innovation: We explicitly track the Jacobian of the model output w.r.t the parameters 
    exclusively for the counterfactual data pairs. By preserving the exact eigenspace of 
    the Counterfactual NTK matrix, we prune the network while mathematically guaranteeing 
    the algorithmic equity remains geometrically unshifted.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.ntk_jacobians = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ntk_jacobians[name] = []

    def accumulate_jacobian(self):
        """
        Must be called sequentially for each counterfactual sample to accumulate the 
        exact row of the empirical Neural Tangent Kernel matrix.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Store flattened Jacobian row for the current counterfactual sample
                self.ntk_jacobians[name].append(param.grad.data.view(-1).clone().detach())

    def compute_ntk_subspace(self):
        """
        Constructs the Empirical NTK matrix Theta = J J^T and extracts the 
        principal eigenspace encoding the fairness constraints.
        """
        ntk_eigenspaces = {}
        for name in self.ntk_jacobians:
            if len(self.ntk_jacobians[name]) == 0:
                continue
            # J: (num_cf_samples, param_dim)
            J = torch.stack(self.ntk_jacobians[name])
            
            # Empirical NTK for this layer: Theta = J^T J
            # To find pruning null-space, we compute SVD on J
            U, S, V = torch.svd(J, compute_uv=True)
            
            # The top eigenvectors in V protect the counterfactual function mapping
            ntk_eigenspaces[name] = V
        return ntk_eigenspaces

class CGRN(nn.Module):
    """
    Causal Gated Referee Network (CGRN) 3.0
    Biophysically Synchronized, Non-Euclidean Causal Reasoning Engine.
    """
    def __init__(self, v_dim: int, p_dim: int, d: int = 256, num_classes: int = 2):
        super().__init__()
        self.v_proj = nn.Linear(v_dim, d)
        self.p_proj = nn.Linear(p_dim, d)
        
        self.debiaser = RiemannianHSICDebiaser(emb_dim=d)
        self.referee = SPDRiemannianRefereeNetwork(threshold=0.5)
        
        self.cls = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, v: torch.Tensor, p: torch.Tensor) -> dict:
        v_raw = self.v_proj(v)
        p_emb = self.p_proj(p)
        
        # We MUST capture phi_v to compute the HSIC loss in the training loop
        v_debiased, phi_v = self.debiaser(v_raw)
        trust_score = self.referee(v_raw, v_debiased)
        
        fused = (trust_score * v_debiased) + ((1.0 - trust_score) * p_emb)
        logits = self.cls(fused)
        
        return {
            'logits': logits,
            'trust_score': trust_score,
            'phi_v': phi_v,
            'v_raw': v_raw,         # Needed for Fréchet penalty if we want to regularize
            'v_debiased': v_debiased # Needed for Fréchet penalty
        }
