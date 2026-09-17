"""
Module: dr_ps_zocr.py
Author: Thesis Engineering Team
Description: Distributionally Robust Path-Specific Zero-Overhead Causal Referee (DR-PS-ZOCR).
             Implements minimax adversarial training over clinically derived tactile/biomechanical
             damping intervals (kappa) for facial Action Units, ensuring zero inference-time latency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class BoundedKappaAdversary(nn.Module):
    """
    Treats structural facial muscle damping (kappa) as an adversarial nuisance parameter 
    bounded strictly by clinical literature rather than unlearned point estimations.
    """
    def __init__(self, zone_bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        super().__init__()
        if zone_bounds is None:
            # Derive brow zone kappa_max using Naeije et al.'s 10% clinical asymmetry cutoff:
            # scarred_excursion = normal * (1 - cutoff) / (1 + cutoff) -> kappa_max = 1 - (scarred / normal)
            asymmetry_cutoff = 0.10
            brow_kappa_max = 1.0 - ((1.0 - asymmetry_cutoff) / (1.0 + asymmetry_cutoff)) # ~0.1818
            
            # Perioral zone anchored in burn-contracture microstomia clinical recovery data (15mm to 55mm aperture)
            perioral_kappa_max = 1.0 - (15.0 / 55.0) # ~0.7273

            zone_bounds = {
                "brow": (0.0, float(brow_kappa_max)),
                "perioral": (0.0, float(perioral_kappa_max)),
                "general_scar": (0.0, 0.35)
            }
            
        self.zone_bounds = zone_bounds
        # Initialize adversarial kappa parameters at the midpoint of each clinical interval box
        self.kappa = nn.ParameterDict({
            zone: nn.Parameter(torch.tensor((lo + hi) / 2.0, dtype=torch.float32))
            for zone, (lo, hi) in zone_bounds.items()
        })

    def project(self) -> None:
        """Projects adversarial kappa parameters back into their strict clinical bounding box."""
        with torch.no_grad():
            for zone, (lo, hi) in self.zone_bounds.items():
                self.kappa[zone].clamp_(lo, hi)

    def forward(self, zone: str) -> torch.Tensor:
        return self.kappa[zone]


class DRPSZOCRModule(nn.Module):
    """
    Path-Specific Causal Referee integrating subspace projection and distributionally robust 
    physical consistency checks.
    """
    def __init__(self, latent_dim: int = 256, proxy_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.proxy_dim = proxy_dim
        
        # Linear projection masks to separate proxy (illegitimate social shortcut) 
        # from resolving (legitimate structural/invariant) subspaces
        self.proxy_projection = nn.Linear(latent_dim, proxy_dim, bias=False)
        self.resolve_projection = nn.Linear(latent_dim, latent_dim - proxy_dim, bias=False)
        
        # Initialize orthogonal projection matrices
        nn.init.orthogonal_(self.proxy_projection.weight)
        nn.init.orthogonal_(self.resolve_projection.weight)

    def compute_physical_consistency_loss(
        self, 
        A_f: torch.Tensor, 
        A_c_raw: torch.Tensor, 
        kappa_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates physical consistency by scaling landmark displacement/amplitude 
        by 1 / (1 - kappa) to undo mechanical muscle damping in the counterfactual view.
        """
        # Ensure numerical stability
        clamped_kappa = torch.clamp(kappa_val, min=0.0, max=0.95)
        A_corrected = A_c_raw / (1.0 - clamped_kappa + 1e-4)
        
        # Cycle-consistency check between physical expectations
        expected_damping = A_corrected * (1.0 - clamped_kappa)
        loss_phys = F.mse_loss(expected_damping, A_f)
        
        return loss_phys, A_corrected

    def compute_causal_invariance_loss(
        self, 
        z_f: torch.Tensor, 
        z_c_corrected: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enforces representation-level counterfactual invariance across projected subspaces."""
        diff = z_f - z_c_corrected
        
        # Project into proxy and resolving subspaces
        z_f_proxy = self.proxy_projection(z_f)
        z_c_proxy = self.proxy_projection(z_c_corrected)
        
        z_f_resolve = self.resolve_projection(z_f)
        z_c_resolve = self.resolve_projection(z_c_corrected)
        
        l_proxy = F.mse_loss(z_f_proxy, z_c_proxy)
        l_resolve = F.mse_loss(z_f_resolve, z_c_resolve)
        
        return l_proxy, l_resolve


def perform_dr_ps_zocr_minimax_step(
    model: nn.Module,
    kappa_adversary: BoundedKappaAdversary,
    optimizer_theta: torch.optim.Optimizer,
    optimizer_kappa: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    A_f: torch.Tensor,
    A_c_raw: torch.Tensor,
    scar_zone: str,
    steps_kappa: int = 3,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5
) -> float:
    """
    Executes the minimax adversarial training loop:
    1. Ascent: Maximizes loss with respect to kappa within the clinical bounding box.
    2. Descent: Minimizes model weights (theta) against the worst-case physical damping state.
    """
    kappa_val = kappa_adversary(scar_zone)
    
    # -----------------------------------------------------------------
    # PHASE 1: ASCENT (Find worst-case kappa within clinical bounds)
    # -----------------------------------------------------------------
    for _ in range(steps_kappa):
        optimizer_kappa.zero_grad()
        loss_val, _ = model.compute_adversarial_loss(x, y, A_f, A_c_raw, kappa_val, alpha, beta, gamma)
        # Maximize loss -> minimize negative loss
        (-loss_val).backward(retain_graph=True)
        optimizer_kappa.step()
        kappa_adversary.project() # Enforce strict adherence to clinical bounds
        kappa_val = kappa_adversary(scar_zone)

    # -----------------------------------------------------------------
    # PHASE 2: DESCENT (Train network theta against worst-case kappa)
    # -----------------------------------------------------------------
    optimizer_theta.zero_grad()
    total_loss, metrics = model.compute_adversarial_loss(
        x, y, A_f, A_c_raw, kappa_val.detach(), alpha, beta, gamma
    )
    total_loss.backward()
    optimizer_theta.step()
    
    return float(total_loss.item())
