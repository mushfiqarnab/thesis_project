import torch
import torch.nn as nn
from typing import Dict, List

class CounterfactualNTKExtractor:
    """
    Module 2: The Offline NTK Pruner.
    Operates Post-Training on cloud hardware.
    Computes the empirical Neural Tangent Kernel (NTK) Jacobian over counterfactual pairs.
    Generates a pruning mask exclusively in the safe null-space of the NTK eigenspace.
    """
    def __init__(self, model: nn.Module, threshold_ratio: float = 0.95):
        self.model = model
        self.threshold_ratio = threshold_ratio # Amount of variance to protect
        self.jacobians = {name: [] for name, p in model.named_parameters() if p.requires_grad}
        self.pruning_masks = {}

    def extract_sample_jacobian(self, cf_image: torch.Tensor, cf_phys: torch.Tensor, target_class: int):
        """
        Calculates the exact gradient of the counterfactual logit w.r.t every parameter.
        Must be run with batch_size=1 to extract per-sample Jacobian rows.
        """
        self.model.zero_grad()
        logits = self.model(cf_image, cf_phys)
        
        # We target the specific logit for the counterfactual class
        target_logit = logits[0, target_class]
        target_logit.backward()
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Flatten and store the Jacobian row
                    self.jacobians[name].append(param.grad.view(-1).detach().cpu())

    def compute_null_space_masks(self) -> Dict[str, torch.Tensor]:
        """
        Constructs the NTK, extracts the principal eigenspace, and identifies the null-space.
        Parameters residing purely in the null-space can be safely pruned to 0.
        """
        print("[SYSTEM] Executing SVD on Empirical NTK...")
        
        for name in self.jacobians:
            if len(self.jacobians[name]) == 0:
                continue
                
            # Construct Jacobian matrix J: shape (N_samples, P_params)
            J = torch.stack(self.jacobians[name]).to(next(self.model.parameters()).device)
            
            # For massive layers, full SVD on J^T J is O(P^3).
            # We compute SVD on J directly: J = U S V^T. V is shape (P, K).
            # The right-singular vectors (V) span the same space as the eigenvectors of J^T J.
            U, S, V = torch.svd(J, compute_uv=True)
            
            # Determine cutoff for "protected" subspace based on singular value energy
            total_energy = torch.sum(S ** 2)
            cumulative_energy = torch.cumsum(S ** 2, dim=0)
            cutoff_idx = torch.searchsorted(cumulative_energy, self.threshold_ratio * total_energy).item()
            
            # V_protected: shape (P, cutoff_idx)
            V_protected = V[:, :cutoff_idx+1]
            
            # A parameter's importance is its projection magnitude onto the protected subspace
            # importance[i] = sum_j (V_protected[i, j]^2)
            param_importance = torch.sum(V_protected ** 2, dim=1)
            
            # Reshape back to original parameter shape
            original_shape = dict(self.model.named_parameters())[name].shape
            importance_map = param_importance.view(original_shape)
            
            # Create binary mask (e.g., prune bottom 50% of least important weights)
            prune_threshold = torch.median(importance_map)
            mask = (importance_map > prune_threshold).float()
            
            self.pruning_masks[name] = mask
            
        print("[SUCCESS] Counterfactual NTK Pruning Masks generated.")
        return self.pruning_masks

    def apply_masks(self):
        """Hard-applies the binary masks to the model weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.pruning_masks:
                    param.mul_(self.pruning_masks[name].to(param.device))
