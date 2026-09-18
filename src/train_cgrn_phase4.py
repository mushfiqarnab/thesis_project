import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Assume project structure
from src.models.cgrn_arch import CGRN
from src.models.ntk_pruner import CounterfactualNTKExtractor

def compute_hsic_loss(phi_v: torch.Tensor, confounder_z: torch.Tensor) -> torch.Tensor:
    """
    Computes the Hilbert-Schmidt Independence Criterion.
    We want the debiased feature mapped in phi_v to be entirely independent 
    of the confounder Z (e.g., the scar mask or demographic tensor).
    """
    # Centering matrices
    n = phi_v.size(0)
    H = torch.eye(n, device=phi_v.device) - (1.0 / n) * torch.ones(n, n, device=phi_v.device)
    
    # Linear kernel for the confounder (can be RBF as well)
    K_z = torch.matmul(confounder_z, confounder_z.T)
    
    # phi_v is already mapped through RFF, so its inner product approximates the RBF kernel
    K_v = torch.matmul(phi_v, phi_v.T)
    
    # HSIC = Tr(K_v * H * K_z * H)
    hsic = torch.trace(torch.matmul(torch.matmul(torch.matmul(K_v, H), K_z), H)) / ((n - 1) ** 2)
    return hsic

def train_phase4_step(model, optimizer, batch, ntk_extractor: CounterfactualNTKExtractor = None):
    v_data, p_data, labels, confounder_z = batch
    
    optimizer.zero_grad()
    
    # 1. Forward Pass (New Dict Signature)
    out = model(v_data, p_data)
    logits = out['logits']
    phi_v = out['phi_v']
    trust_score = out['trust_score']
    
    # 2. Compute Losses
    # Task Loss (Cross Entropy)
    loss_ce = F.cross_entropy(logits, labels)
    
    # HSIC Independence Loss (Crucial: forces phi_v to drop confounder correlation)
    loss_hsic = compute_hsic_loss(phi_v, confounder_z)
    
    # 3. Total Loss and Backward
    total_loss = loss_ce + 0.5 * loss_hsic
    total_loss.backward()
    
    # 4. (Optional) NTK Extraction for Counterfactual Samples
    # If this batch contains counterfactuals, we extract the Jacobian rows
    # This must be done via a separate pass per sample in reality, 
    # but theoretically orchestrated here.
    # if ntk_extractor is active...
    
    optimizer.step()
    
    return total_loss.item(), loss_hsic.item(), trust_score.mean().item()
