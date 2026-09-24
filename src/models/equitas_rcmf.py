"""
equitas_rcmf.py
================================================================================
Riemannian Causal Manifold Fusion with Orthogonal Subspace Disentanglement
(EQUITAS-RCMF)

Core Mathematical Innovations:
1. Closed-form Stiefel Orthogonal Subspace Decomposition:
   Decomposes vision features into V_causal (facial expression) and
   V_confounder (scar artifact) such that:
       W_causal @ W_confounder^T = 0 (Strict algebraic independence)
   via exact Riemannian thin QR projection on the Stiefel manifold St(d, D).
   Supports precomputation and weight baking for zero-overhead edge deployment.

2. Bidirectional Thermodynamic Attention Gating:
   Continuous energy barrier with cross-modal query:
       G(x) = sigmoid(w^T [v_causal; P_emb] + b) * exp(-kappa * Focus(x))
   Physically suppresses visual transmission when scar attention spikes,
   without suffering from the "collateral damage" of silencing unconfounded
   facial musculature features.

3. Learning Under Privileged Information (LUPI) Confounder Bridge:
   - Privileged Mode (Training): Focus is computed from ground-truth scar masks.
   - Autonomous Mode (Inference / Edge): Focus is estimated intrinsically from
     the orthogonal confounder subspace projection v_confounder, enabling
     autonomous real-world deployment without ground-truth masks.

4. Dual-Level Invariant Latent Space Z:
   Z = G * v_causal + (1 - G) * p_emb
   Isometrically aligned across clean and scarred counterfactual pairs.
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


@dataclass
class RCMFOutput:
    logits: torch.Tensor                     # (B, num_classes)
    latent_z: torch.Tensor                   # (B, d_causal)
    v_causal: torch.Tensor                   # (B, d_causal)
    v_confounder: torch.Tensor               # (B, d_confounder)
    gate: torch.Tensor                       # (B, 1)
    focus: torch.Tensor                      # (B, 1)


class StiefelOrthogonalDecomposition(nn.Module):
    """
    Decomposes an input feature vector into two mutually orthogonal subspaces:
      S_causal (dimension d_causal) and S_confounder (dimension d_confounder)
    constrained to the Stiefel manifold St(d_c + d_b, in_features) such that:
      W_causal @ W_confounder^T = 0 (Strict algebraic independence)
    """

    def __init__(
        self,
        in_features: int,
        d_causal: int = 192,
        d_confounder: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_features = in_features
        self.d_causal = d_causal
        self.d_confounder = d_confounder
        self.total_out = d_causal + d_confounder
        self.eps = eps

        if self.total_out > in_features:
            raise ValueError(
                f"Stiefel constraint requires (d_causal + d_confounder) <= in_features, "
                f"got {self.total_out} > {in_features}"
            )

        # Joint unconstrained weight parameter of shape (total_out, in_features)
        self.weight_raw = nn.Parameter(torch.Tensor(self.total_out, in_features))
        self.bias_causal = nn.Parameter(torch.zeros(d_causal))
        self.bias_confounder = nn.Parameter(torch.zeros(d_confounder))

        # State flag and buffers for baked inference weights
        self.register_buffer("is_baked", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("W_causal_baked", torch.zeros(d_causal, in_features))
        self.register_buffer("W_confounder_baked", torch.zeros(d_confounder, in_features))

        # Initialize with exact Haar-distributed orthogonal weights
        nn.init.orthogonal_(self.weight_raw)

    def get_stiefel_matrix(self) -> torch.Tensor:
        """
        Projects self.weight_raw onto the Stiefel manifold St(total_out, in_features)
        via exact thin QR decomposition:
          W_raw^T = Q R, so W_stiefel = Q^T.
        Guarantees W_stiefel @ W_stiefel^T = I_{total_out} with machine-level precision (< 1e-6).
        Autograd backpropagates exactly through the Stiefel tangent space.
        """
        if bool(self.is_baked.item()):
            return torch.cat([self.W_causal_baked, self.W_confounder_baked], dim=0)

        # Thin QR on transposed weights: (in_features, total_out) -> Q is (in_features, total_out)
        Q_thin, _ = torch.linalg.qr(self.weight_raw.t(), mode="reduced")
        return Q_thin.t()

    def freeze_stiefel_weights(self):
        """Precomputes and bakes the Stiefel weights as fixed buffers for lightning-fast edge inference."""
        with torch.no_grad():
            W_stiefel = self.get_stiefel_matrix()
            self.W_causal_baked.copy_(W_stiefel[: self.d_causal, :])
            self.W_confounder_baked.copy_(W_stiefel[self.d_causal :, :])
            self.is_baked.fill_(True)

    def unfreeze_stiefel_weights(self):
        """Unbakes weights to resume Riemannian gradient updates."""
        self.is_baked.fill_(False)

    def forward(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if bool(self.is_baked.item()):
            v_causal = F.linear(v, self.W_causal_baked, self.bias_causal)
            v_confounder = F.linear(v, self.W_confounder_baked, self.bias_confounder)
            return v_causal, v_confounder

        W_stiefel = self.get_stiefel_matrix()
        W_causal = W_stiefel[: self.d_causal, :]
        W_confounder = W_stiefel[self.d_causal :, :]

        v_causal = F.linear(v, W_causal, self.bias_causal)
        v_confounder = F.linear(v, W_confounder, self.bias_confounder)
        return v_causal, v_confounder

    def verify_mutual_orthogonality(self) -> float:
        """Computes ||W_causal @ W_confounder^T||_F (should be ~0.0)."""
        with torch.no_grad():
            W = self.get_stiefel_matrix()
            W_c = W[: self.d_causal, :]
            W_b = W[self.d_causal :, :]
            cross = torch.matmul(W_c, W_b.t())
            return float(torch.linalg.matrix_norm(cross, ord="fro").item())


class ThermodynamicAttentionGate(nn.Module):
    """
    Thermodynamic Exponential Attention Gating:
        G(x) = sigmoid(w^T [v_causal; P_emb] + b) * exp(-kappa * Focus(x))
    Operates as a continuous topological filter:
      - When Focus ~ 0 (no scar fixation), G dynamically balances vision and physiology.
      - When Focus >> 0 (scar detected), G smoothly plunges to 0, diverting trust to physiology.
    """

    def __init__(self, d_causal: int, initial_kappa: float = 1.5):
        super().__init__()
        # Cross-modal query integrating unconfounded causal vision and physiology
        self.cross_modal_mlp = nn.Sequential(
            nn.Linear(2 * d_causal, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # Learnable log-temperature parameter (ensures kappa >= 0.1)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(initial_kappa), dtype=torch.float32))

    @property
    def kappa(self) -> torch.Tensor:
        return torch.exp(self.log_kappa).clamp(min=0.1, max=10.0)

    def forward(self, v_causal: torch.Tensor, p_emb: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
        # Cross-modal multimodal baseline gate
        fusion_query = torch.cat([v_causal, p_emb], dim=-1)
        base_gate = torch.sigmoid(self.cross_modal_mlp(fusion_query))

        # Continuous thermodynamic energy suppression
        energy_barrier = torch.exp(-self.kappa * focus.clamp(min=0.0, max=10.0))
        gate = base_gate * energy_barrier
        return gate


class EquitasRCMFModel(nn.Module):
    """
    Complete EQUITAS-RCMF Network Architecture:
      - MobileNetV3-Small Vision Encoder (576D)
      - Dynamic Physiological MLP (4D or 2D -> d_causal)
      - Stiefel Orthogonal Subspace Decomposition (576 -> 192 causal + 64 confounder)
      - Dual-Mode Thermodynamic Attention Gating (Privileged Mask & Autonomous LUPI)
      - Fused Isometric Latent Decision Head
    """

    def __init__(
        self,
        phys_dim: int = 2,
        vision_backbone: str = "mobilenet_v3_small",
        d_causal: int = 192,
        d_confounder: int = 64,
        num_classes: int = 2,
        initial_kappa: float = 1.5,
    ):
        super().__init__()
        self.d_causal = d_causal
        self.d_confounder = d_confounder
        self.phys_dim = phys_dim

        # 1. Vision Encoder
        if vision_backbone == "mobilenet_v3_small":
            m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.vision_features = m.features
            self.vision_pool = nn.AdaptiveAvgPool2d(1)
            v_dim = 576
        else:
            raise ValueError(f"Backbone not supported: {vision_backbone}")

        # 2. Physiology Encoder
        self.phys_mlp = nn.Sequential(
            nn.Linear(phys_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, d_causal),
            nn.LayerNorm(d_causal),
        )

        # 3. Riemannian Stiefel Subspace Decomposition
        self.stiefel_decomp = StiefelOrthogonalDecomposition(
            in_features=v_dim,
            d_causal=d_causal,
            d_confounder=d_confounder,
        )
        self.causal_norm = nn.LayerNorm(d_causal)

        # 4. Thermodynamic Attention Gate (Cross-Modal)
        self.gate_engine = ThermodynamicAttentionGate(
            d_causal=d_causal,
            initial_kappa=initial_kappa,
        )

        # 5. Dual-Branch Decision Heads
        # Main task classifier operating on purified multimodal latent space Z
        self.task_classifier = nn.Sequential(
            nn.Linear(d_causal, 128),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

        # Auxiliary confounder projection head (forces v_confounder to capture scar artifact)
        self.confounder_head = nn.Sequential(
            nn.Linear(d_confounder, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def freeze_stiefel_weights(self):
        """Bakes Stiefel orthogonal weights for zero-overhead inference / ONNX export."""
        self.stiefel_decomp.freeze_stiefel_weights()

    def unfreeze_stiefel_weights(self):
        """Unbakes Stiefel weights for continued training."""
        self.stiefel_decomp.unfreeze_stiefel_weights()

    @staticmethod
    def compute_concentric_focus(fmap: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Computes the ratio of activation energy concentrated in the mask:
            focus = log1p( (energy_inside / mask_pixels) / (overall_energy / total_pixels) )
        Guarantees focus == 0.0 when mask has zero foreground.
        """
        B, C, h, w = fmap.shape
        m = F.interpolate(mask.float(), size=(h, w), mode="nearest")
        energy = fmap.abs()

        mask_pix = m.sum(dim=(2, 3), keepdim=False).squeeze(1)  # (B,)
        has_foreground = (mask_pix > 0.5)

        inside_sum = (energy * m).sum(dim=(1, 2, 3))             # (B,)
        inside_mean = inside_sum / (mask_pix * C + eps)

        overall_mean = energy.mean(dim=(1, 2, 3)) + eps          # (B,)
        ratio = inside_mean / overall_mean

        focus = torch.log1p(ratio).unsqueeze(1)                  # (B, 1)
        # Strictly zero out samples with no foreground mask
        focus = torch.where(has_foreground.unsqueeze(1), focus, torch.zeros_like(focus))
        return focus

    def forward(
        self,
        img: torch.Tensor,
        phys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scar_label: Optional[torch.Tensor] = None,
    ) -> RCMFOutput:
        # Step A: Vision Feature Extraction
        fmap = self.vision_features(img)                       # (B, 576, h, w)
        v_raw = self.vision_pool(fmap).flatten(1)              # (B, 576)

        # Step B: Physiology Projection
        p_emb = self.phys_mlp(phys)                            # (B, d_causal)

        # Step C: Riemannian Stiefel Subspace Decomposition
        v_causal_raw, v_confounder = self.stiefel_decomp(v_raw)
        v_causal = self.causal_norm(v_causal_raw)              # (B, d_causal)

        # Step D: Confounder Attention Energy Calculation (Privileged vs Autonomous LUPI)
        if mask is not None:
            # Privileged mode: use exact spatial scar mask
            if scar_label is not None:
                effective_mask = mask * scar_label.view(-1, 1, 1, 1)
            else:
                effective_mask = mask
            focus = self.compute_concentric_focus(fmap, effective_mask)
        else:
            # Autonomous Edge mode: compute intrinsic focus from confounder subspace
            conf_pred = self.confounder_head(v_confounder)      # (B, 1)
            focus = torch.sigmoid(conf_pred)                    # Continuous [0, 1] probability

        # Step E: Bidirectional Thermodynamic Exponential Gating
        gate = self.gate_engine(v_causal, p_emb, focus)        # (B, 1)

        # Step F: Latent Multimodal Isometric Fusion
        latent_z = gate * v_causal + (1.0 - gate) * p_emb      # (B, d_causal)

        # Step G: Decision Logits
        logits = self.task_classifier(latent_z)

        return RCMFOutput(
            logits=logits,
            latent_z=latent_z,
            v_causal=v_causal,
            v_confounder=v_confounder,
            gate=gate,
            focus=focus,
        )
