from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


@dataclass
class ModelOut:
    logits: torch.Tensor                 # (B,2)
    gate: Optional[torch.Tensor] = None  # (B,1)
    focus: Optional[torch.Tensor] = None # (B,1)


class PhysMLP(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisionEncoder(nn.Module):
    """
    Returns:
      emb: (B,D)
      fmap: (B,C,h,w) only for CNN backbones (needed for CGF focus), else None.
    """
    def __init__(self, name: str = "mobilenet_v3_small", freeze: bool = False):
        super().__init__()
        self.name = name.lower()

        if self.name == "mobilenet_v3_small":
            m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.features = m.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.emb_dim = 576  # torchvision mobilenet_v3_small final feature channels
        elif self.name == "vit_b_16":
            vit = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
            vit.heads = nn.Identity()
            self.vit = vit
            self.emb_dim = vit.hidden_dim
        else:
            raise ValueError(f"Unknown vision backbone: {name}")

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.name == "mobilenet_v3_small":
            fmap = self.features(img)                # (B,C,h,w)
            emb = self.pool(fmap).flatten(1)         # (B,C)
            return emb, fmap
        else:
            emb = self.vit(img)                      # (B,D)
            return emb, None


class FusionConcat(nn.Module):
    def __init__(self, v_dim: int, p_dim: int, hidden: int = 128, num_classes: int = 2):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(v_dim + p_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, v: torch.Tensor, p: torch.Tensor) -> ModelOut:
        return ModelOut(logits=self.cls(torch.cat([v, p], dim=1)))


class CausalGatedFusion(nn.Module):
    """
    Innovation-1 (CGF):
      focus = energy in mask region / energy overall  (B,1)
      gate  = sigmoid(MLP([phys_proj, focus]))  -> trust vision
      fused = gate*vision_proj + (1-gate)*phys_proj
    """
    def __init__(self, v_dim: int, p_dim: int, d: int = 256, num_classes: int = 2):
        super().__init__()
        self.v_proj = nn.Linear(v_dim, d)
        self.p_proj = nn.Linear(p_dim, d)

        self.gate_mlp = nn.Sequential(
            nn.Linear(d + 1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        # small bias: start by trusting physiology a bit more
        nn.init.constant_(self.gate_mlp[-1].bias, -0.5)

        self.cls = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def focus_from_mask(fmap: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        fmap: (B,C,h,w)
        mask: (B,1,H,W)
        returns focus: (B,1)
        """
        B, C, h, w = fmap.shape
        m = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)  # (B,1,h,w)
        energy = fmap.abs()

        inside = (energy * m).mean(dim=(1, 2, 3))          # (B,)
        overall = energy.mean(dim=(1, 2, 3)) + eps         # (B,)
        focus = (inside / overall).unsqueeze(1)            # (B,1)
        return focus

    def forward(self, v: torch.Tensor, p: torch.Tensor, fmap: Optional[torch.Tensor], mask: torch.Tensor) -> ModelOut:
        v_ = self.v_proj(v)
        p_ = self.p_proj(p)

        if fmap is None:
            focus = torch.zeros((v.size(0), 1), device=v.device, dtype=v.dtype)
        else:
            focus = self.focus_from_mask(fmap, mask)

        gate_in = torch.cat([p_, focus], dim=1)            # (B,d+1)
        gate = torch.sigmoid(self.gate_mlp(gate_in))       # (B,1)

        fused = gate * v_ + (1.0 - gate) * p_
        logits = self.cls(fused)
        return ModelOut(logits=logits, gate=gate, focus=focus)


class MultimodalThreatModel(nn.Module):
    """
    fusion="concat" -> Design A
    fusion="cgf"    -> Design B (innovation)
    """
    def __init__(
        self,
        phys_dim: int,
        vision_backbone: str = "mobilenet_v3_small",
        fusion: str = "concat",
        num_classes: int = 2,
        freeze_vision: bool = False,
    ):
        super().__init__()
        self.vision = VisionEncoder(vision_backbone, freeze=freeze_vision)
        self.phys = PhysMLP(phys_dim, emb_dim=64)

        fusion = fusion.lower()
        self.fusion_name = fusion
        if fusion == "concat":
            self.fuse = FusionConcat(self.vision.emb_dim, 64, hidden=128, num_classes=num_classes)
        elif fusion == "cgf":
            self.fuse = CausalGatedFusion(self.vision.emb_dim, 64, d=256, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

    def forward(self, img: torch.Tensor, phys: torch.Tensor, mask: Optional[torch.Tensor] = None) -> ModelOut:
        v, fmap = self.vision(img)
        p = self.phys(phys)

        if self.fusion_name == "concat":
            return self.fuse(v, p)

        if mask is None:
            mask = torch.zeros((img.size(0), 1, img.size(2), img.size(3)), device=img.device, dtype=img.dtype)
        return self.fuse(v, p, fmap=fmap, mask=mask)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
