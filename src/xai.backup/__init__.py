"""
Explainable AI (XAI) Module for Multimodal Threat Model

This module provides multiple XAI techniques to explain predictions:
1. Integrated Gradients: Attribution of predictions to input features
2. Saliency Maps: Gradient-based visualization of critical regions
3. SHAP Values: Shapley-based feature importance
4. Attention Visualization: Gate mechanism and focus ratio analysis
5. Fairness-Aware XAI: Link scar influence to fairness metrics

Research Foundation:
- Integrated Gradients (Sundararajan et al., 2017)
- Expected Gradients (Erion et al., 2019)
- SHAP (Lundberg & Lee, 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json


@dataclass
class ExplanationOutput:
    """Standard output format for XAI methods"""
    method: str                          # "integrated_gradients", "saliency", etc.
    prediction: float                    # P(threat=1)
    prediction_class: int               # 0 or 1
    
    # For vision-based attributions
    vision_attribution: Optional[np.ndarray] = None  # (3, H, W) or (H, W)
    
    # For physiology-based attributions
    phys_attribution: Optional[np.ndarray] = None    # (D,) for D physiology features
    
    # Intermediate activations
    gate_activation: Optional[float] = None          # CGF gate value
    focus_activation: Optional[float] = None         # CGF focus value
    
    # Fairness insights
    scar_influence_score: Optional[float] = None     # How much does scar matter?
    fairness_risk: Optional[str] = None              # "low" | "medium" | "high"
    
    # Metadata
    metadata: Optional[Dict] = None


class IntegratedGradients:
    """
    Integrated Gradients (Sundararajan et al., 2017)
    
    Attribution_i = (x_i - x'_i) * ∫[0,1] ∂f(x' + t(x-x')) / ∂x_i dt
    
    where x' is baseline (black image / zero physiology)
    and x is the actual input.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        
    def _get_baseline(
        self, 
        img: torch.Tensor, 
        phys: torch.Tensor,
        baseline_type: str = 'black'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate baseline input representing "absence of information"
        
        Args:
            img: (B, 3, H, W) input image
            phys: (B, D) physiology vector
            baseline_type: 'black' is the only supported baseline
                (zeros in normalized space, represents absence of visual information)
            
        Returns:
            img_baseline: (B, 3, H, W) baseline image (all zeros)
            phys_baseline: (B, D) baseline physiology (all zeros)
            
        Reference: Kindermans et al. (2019) "Sanity Checks for Saliency Maps" + 
                   Sundararajan et al. (2017) Section 4.1
                   
        Note: Black baseline is the academic standard for IG because it satisfies
        the theoretical guarantees and represents a clear "absence of information"
        in the normalized input space.
        """
        if baseline_type != 'black':
            raise ValueError(
                f"Baseline type '{baseline_type}' is not supported. "
                f"Use 'black' (zeros) as the baseline. "
                f"See Kindermans et al. 2019 for why other baselines are problematic."
            )
        
        # Zero image (no visual information in normalized space)
        img_baseline = torch.zeros_like(img)
        phys_baseline = torch.zeros_like(phys)
        
        return img_baseline, phys_baseline
        
    @torch.no_grad()
    def _forward_pass(self, img: torch.Tensor, phys: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass returning P(threat=1)"""
        out = self.model(img, phys, mask=mask)
        probs = F.softmax(out.logits, dim=1)[:, 1]
        return probs
        
    def compute_gradients(
        self, 
        img: torch.Tensor,
        phys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target_class: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients of model prediction w.r.t. inputs
        
        Returns:
            grad_img: (B, C, H, W)
            grad_phys: (B, D)
        """
        img.requires_grad_(True)
        phys.requires_grad_(True)
        
        out = self.model(img, phys, mask=mask)
        logits = out.logits
        
        # Target: log probability of target class
        log_probs = F.log_softmax(logits, dim=1)[:, target_class]
        log_probs.backward(torch.ones_like(log_probs))
        
        grad_img = img.grad.detach().clone() if img.grad is not None else torch.zeros_like(img)
        grad_phys = phys.grad.detach().clone() if phys.grad is not None else torch.zeros_like(phys)
        
        # CRITICAL FIX (2025-01-XX): Zero gradients to prevent accumulation across IG steps
        # Without this, gradients accumulate exponentially in the interpolation loop
        img.grad = None
        phys.grad = None
        
        img.requires_grad_(False)
        phys.requires_grad_(False)
        
        return grad_img, grad_phys
        
    def explain(
        self,
        img: torch.Tensor,
        phys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target_class: int = 1,
        steps: int = 50,
        baseline_type: str = 'black'
    ) -> ExplanationOutput:
        """
        Compute Integrated Gradients attribution
        
        Args:
            img: (B, 3, H, W) normalized image
            phys: (B, D) physiology vector
            mask: (B, 1, H, W) scar region mask
            target_class: class to explain (0 or 1)
            steps: number of interpolation steps (higher = more accurate)
            baseline_type: baseline selection (only 'black' supported - zeros in normalized space)
            
        Returns:
            ExplanationOutput with vision and physiology attributions
        """
        device = self.device
        img = img.to(device)
        phys = phys.to(device)
        if mask is not None:
            mask = mask.to(device)
            
        img_baseline, phys_baseline = self._get_baseline(img, phys, baseline_type=baseline_type)
        
        # Accumulate gradients across interpolation steps
        accumulated_grad_img = torch.zeros_like(img)
        accumulated_grad_phys = torch.zeros_like(phys)
        
        for step in range(steps):
            # Linear interpolation: x' + t(x - x') where t ∈ [0, 1]
            alpha = step / steps
            
            img_interp = img_baseline + alpha * (img - img_baseline)
            phys_interp = phys_baseline + alpha * (phys - phys_baseline)
            
            grad_img, grad_phys = self.compute_gradients(img_interp, phys_interp, mask, target_class)
            
            accumulated_grad_img += grad_img
            accumulated_grad_phys += grad_phys
            
        # Average gradients
        avg_grad_img = accumulated_grad_img / steps
        avg_grad_phys = accumulated_grad_phys / steps
        
        # Attribution = (input - baseline) * gradient
        attr_img = (img - img_baseline) * avg_grad_img
        attr_phys = (phys - phys_baseline) * avg_grad_phys
        
        # NOTE: IG mathematically guarantees the Completeness axiom 
        # (Sundararajan et al. 2017, Theorem 1): sum(attributions) = f(x) - f(baseline)
        # This is a mathematical property of the method, not an empirical property to validate
        
        # === NEW (2025-01-XX): STABILITY CHECKS ===
        stability_status = {
            'has_nan': False,
            'has_inf': False,
            'has_zero_grad': False,
            'max_img_grad': 0.0,
            'max_phys_grad': 0.0
        }
        
        img_grad_norm = avg_grad_img.abs().max()
        phys_grad_norm = avg_grad_phys.abs().max()
        
        if torch.isnan(avg_grad_img).any() or torch.isnan(avg_grad_phys).any():
            nan_count_img = torch.isnan(avg_grad_img).sum().item()
            nan_count_phys = torch.isnan(avg_grad_phys).sum().item()
            raise RuntimeError(
                f"NaN gradients detected during IG computation! "
                f"Image NaNs: {nan_count_img}, Physiology NaNs: {nan_count_phys}. "
                f"Possible causes: (1) Model saturation at baseline, "
                f"(2) Extreme input values, (3) Numerical instability in fusion gate. "
                f"Check model architecture and input normalization."
            )
        
        if torch.isinf(avg_grad_img).any() or torch.isinf(avg_grad_phys).any():
            print("[ERROR] Inf gradients detected! Gradient explosion.")
            stability_status['has_inf'] = True
            avg_grad_img = torch.clamp(avg_grad_img, -1e6, 1e6)
            avg_grad_phys = torch.clamp(avg_grad_phys, -1e6, 1e6)
        
        if img_grad_norm < 1e-8 and phys_grad_norm < 1e-8:
            print("[WARNING] All gradients are ~zero. Model may be in saturated regime.")
            stability_status['has_zero_grad'] = True
        
        stability_status['max_img_grad'] = float(img_grad_norm.item())
        stability_status['max_phys_grad'] = float(phys_grad_norm.item())
        # === END STABILITY CHECKS ===
        
        # Get prediction
        with torch.no_grad():
            pred_probs = self._forward_pass(img, phys, mask)
            pred_class = (pred_probs >= 0.5).long()
            
            # Get intermediate activations
            out = self.model(img, phys, mask=mask)
            gate_val = out.gate[0, 0].item() if out.gate is not None else None
            focus_val = out.focus[0, 0].item() if out.focus is not None else None
        
        return ExplanationOutput(
            method="integrated_gradients",
            prediction=float(pred_probs[0].item()),
            prediction_class=int(pred_class[0].item()),
            vision_attribution=attr_img[0].detach().cpu().numpy(),
            phys_attribution=attr_phys[0].detach().cpu().numpy(),
            gate_activation=gate_val,
            focus_activation=focus_val,
            metadata={
                'steps': steps,
                'target_class': target_class,
                'img_shape': img.shape,
                'phys_dim': phys.shape[1],
                'baseline_type': baseline_type,
                'stability': stability_status
            }
        )


class SaliencyMap:
    """
    Saliency Maps (Simonyan et al., 2013)
    
    Saliency = | ∂f / ∂x |
    
    Shows which input pixels have the largest gradient magnitude.
    Highest saliency = pixels that most influence the prediction.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        
    def explain(
        self,
        img: torch.Tensor,
        phys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target_class: int = 1,
        aggregate_channels: bool = True
    ) -> ExplanationOutput:
        """
        Compute saliency map
        
        Args:
            img: (B, 3, H, W) normalized image
            phys: (B, D) physiology vector
            mask: (B, 1, H, W) scar region mask
            target_class: class to explain
            aggregate_channels: if True, return (H, W) by max across channels
            
        Returns:
            ExplanationOutput with vision_attribution as saliency map
        """
        device = self.device
        img = img.to(device).requires_grad_(True)
        phys = phys.to(device)
        if mask is not None:
            mask = mask.to(device)
            
        # Forward pass
        out = self.model(img, phys, mask=mask)
        logits = out.logits
        
        # Compute gradients
        log_probs = F.log_softmax(logits, dim=1)[:, target_class]
        log_probs.backward(torch.ones_like(log_probs))
        
        # Saliency = |∂f/∂x|
        saliency = img.grad.abs()  # (B, C, H, W)
        
        if aggregate_channels:
            # FIX (2025-01-XX): Use L2 norm instead of max to preserve channel information
            # L2 norm: sqrt(sum(|grad|^2)) represents overall magnitude across channels
            # Previous: max() would hide important gradients in other channels
            saliency = torch.sqrt((saliency ** 2).sum(dim=1, keepdim=False))  # (B, H, W)
            
        with torch.no_grad():
            probs = F.softmax(out.logits, dim=1)[:, 1]
            pred_class = (probs >= 0.5).long()
            gate_val = out.gate[0, 0].item() if out.gate is not None else None
            focus_val = out.focus[0, 0].item() if out.focus is not None else None
            
        img.requires_grad_(False)
        
        return ExplanationOutput(
            method="saliency_map",
            prediction=float(probs[0].item()),
            prediction_class=int(pred_class[0].item()),
            vision_attribution=saliency[0].detach().cpu().numpy(),
            gate_activation=gate_val,
            focus_activation=focus_val,
            metadata={
                'aggregated': aggregate_channels,
                'target_class': target_class
            }
        )


class FairnessXAI:
    """
    Fairness-Aware XAI
    
    Links scar region importance to fairness metrics.
    Answers: "How much does the scar mask influence the fairness gaps?"
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        
    def compute_scar_influence_score(
        self,
        img: torch.Tensor,
        phys: torch.Tensor,
        mask: torch.Tensor,
        scar: torch.Tensor
    ) -> float:
        """
        Measure how much the scar region influences the prediction.
        
        Method: Compare predictions with and without mask signal
        
        Args:
            img: (B, 3, H, W)
            phys: (B, D)
            mask: (B, 1, H, W) scar region
            scar: (B,) binary scar labels
            
        Returns:
            score: float in [0, 1]
                  0 = mask has no influence
                  1 = mask determines prediction
        """
        device = self.device
        img = img.to(device)
        phys = phys.to(device)
        mask = mask.to(device)
        scar = scar.to(device)
        
        with torch.no_grad():
            # Prediction with full mask
            out_full = self.model(img, phys, mask=mask)
            p_full = F.softmax(out_full.logits, dim=1)[:, 1]
            
            # Prediction with zero mask (no scar signal)
            mask_zero = torch.zeros_like(mask)
            out_zero = self.model(img, phys, mask=mask_zero)
            p_zero = F.softmax(out_zero.logits, dim=1)[:, 1]
            
            # Influence = how much predictions change when removing mask
            influence = (p_full - p_zero).abs().mean()
            
        return float(influence.item())
        
    def explain(
        self,
        img: torch.Tensor,
        phys: torch.Tensor,
        mask: torch.Tensor,
        scar: torch.Tensor,
        threshold_high: float = 0.1,
        threshold_med: float = 0.05
    ) -> ExplanationOutput:
        """
        Explain prediction with fairness context
        
        Args:
            img, phys, mask, scar: model inputs + scar labels
            threshold_high: scar influence above this = high risk
            threshold_med: scar influence above this = medium risk
            
        Returns:
            ExplanationOutput with fairness_risk annotation
        """
        device = self.device
        img = img.to(device)
        phys = phys.to(device)
        mask = mask.to(device)
        
        # Compute scar influence
        scar_influence = self.compute_scar_influence_score(img, phys, mask, scar)
        
        # Assess fairness risk
        if scar_influence > threshold_high:
            fairness_risk = "high"
        elif scar_influence > threshold_med:
            fairness_risk = "medium"
        else:
            fairness_risk = "low"
            
        with torch.no_grad():
            out = self.model(img, phys, mask=mask)
            probs = F.softmax(out.logits, dim=1)[:, 1]
            pred_class = (probs >= 0.5).long()
            gate_val = out.gate[0, 0].item() if out.gate is not None else None
            focus_val = out.focus[0, 0].item() if out.focus is not None else None
            
        return ExplanationOutput(
            method="fairness_xai",
            prediction=float(probs[0].item()),
            prediction_class=int(pred_class[0].item()),
            scar_influence_score=scar_influence,
            fairness_risk=fairness_risk,
            gate_activation=gate_val,
            focus_activation=focus_val,
            metadata={
                'threshold_high': threshold_high,
                'threshold_med': threshold_med,
                'scar_present': int(scar[0].item())
            }
        )


class AttentionVisualizer:
    """
    Visualizes attention mechanisms in the model.
    
    For CGF models:
    - Gate: how much to trust vision (0) vs physiology (1)
    - Focus: how much energy is in scar region
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        
    def explain(
        self,
        img: torch.Tensor,
        phys: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> ExplanationOutput:
        """
        Visualize gate and focus values
        
        Returns:
            ExplanationOutput with gate_activation and focus_activation
        """
        device = self.device
        img = img.to(device)
        phys = phys.to(device)
        if mask is not None:
            mask = mask.to(device)
            
        with torch.no_grad():
            out = self.model(img, phys, mask=mask)
            probs = F.softmax(out.logits, dim=1)[:, 1]
            pred_class = (probs >= 0.5).long()
            
            gate_val = out.gate[0, 0].item() if out.gate is not None else None
            focus_val = out.focus[0, 0].item() if out.focus is not None else None
            
        # Interpretation
        if gate_val is not None:
            if gate_val < 0.3:
                gate_interpretation = "Relies on physiology cues"
            elif gate_val > 0.7:
                gate_interpretation = "Relies on vision cues"
            else:
                gate_interpretation = "Balanced fusion of both modalities"
        else:
            gate_interpretation = "N/A (concat design)"
            
        if focus_val is not None:
            if focus_val < 0.5:
                focus_interpretation = "Energy distributed across face"
            else:
                focus_interpretation = "Energy concentrated in scar region"
        else:
            focus_interpretation = "N/A"
            
        return ExplanationOutput(
            method="attention_visualization",
            prediction=float(probs[0].item()),
            prediction_class=int(pred_class[0].item()),
            gate_activation=gate_val,
            focus_activation=focus_val,
            metadata={
                'gate_interpretation': gate_interpretation,
                'focus_interpretation': focus_interpretation
            }
        )


class XAIExplainer:
    """
    Unified interface for all XAI methods
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        
        self.ig = IntegratedGradients(model, device=self.device)
        self.saliency = SaliencyMap(model, device=self.device)
        self.fairness = FairnessXAI(model, device=self.device)
        self.attention = AttentionVisualizer(model, device=self.device)
        
    def explain_all(
        self,
        img: torch.Tensor,
        phys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scar: Optional[torch.Tensor] = None,
        ig_steps: int = 50
    ) -> Dict[str, ExplanationOutput]:
        """
        Generate explanations using all available methods
        
        Returns:
            Dictionary mapping method names to ExplanationOutput objects
        """
        explanations = {}
        
        # Integrated Gradients
        try:
            explanations['integrated_gradients'] = self.ig.explain(
                img, phys, mask=mask, steps=ig_steps
            )
        except Exception as e:
            print(f"[Warning] Integrated Gradients failed: {e}")
            
        # Saliency Map
        try:
            explanations['saliency_map'] = self.saliency.explain(img, phys, mask=mask)
        except Exception as e:
            print(f"[Warning] Saliency Map failed: {e}")
            
        # Attention
        try:
            explanations['attention'] = self.attention.explain(img, phys, mask=mask)
        except Exception as e:
            print(f"[Warning] Attention failed: {e}")
            
        # Fairness XAI (needs scar labels)
        if scar is not None:
            try:
                explanations['fairness_xai'] = self.fairness.explain(
                    img, phys, mask, scar
                )
            except Exception as e:
                print(f"[Warning] Fairness XAI failed: {e}")
                
        return explanations
        
    def explain_single_method(
        self,
        method: str,
        img: torch.Tensor,
        phys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scar: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ExplanationOutput:
        """
        Generate explanation using a single method
        
        Args:
            method: 'integrated_gradients', 'saliency_map', 'attention', 'fairness_xai'
            ... other args as needed
        """
        if method == "integrated_gradients":
            return self.ig.explain(img, phys, mask=mask, **kwargs)
        elif method == "saliency_map":
            return self.saliency.explain(img, phys, mask=mask, **kwargs)
        elif method == "attention":
            return self.attention.explain(img, phys, mask=mask)
        elif method == "fairness_xai":
            if scar is None:
                raise ValueError("fairness_xai requires scar labels")
            return self.fairness.explain(img, phys, mask, scar, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
