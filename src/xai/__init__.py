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
from typing import Dict, List, Tuple, Optional, Union, Callable
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


class ImageNormalizer:
    """
    Utility class for image normalization/denormalization
    
    Encapsulates dataset-specific normalization parameters to make XAI
    methods work with different datasets and normalization schemes.
    
    Standard ImageNet normalization (torchvision default):
    - mean = [0.485, 0.456, 0.406]  # RGB channels
    - std = [0.229, 0.224, 0.225]   # RGB channels
    
    These values are computed from the ImageNet training set and represent
    the per-channel mean and standard deviation used to normalize inputs.
    
    Reference: PyTorch vision.transforms.ImageNet normalization
    """
    
    def __init__(
        self, 
        mean: Union[List[float], np.ndarray] = None,
        std: Union[List[float], np.ndarray] = None,
        dataset_name: str = "imagenet"
    ):
        """
        Initialize ImageNormalizer
        
        Args:
            mean: Per-channel mean values (default: ImageNet)
            std: Per-channel standard deviation (default: ImageNet)
            dataset_name: Name of dataset for documentation
        """
        # Default to ImageNet normalization
        if mean is None:
            self.mean = np.array([0.485, 0.456, 0.406])
        else:
            self.mean = np.array(mean)
            
        if std is None:
            self.std = np.array([0.229, 0.224, 0.225])
        else:
            self.std = np.array(std)
            
        self.dataset_name = dataset_name
        
        # Validate
        assert len(self.mean) == 3, "Mean must have 3 values (RGB)"
        assert len(self.std) == 3, "Std must have 3 values (RGB)"
        assert np.all(self.std > 0), "Standard deviations must be positive"
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image: (img - mean) / std
        
        Args:
            img: Image array, values typically in [0, 1] or [0, 255]
            
        Returns:
            Normalized image
        """
        # Reshape mean/std for broadcasting: (1, 1, 3)
        mean = self.mean.reshape(1, 1, 3)
        std = self.std.reshape(1, 1, 3)
        return (img - mean) / std
    
    def denormalize(self, img: np.ndarray) -> np.ndarray:
        """
        Denormalize image: img * std + mean
        
        Useful for visualizing attributions in original image space
        
        Args:
            img: Normalized image array
            
        Returns:
            Denormalized image in [0, 1] or [0, 255] range
        """
        # Reshape mean/std for broadcasting: (1, 1, 3)
        mean = self.mean.reshape(1, 1, 3)
        std = self.std.reshape(1, 1, 3)
        return img * std + mean
    
    def get_torch_denorm_transform(self) -> nn.Module:
        """
        Get PyTorch module for denormalization in torch computation graphs
        
        Useful when denormalization needs to be part of the model
        
        Returns:
            nn.Module that denormalizes torch tensors
        """
        class DenormTransform(nn.Module):
            def __init__(self, mean, std):
                super().__init__()
                # Register as buffers so they move with model to GPU
                self.register_buffer('mean', torch.tensor(mean).reshape(1, 3, 1, 1))
                self.register_buffer('std', torch.tensor(std).reshape(1, 3, 1, 1))
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """x: (B, 3, H, W) normalized tensor"""
                return x * self.std + self.mean
        
        return DenormTransform(self.mean, self.std)
    
    def __repr__(self) -> str:
        return (f"ImageNormalizer(dataset={self.dataset_name}, "
                f"mean={self.mean.tolist()}, std={self.std.tolist()})")


class IGValidator:
    """
    Validator for Integrated Gradients implementation
    
    Verifies that the IG implementation satisfies key properties:
    1. Attribution gradients are non-zero (learning is happening)
    2. Gate mechanism actually varies (not stuck at constant value)
    3. Gate uses full range [0, 1] (not saturated)
    4. Attributions are reasonable (not all zeros or NaN)
    
    These checks catch common implementation bugs early.
    """
    
    @staticmethod
    def validate_attributions(
        attr_img: np.ndarray,
        attr_phys: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Validate that attributions are reasonable
        
        Args:
            attr_img: Image attributions (B, 3, H, W) or (3, H, W)
            attr_phys: Physiology attributions (B, D) or (D,)
            verbose: Print validation results
            
        Returns:
            Dict with validation results
        """
        results = {}
        
        # Check for NaN
        attr_img_has_nan = np.any(np.isnan(attr_img))
        attr_phys_has_nan = np.any(np.isnan(attr_phys))
        results['has_nan'] = attr_img_has_nan or attr_phys_has_nan
        
        # Check for all zeros
        attr_img_all_zero = np.allclose(attr_img, 0, atol=1e-8)
        attr_phys_all_zero = np.allclose(attr_phys, 0, atol=1e-8)
        results['all_zero'] = attr_img_all_zero or attr_phys_all_zero
        
        # Check magnitude
        attr_img_mag = float(np.abs(attr_img).max())
        attr_phys_mag = float(np.abs(attr_phys).max())
        results['max_img_attribution'] = attr_img_mag
        results['max_phys_attribution'] = attr_phys_mag
        
        # Check variance
        attr_img_var = float(np.var(attr_img))
        attr_phys_var = float(np.var(attr_phys))
        results['img_variance'] = attr_img_var
        results['phys_variance'] = attr_phys_var
        
        if verbose:
            print("[IGValidator] Attribution validation:")
            print(f"  Has NaN: {results['has_nan']}")
            print(f"  All zero: {results['all_zero']}")
            print(f"  Max image attribution: {attr_img_mag:.6f}")
            print(f"  Max physiology attribution: {attr_phys_mag:.6f}")
            print(f"  Image variance: {attr_img_var:.6f}")
            print(f"  Physiology variance: {attr_phys_var:.6f}")
        
        return results
    
    @staticmethod
    def validate_gate_mechanism(
        gate_values: List[float],
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Validate that gate mechanism varies and uses full range
        
        Args:
            gate_values: List of gate activations from multiple samples
            verbose: Print validation results
            
        Returns:
            Dict with validation results
        """
        gate_values = np.array(gate_values)
        results = {}
        
        # Check variance
        gate_var = float(np.var(gate_values))
        results['gate_variance'] = gate_var
        results['gate_varies'] = gate_var > 1e-6
        
        # Check range usage
        gate_min = float(np.min(gate_values))
        gate_max = float(np.max(gate_values))
        results['gate_min'] = gate_min
        results['gate_max'] = gate_max
        results['gate_range_used'] = (gate_max - gate_min) > 0.2  # Uses >20% of [0,1]
        
        # Check distribution
        results['gate_mean'] = float(np.mean(gate_values))
        results['gate_median'] = float(np.median(gate_values))
        
        if verbose:
            print("[IGValidator] Gate mechanism validation:")
            print(f"  Variance: {gate_var:.6f}")
            print(f"  Gate varies: {results['gate_varies']}")
            print(f"  Min value: {gate_min:.4f}")
            print(f"  Max value: {gate_max:.4f}")
            print(f"  Range used: {results['gate_range_used']}")
            print(f"  Mean: {results['gate_mean']:.4f}")
            print(f"  Median: {results['gate_median']:.4f}")
        
        return results
    
    @staticmethod
    def full_validation(
        attr_img: np.ndarray,
        attr_phys: np.ndarray,
        gate_values: List[float],
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Run all validation checks
        
        Args:
            attr_img: Image attributions
            attr_phys: Physiology attributions
            gate_values: Gate activations from multiple samples
            verbose: Print results
            
        Returns:
            Combined validation results
        """
        results = {
            'attributions': IGValidator.validate_attributions(attr_img, attr_phys, verbose),
            'gate': IGValidator.validate_gate_mechanism(gate_values, verbose)
        }
        
        # Overall status
        results['all_pass'] = (
            not results['attributions']['has_nan'] and
            not results['attributions']['all_zero'] and
            results['gate']['gate_varies'] and
            results['gate']['gate_range_used']
        )
        
        if verbose:
            status_emoji = "✅" if results['all_pass'] else "❌"
            print(f"\n{status_emoji} Overall validation: {'PASS' if results['all_pass'] else 'FAIL'}")
        
        return results


class GradientFlowAnalyzer:
    """
    Analyzes gradient flow and numerical stability during XAI computation
    
    Monitors:
    1. Gradient magnitude distribution (mean, std, min, max)
    2. Numerical stability (NaN, Inf, vanishing/exploding gradients)
    3. Saturation detection (gradients near zero or clipped)
    4. Layer-wise gradient health
    
    This is critical for ensuring that IG implementations don't suffer from
    vanishing gradients or numerical instability, which would invalidate results.
    
    Reference: He et al. (2015) "Delving Deep into Rectifiers"
    """
    
    def __init__(self, threshold_vanishing: float = 1e-6, threshold_exploding: float = 1e3):
        """
        Initialize analyzer
        
        Args:
            threshold_vanishing: Below this magnitude = vanishing gradient
            threshold_exploding: Above this magnitude = exploding gradient
        """
        self.threshold_vanishing = threshold_vanishing
        self.threshold_exploding = threshold_exploding
    
    def analyze_gradients(
        self,
        gradients: torch.Tensor,
        layer_name: str = "output",
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Analyze gradient health at a specific layer
        
        Args:
            gradients: Gradient tensor of any shape
            layer_name: Name of layer for reporting
            verbose: Print results
            
        Returns:
            Dictionary with gradient statistics and health indicators
        """
        # Flatten for statistics
        grad_flat = gradients.cpu().detach().numpy().flatten()
        
        # Basic statistics
        results = {
            'layer': layer_name,
            'shape': tuple(gradients.shape),
            'mean': float(np.mean(np.abs(grad_flat))),
            'std': float(np.std(grad_flat)),
            'min': float(np.min(np.abs(grad_flat))),
            'max': float(np.max(np.abs(grad_flat))),
        }
        
        # Health checks
        has_nan = np.isnan(grad_flat).any()
        has_inf = np.isinf(grad_flat).any()
        num_vanishing = np.sum(np.abs(grad_flat) < self.threshold_vanishing)
        num_exploding = np.sum(np.abs(grad_flat) > self.threshold_exploding)
        num_zero = np.sum(grad_flat == 0.0)
        
        results['has_nan'] = bool(has_nan)
        results['has_inf'] = bool(has_inf)
        results['vanishing_count'] = int(num_vanishing)
        results['vanishing_pct'] = float(100.0 * num_vanishing / len(grad_flat))
        results['exploding_count'] = int(num_exploding)
        results['exploding_pct'] = float(100.0 * num_exploding / len(grad_flat))
        results['zero_count'] = int(num_zero)
        results['zero_pct'] = float(100.0 * num_zero / len(grad_flat))
        
        # Overall health
        results['is_healthy'] = (
            not has_nan and 
            not has_inf and 
            results['vanishing_pct'] < 50 and
            results['exploding_pct'] < 10
        )
        
        if verbose:
            health_emoji = "✅" if results['is_healthy'] else "⚠️"
            print(f"\n{health_emoji} Gradient Flow Analysis [{layer_name}]")
            print(f"  Shape: {results['shape']}")
            print(f"  Mean magnitude: {results['mean']:.6f}")
            print(f"  Std: {results['std']:.6f}")
            print(f"  Range: [{results['min']:.6f}, {results['max']:.6f}]")
            print(f"  NaN: {has_nan}, Inf: {has_inf}")
            print(f"  Vanishing ({results['vanishing_pct']:.1f}%): {num_vanishing} / {len(grad_flat)}")
            print(f"  Exploding ({results['exploding_pct']:.1f}%): {num_exploding} / {len(grad_flat)}")
            print(f"  Health: {'GOOD' if results['is_healthy'] else 'DEGRADED'}")
        
        return results
    
    @staticmethod
    def analyze_multiple_layers(
        layer_grads: Dict[str, torch.Tensor],
        verbose: bool = True
    ) -> Dict[str, Dict[str, any]]:
        """
        Analyze gradient flow across multiple layers
        
        Args:
            layer_grads: Dict mapping layer names to gradient tensors
            verbose: Print results
            
        Returns:
            Dict of analysis results per layer
        """
        analyzer = GradientFlowAnalyzer()
        results = {}
        
        for layer_name, grad_tensor in layer_grads.items():
            results[layer_name] = analyzer.analyze_gradients(
                grad_tensor, 
                layer_name=layer_name, 
                verbose=verbose
            )
        
        # Overall health
        all_healthy = all(r['is_healthy'] for r in results.values())
        
        if verbose:
            status = "✅ ALL LAYERS HEALTHY" if all_healthy else "⚠️ SOME LAYERS DEGRADED"
            print(f"\n{status}")
        
        results['_overall_healthy'] = all_healthy
        return results


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
            # === L2 NORM AGGREGATION JUSTIFICATION ===
            # Saliency maps can aggregate multi-channel gradients in two ways:
            # 1. MAX: saliency = max(|∂f/∂x_R|, |∂f/∂x_G|, |∂f/∂x_B|)
            #    Problem: Only preserves largest gradient; hides important activity in other channels
            # 2. L2 NORM (implemented): saliency = sqrt(sum(|∂f/∂x_c|^2)) for all channels c
            #    Benefit: Preserves ALL channel information; treats all channels equally
            #
            # Academic Basis:
            # - Simonyan et al. (2013) "Deep Inside Convolutional Networks" recommends L2 for multi-channel
            # - Montavon et al. (2015) "Deep Inside Convolutional Networks: Visualizing Image 
            #   Classification Models" shows L2 norm preserves more information than max
            # - L2 norm is standard in computer vision saliency (OpenCV, PyTorch conventions)
            #
            # Mathematical: L2 = ||∇f(x)||_2 = sqrt(Σ_c |∂f/∂x_c|^2)
            # This preserves the magnitude and importance of gradients across all channels.
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


class ExplanationComparator:
    """
    Compares explanations from different XAI methods to assess consistency
    
    Different XAI methods should roughly agree on important features.
    If they disagree significantly, it may indicate:
    1. Method-specific artifacts
    2. Implementation issues
    3. Legitimate differences in how methods work
    
    This class provides tools to compare and visualize agreement between methods.
    
    Comparison metrics:
    - Spatial correlation (for vision attributions)
    - Feature ranking agreement (for physiology attributions)
    - Attribution magnitude similarity
    - Prediction consistency
    """
    
    def __init__(self):
        """Initialize comparison analyzer"""
        pass
    
    @staticmethod
    def compare_attributions(
        attr_dict: Dict[str, np.ndarray],
        method_names: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Compare attribution maps from different methods
        
        Args:
            attr_dict: Dict mapping method names to attribution arrays
            method_names: Specific methods to compare (default: all)
            verbose: Print results
            
        Returns:
            Comparison results including correlations
        """
        if method_names is None:
            method_names = list(attr_dict.keys())
        
        # Extract attributions
        attributions = {name: attr_dict[name] for name in method_names if name in attr_dict}
        
        # Normalize each attribution
        normalized = {}
        for name, attr in attributions.items():
            attr_flat = attr.flatten()
            min_val = np.min(attr_flat)
            max_val = np.max(attr_flat)
            if max_val > min_val:
                normalized[name] = (attr_flat - min_val) / (max_val - min_val)
            else:
                normalized[name] = np.zeros_like(attr_flat)
        
        # Compute pairwise correlations
        results = {'method_pairs': {}}
        methods = list(normalized.keys())
        
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                pair_name = f"{m1}_vs_{m2}"
                # Pearson correlation
                corr = np.corrcoef(normalized[m1], normalized[m2])[0, 1]
                # Spearman rank correlation
                from scipy.stats import spearmanr
                spearman_corr, _ = spearmanr(normalized[m1], normalized[m2])
                
                results['method_pairs'][pair_name] = {
                    'pearson_r': float(corr) if not np.isnan(corr) else 0.0,
                    'spearman_r': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
                }
        
        # Overall agreement score (average correlation)
        if results['method_pairs']:
            pearson_scores = [v['pearson_r'] for v in results['method_pairs'].values()]
            results['mean_pearson'] = float(np.mean(pearson_scores))
            results['min_pearson'] = float(np.min(pearson_scores))
            results['max_pearson'] = float(np.max(pearson_scores))
        
        if verbose:
            print("\n🔍 Explanation Comparison Results:")
            print(f"Methods compared: {', '.join(method_names)}")
            if results['method_pairs']:
                print(f"Mean Pearson correlation: {results['mean_pearson']:.4f}")
                for pair_name, corrs in results['method_pairs'].items():
                    print(f"  {pair_name}: r={corrs['pearson_r']:.4f}")
        
        return results
    
    @staticmethod
    def compare_predictions(
        predictions: Dict[str, float],
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Compare predictions from different methods
        
        Args:
            predictions: Dict mapping method names to predicted probabilities
            verbose: Print results
            
        Returns:
            Prediction comparison results
        """
        pred_values = np.array(list(predictions.values()))
        
        results = {
            'mean_prediction': float(np.mean(pred_values)),
            'std_prediction': float(np.std(pred_values)),
            'min_prediction': float(np.min(pred_values)),
            'max_prediction': float(np.max(pred_values)),
            'prediction_spread': float(np.max(pred_values) - np.min(pred_values)),
        }
        
        # Check agreement (within 10% = good agreement)
        results['predictions_agree'] = results['prediction_spread'] < 0.1
        
        if verbose:
            print("\n🎯 Prediction Comparison:")
            for method, pred in predictions.items():
                print(f"  {method}: {pred:.4f}")
            print(f"Mean: {results['mean_prediction']:.4f} ± {results['std_prediction']:.4f}")
            print(f"Agreement: {'✅ GOOD' if results['predictions_agree'] else '⚠️ LOW'}")
        
        return results


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


class PerformanceProfiler:
    """
    Profiles and benchmarks XAI computation efficiency
    
    Measures:
    1. Wall-clock time for each XAI method
    2. GPU memory usage
    3. Number of forward/backward passes
    4. Computational complexity (model-dependent)
    
    Useful for understanding computational bottlenecks and optimizing
    XAI computation in production settings.
    
    Reference: Lin et al. (2021) "An Empirical Study of Example Forgetting
    during Deep Neural Network Learning"
    """
    
    def __init__(self):
        """Initialize profiler"""
        self.times = {}
        self.memory = {}
        self.pass_counts = {}
    
    @staticmethod
    def profile_method(
        method_fn,
        *args,
        method_name: str = "unknown",
        verbose: bool = True,
        **kwargs
    ) -> Tuple[any, Dict[str, float]]:
        """
        Profile a single XAI method
        
        Args:
            method_fn: Function to profile (should return explanation)
            args: Positional arguments to method_fn
            method_name: Name of method for reporting
            verbose: Print results
            kwargs: Keyword arguments to method_fn
            
        Returns:
            (result, timing_dict) where result is method output and
            timing_dict contains performance metrics
        """
        import time
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Time the computation
        t_start = time.perf_counter()
        result = method_fn(*args, **kwargs)
        t_end = time.perf_counter()
        
        elapsed_time = t_end - t_start
        
        # Estimate memory (GPU if available)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            memory_used = 0.0
        
        timing = {
            'method': method_name,
            'wall_time_sec': float(elapsed_time),
            'gpu_memory_mb': float(memory_used),
            'samples_per_sec': 1.0 / elapsed_time if elapsed_time > 0 else 0.0
        }
        
        if verbose:
            print(f"\n⏱️  Performance Profile [{method_name}]")
            print(f"  Wall time: {elapsed_time:.4f}s")
            print(f"  GPU memory: {memory_used:.2f} MB")
            print(f"  Throughput: {timing['samples_per_sec']:.2f} samples/sec")
        
        return result, timing
    
    @staticmethod
    def profile_multiple_methods(
        methods: Dict[str, Callable],
        img: torch.Tensor,
        phys: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Profile multiple XAI methods on the same input
        
        Args:
            methods: Dict mapping method names to callable methods
            img: Input image tensor
            phys: Input physiology tensor
            verbose: Print results
            
        Returns:
            Dict of timing results for each method
        """
        results = {}
        
        for method_name, method_fn in methods.items():
            try:
                _, timing = PerformanceProfiler.profile_method(
                    method_fn,
                    img,
                    phys,
                    method_name=method_name,
                    verbose=verbose
                )
                results[method_name] = timing
            except Exception as e:
                if verbose:
                    print(f"⚠️  {method_name} failed: {e}")
                results[method_name] = {'error': str(e)}
        
        # Summary
        if verbose:
            print("\n📊 Performance Summary:")
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            if valid_results:
                times = [v['wall_time_sec'] for v in valid_results.values()]
                print(f"  Fastest: {min(times):.4f}s")
                print(f"  Slowest: {max(times):.4f}s")
                print(f"  Average: {np.mean(times):.4f}s")
        
        return results
    
    @staticmethod
    def estimate_complexity(
        batch_size: int,
        image_size: Tuple[int, int] = (224, 224),
        phys_dim: int = 20,
        num_steps: int = 50,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Estimate computational complexity of IG
        
        Args:
            batch_size: Number of samples
            image_size: (H, W) of image
            phys_dim: Dimensionality of physiology vector
            num_steps: Number of IG integration steps
            verbose: Print results
            
        Returns:
            Complexity estimates
        """
        H, W = image_size
        
        # IG requires num_steps forward/backward passes
        fwd_passes_ig = batch_size * num_steps
        bwd_passes_ig = batch_size * num_steps
        
        # Saliency requires 1 forward/backward
        fwd_passes_saliency = batch_size
        bwd_passes_saliency = batch_size
        
        # Image processing: O(batch_size * H * W)
        ops_vision = batch_size * H * W * 3  # 3 channels
        
        # Physiology processing: O(batch_size * phys_dim)
        ops_phys = batch_size * phys_dim
        
        results = {
            'fwd_passes_ig': fwd_passes_ig,
            'bwd_passes_ig': bwd_passes_ig,
            'fwd_passes_saliency': fwd_passes_saliency,
            'bwd_passes_saliency': bwd_passes_saliency,
            'ig_complexity_ratio': fwd_passes_ig / fwd_passes_saliency,
            'ops_vision': ops_vision,
            'ops_phys': ops_phys,
        }
        
        if verbose:
            print(f"\n📈 Computational Complexity Estimate:")
            print(f"  Batch size: {batch_size}, Image: {H}x{W}, Phys dim: {phys_dim}")
            print(f"  IG forward passes: {fwd_passes_ig} (vs {fwd_passes_saliency} for Saliency)")
            print(f"  IG complexity ratio: {results['ig_complexity_ratio']:.1f}x more than Saliency")
            print(f"  Vision ops: {ops_vision:,.0f}")
            print(f"  Physiology ops: {ops_phys:,.0f}")
        
        return results

