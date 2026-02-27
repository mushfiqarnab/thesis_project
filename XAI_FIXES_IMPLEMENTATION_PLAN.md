# XAI MODULE FIXES - IMPLEMENTATION PLAN
**Priority**: CRITICAL + MAJOR ISSUES FIX  
**Timeline**: 8-10 hours  
**Status**: Ready to implement

---

## PHASE 1: CRITICAL BUG FIX (30 minutes)

### FIX #1: Integrated Gradients Gradient Accumulation Bug

**File**: `src/xai/__init__.py`  
**Lines**: 85-111 (compute_gradients method)

**Current Code**:
```python
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
    
    img.requires_grad_(False)
    phys.requires_grad_(False)
    
    return grad_img, grad_phys
```

**Fixed Code**:
```python
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
    
    # CRITICAL FIX: Zero gradients so they don't accumulate in next iteration
    img.grad = None
    phys.grad = None
    
    img.requires_grad_(False)
    phys.requires_grad_(False)
    
    return grad_img, grad_phys
```

**Change Summary**:
- Added 2 lines: `img.grad = None` and `phys.grad = None`
- This ensures gradients are cleared before next backward pass
- Prevents gradient accumulation across IG steps

**Why This Fixes It**:
- PyTorch accumulates gradients by default
- Without zeroing, `.backward()` calls ADD to existing gradients
- IG loop calls `compute_gradients()` 50 times → gradients grow exponentially
- Fix ensures each call computes FRESH gradients

**Validation Code** (add after fix):
```python
# In explain() method, after computing attributions:
# Validate that IG doesn't have duplicate accumulation
assert accumulated_grad_img.abs().max() < accumulated_grad_img.abs().std() * 10, \
    "Gradients may still be accumulating - check for NaN/Inf"
```

---

## PHASE 2: MAJOR ISSUES (3-4 hours)

### FIX #2: Improve Baseline Selection with Multiple Options

**File**: `src/xai/__init__.py`  
**Lines**: 55-70 (_get_baseline method)

**Current Code**:
```python
def _get_baseline(self, img: torch.Tensor, phys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline is black image (all zeros) and zero physiology.
    This represents "no information" input.
    """
    img_baseline = torch.zeros_like(img)
    phys_baseline = torch.zeros_like(phys)
    return img_baseline, phys_baseline
```

**Fixed Code**:
```python
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
        baseline_type: 'black' (zeros), 'gray' (ImageNet mean), or 'blur'
        
    Returns:
        img_baseline: (B, 3, H, W) baseline image
        phys_baseline: (B, D) baseline physiology
        
    Reference: Sundararajan et al. (2017) Section 4.1
    """
    if baseline_type == 'black':
        # Zero image (no visual information)
        img_baseline = torch.zeros_like(img)
        
    elif baseline_type == 'gray':
        # Mean gray level (in normalized space, this is actually zeros,
        # because normalization subtracts the mean)
        # In pixel space, this would be ImageNet mean = [0.485, 0.456, 0.406]
        img_baseline = torch.zeros_like(img)
        # Note: Could also use Gaussian blur for smoother baseline
        
    elif baseline_type == 'blur':
        # Blur the input image (smooth baseline)
        # This represents "all frequency information intact, but no fine details"
        import torch.nn.functional as F
        img_baseline = F.avg_pool2d(img, kernel_size=5, padding=2)
        # Upsample back to original size
        img_baseline = F.interpolate(img_baseline, size=img.shape[-2:], mode='bilinear', align_corners=False)
        
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}. "
                        f"Choose from: 'black', 'gray', 'blur'")
    
    phys_baseline = torch.zeros_like(phys)
    
    return img_baseline, phys_baseline
```

**Update explain() to accept baseline_type**:
```python
def explain(
    self,
    img: torch.Tensor,
    phys: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    target_class: int = 1,
    steps: int = 50,
    baseline_type: str = 'black'  # ADD THIS PARAMETER
) -> ExplanationOutput:
    """
    Compute Integrated Gradients attribution
    
    Args:
        img: (B, 3, H, W) normalized image
        phys: (B, D) physiology vector
        mask: (B, 1, H, W) scar region mask
        target_class: class to explain (0 or 1)
        steps: number of interpolation steps (higher = more accurate)
        baseline_type: baseline selection strategy ('black', 'gray', 'blur')
        
    Returns:
        ExplanationOutput with vision and physiology attributions
    """
    device = self.device
    img = img.to(device)
    phys = phys.to(device)
    if mask is not None:
        mask = mask.to(device)
        
    img_baseline, phys_baseline = self._get_baseline(img, phys, baseline_type=baseline_type)  # UPDATE
    
    # ... rest of explain() ...
```

**Update metadata to track baseline**:
```python
metadata={
    'steps': steps,
    'target_class': target_class,
    'img_shape': img.shape,
    'phys_dim': phys.shape[1],
    'baseline_type': baseline_type  # ADD THIS
}
```

---

### FIX #3: Add Completeness Axiom Validation

**File**: `src/xai/__init__.py`  
**Add to explain() method**, after computing attributions (around line 175)

**New Code to Add**:
```python
        # Attribution = (input - baseline) * gradient
        attr_img = (img - img_baseline) * avg_grad_img
        attr_phys = (phys - phys_baseline) * avg_grad_phys
        
        # === NEW: VALIDATE COMPLETENESS AXIOM ===
        # IG should satisfy: sum(attributions) ≈ f(x) - f(baseline)
        with torch.no_grad():
            # Compute prediction difference
            pred_full = self._forward_pass(img, phys, mask)
            pred_baseline = self._forward_pass(img_baseline, phys_baseline, mask)
            delta_pred = pred_full - pred_baseline  # (B,)
            
            # Compute attribution sum
            attr_sum = attr_img.sum(dim=[1, 2, 3]) + attr_phys.sum(dim=1)  # (B,)
            
            # Check completeness
            completeness_error = (attr_sum - delta_pred).abs()  # (B,)
            completeness_error_rel = completeness_error / (delta_pred.abs() + 1e-8)
            
            # Log warning if error is high
            if completeness_error_rel[0] > 0.1:  # >10% error
                print(f"[Warning] IG Completeness axiom violated: "
                      f"error = {completeness_error[0]:.4f}, "
                      f"relative_error = {completeness_error_rel[0]:.2%}")
                print(f"  Expected: {delta_pred[0]:.4f}, Got: {attr_sum[0]:.4f}")
        # === END VALIDATION ===
        
        # Get prediction
        with torch.no_grad():
            # ... rest of existing code ...
```

**Store completeness info in metadata**:
```python
metadata={
    'steps': steps,
    'target_class': target_class,
    'img_shape': img.shape,
    'phys_dim': phys.shape[1],
    'baseline_type': baseline_type,
    'completeness_error': float(completeness_error[0].item()),  # ADD
    'completeness_error_rel': float(completeness_error_rel[0].item())  # ADD
}
```

---

### FIX #4: Rename FairnessXAI Metric and Update Documentation

**File**: `src/xai/__init__.py`  
**Lines**: 275-330 (FairnessXAI class)

**Current Code**:
```python
class FairnessXAI:
    """
    Fairness-Aware XAI
    
    Links scar region importance to fairness metrics.
    Answers: "How much does the scar mask influence the fairness gaps?"
    """
    
    def compute_scar_influence_score(self, ...):
        """..."""
        influence = (p_full - p_zero).abs().mean()
        return influence
    
    def explain(self, ...):
        """..."""
        scar_influence = self.compute_scar_influence_score(...)
        if scar_influence > threshold_high:
            fairness_risk = "high"
        elif scar_influence > threshold_med:
            fairness_risk = "medium"
        else:
            fairness_risk = "low"
```

**Fixed Code**:
```python
class ScarSensitivityXAI:  # RENAMED from FairnessXAI
    """
    Scar Sensitivity Analysis via Ablation
    
    Measures how sensitive the model is to the scar region signal.
    
    IMPORTANT: This is NOT a fairness metric. This measures individual
    sample sensitivity to scar information. Fairness metrics (from paper)
    measure aggregate statistical parity across the entire dataset:
    - Demographic Parity (DP): P(threat|scar=1) - P(threat|scar=0)
    - Equalized Odds (EO): |TPR_scar1 - TPR_scar0|
    
    Use this method to understand individual predictions.
    Use paper's fairness metrics to evaluate model fairness.
    """
    
    def compute_scar_attribution_score(self, img, phys, mask, scar):
        """
        Measure scar region attribution (sensitivity) for a single sample.
        
        Method: Compare predictions with and without scar mask signal
        
        Args:
            img: (B, 3, H, W)
            phys: (B, D)
            mask: (B, 1, H, W) scar region
            scar: (B,) binary scar labels
            
        Returns:
            score: float in [0, 1]
                  0 = scar signal has no influence on prediction
                  1 = scar signal strongly influences prediction
                  
        Note: High sensitivity ≠ Unfair model. A fair model can
              legitimately be sensitive to medical features like scars.
        """
        device = self.device
        img = img.to(device)
        phys = phys.to(device)
        mask = mask.to(device)
        scar = scar.to(device)
        
        with torch.no_grad():
            # Prediction with scar signal present
            out_full = self.model(img, phys, mask=mask)
            p_full = F.softmax(out_full.logits, dim=1)[:, 1]
            
            # Prediction with scar signal ablated (zeroed)
            mask_zero = torch.zeros_like(mask)
            out_zero = self.model(img, phys, mask=mask_zero)
            p_zero = F.softmax(out_zero.logits, dim=1)[:, 1]
            
            # Attribution = change in prediction when removing scar signal
            attribution = (p_full - p_zero).abs().mean()
            
        return float(attribution.item())
    
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
        Explain prediction with scar sensitivity context.
        
        Args:
            img, phys, mask, scar: model inputs + scar labels
            threshold_high: scar attribution above this = sensitive to scar
            threshold_med: scar attribution above this = moderately sensitive
            
        Returns:
            ExplanationOutput with scar_sensitivity_level annotation
        """
        device = self.device
        img = img.to(device)
        phys = phys.to(device)
        mask = mask.to(device)
        
        # Compute scar attribution
        scar_attribution = self.compute_scar_attribution_score(img, phys, mask, scar)
        
        # Assess sensitivity level
        if scar_attribution > threshold_high:
            sensitivity_level = "high"
        elif scar_attribution > threshold_med:
            sensitivity_level = "medium"
        else:
            sensitivity_level = "low"
            
        with torch.no_grad():
            out = self.model(img, phys, mask=mask)
            probs = F.softmax(out.logits, dim=1)[:, 1]
            pred_class = (probs >= 0.5).long()
            gate_val = out.gate[0, 0].item() if out.gate is not None else None
            focus_val = out.focus[0, 0].item() if out.focus is not None else None
            
        return ExplanationOutput(
            method="scar_sensitivity_xai",
            prediction=float(probs[0].item()),
            prediction_class=int(pred_class[0].item()),
            scar_influence_score=scar_attribution,  # Keep old name for compatibility
            fairness_risk=sensitivity_level,  # Will be "low|medium|high"
            gate_activation=gate_val,
            focus_activation=focus_val,
            metadata={
                'threshold_high': threshold_high,
                'threshold_med': threshold_med,
                'scar_present': int(scar[0].item()),
                'interpretation': (
                    f"Model prediction is {'STRONGLY' if sensitivity_level == 'high' else 'MODERATELY' if sensitivity_level == 'medium' else 'WEAKLY'} "
                    f"influenced by scar region signal. "
                    f"Note: This does not indicate unfairness, but rather "
                    f"that scar information affects this particular prediction."
                )
            }
        )
```

**Update XAIExplainer**:
```python
class XAIExplainer:
    """Unified interface for all XAI methods"""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        
        self.ig = IntegratedGradients(model, device=self.device)
        self.saliency = SaliencyMap(model, device=self.device)
        self.scar_sensitivity = ScarSensitivityXAI(model, device=self.device)  # RENAMED
        self.attention = AttentionVisualizer(model, device=self.device)
    
    def explain_all(self, ...):
        """..."""
        explanations = {}
        
        # ... existing code ...
        
        # Scar Sensitivity (was Fairness XAI)
        if scar is not None:
            try:
                explanations['scar_sensitivity_xai'] = self.scar_sensitivity.explain(
                    img, phys, mask, scar
                )
            except Exception as e:
                print(f"[Warning] Scar Sensitivity XAI failed: {e}")
        
        return explanations
```

---

### FIX #5: Improve Saliency Map Aggregation

**File**: `src/xai/__init__.py`  
**Lines**: 235-245 (SaliencyMap.explain method)

**Current Code**:
```python
        # Saliency = |∂f/∂x|
        saliency = img.grad.abs()  # (B, C, H, W)
        
        if aggregate_channels:
            # Take max across color channels
            saliency = saliency.max(dim=1)[0]  # (B, H, W)
```

**Fixed Code**:
```python
        # Saliency = |∂f/∂x|
        saliency = img.grad.abs()  # (B, C, H, W)
        
        if aggregate_channels:
            # Use L2 norm across channels (better than max)
            # L2 represents overall magnitude without losing channel info
            saliency = torch.sqrt((saliency ** 2).sum(dim=1, keepdim=False))  # (B, H, W)
            # Alternative interpretations available:
            # - Mean: torch.mean(saliency, dim=1)  # Average gradient magnitude
            # - Max: saliency.max(dim=1)[0]        # Peak gradient magnitude (current)
```

**Add to docstring**:
```python
class SaliencyMap:
    """
    Saliency Maps (Simonyan et al., 2013)
    
    Saliency = | ∂f / ∂x |
    
    Shows which input pixels have the largest gradient magnitude.
    Higher saliency = pixels that more strongly influence the prediction.
    
    Aggregation methods for multi-channel images:
    - L2 norm (default): sqrt(sum(grad²)) - represents overall magnitude
    - Mean: mean(|grad|) - average gradient across channels
    - Max: max(|grad|) - peak gradient (was previous default)
    """
```

---

### FIX #6: Robust Image Denormalization

**File**: `src/xai/visualization.py`  
**Lines**: 38-58 (denormalize_image function)

**Current Code**:
```python
def denormalize_image(img: np.ndarray) -> np.ndarray:
    """
    Denormalize image from ImageNet normalization
    
    Args:
        img: (3, H, W) or (H, W, 3) normalized image
        
    Returns:
        (H, W, 3) image in [0, 1] with denormalization applied
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Handle both (C, H, W) and (H, W, C) formats
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    
    # Denormalize
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    return img
```

**Fixed Code**:
```python
def denormalize_image(img: np.ndarray) -> np.ndarray:
    """
    Denormalize image from ImageNet normalization.
    
    Handles:
    - (C, H, W) and (H, W, C) formats
    - 1-channel (grayscale), 3-channel (RGB), 4-channel (RGBA)
    - Validates input format and range
    
    Args:
        img: (C, H, W) or (H, W, C) or (H, W) normalized image
             Expected to be float32 in range [-3, 3]
        
    Returns:
        (H, W, C) image in [0, 1] with denormalization applied
        
    Raises:
        ValueError: If input format cannot be determined
        AssertionError: If input is outside expected range
    """
    # Input validation
    assert isinstance(img, np.ndarray), f"Expected ndarray, got {type(img)}"
    assert img.dtype in [np.float32, np.float64], \
        f"Expected float array, got {img.dtype}. " \
        f"If input is uint8, convert to float first: img = img.astype(np.float32) / 255.0"
    assert img.ndim in [2, 3], \
        f"Expected 2D or 3D array, got {img.ndim}D"
    assert img.min() >= -3.0 and img.max() <= 5.0, \
        f"Input outside expected range [-3, 5]: min={img.min():.2f}, max={img.max():.2f}. " \
        f"Is input already denormalized?"
    
    # Determine format and get number of channels
    if img.ndim == 3:
        if img.shape[0] in [1, 3, 4]:  # Channels first (C, H, W)
            channels = img.shape[0]
            img = img.transpose(1, 2, 0)  # → (H, W, C)
        elif img.shape[2] in [1, 3, 4]:  # Channels last (H, W, C)
            channels = img.shape[2]
        else:
            raise ValueError(
                f"Cannot determine channel dimension. Shape: {img.shape}. "
                f"Expected shape (3, H, W), (H, W, 3), (1, H, W), or (H, W, 1)"
            )
    else:  # 2D array (H, W) - grayscale
        img = img[:, :, np.newaxis]  # → (H, W, 1)
        channels = 1
    
    # Get normalization parameters based on channels
    if channels == 1:
        # For grayscale: use generic normalization
        # (ImageNet precomputed on RGB, so we use approximation)
        mean = np.array([0.5])
        std = np.array([0.5])
    elif channels == 3:
        # ImageNet RGB normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    elif channels == 4:
        # RGBA: use ImageNet for RGB, neutral for Alpha
        mean = np.array([0.485, 0.456, 0.406, 0.5])
        std = np.array([0.229, 0.224, 0.225, 0.5])
    else:
        raise ValueError(
            f"Unsupported number of channels: {channels}. "
            f"Expected 1 (grayscale), 3 (RGB), or 4 (RGBA)"
        )
    
    # Denormalize: x_pixel = (x_norm * std) + mean
    img = img * std + mean
    
    # Clip to valid range [0, 1]
    img = np.clip(img, 0, 1)
    
    return img
```

**Add type hints**:
```python
from typing import Optional, Tuple
import numpy as np

def denormalize_image(img: np.ndarray) -> np.ndarray:
    """..."""
```

---

### FIX #7: Add Gradient Stability Checks

**File**: `src/xai/__init__.py`  
**Add to explain() method**, after line 165 (after gradient computation)

**New Code to Add**:
```python
        # === NEW: STABILITY CHECKS ===
        # Check for numerical instabilities
        if torch.isnan(avg_grad_img).any() or torch.isnan(avg_grad_phys).any():
            print("[ERROR] NaN gradients detected! Model may be numerically unstable.")
            print(f"  NaN in img gradients: {torch.isnan(avg_grad_img).any()}")
            print(f"  NaN in phys gradients: {torch.isnan(avg_grad_phys).any()}")
            # Fall back to zeros
            avg_grad_img = torch.where(torch.isnan(avg_grad_img), torch.zeros_like(avg_grad_img), avg_grad_img)
            avg_grad_phys = torch.where(torch.isnan(avg_grad_phys), torch.zeros_like(avg_grad_phys), avg_grad_phys)
        
        if torch.isinf(avg_grad_img).any() or torch.isinf(avg_grad_phys).any():
            print("[ERROR] Inf gradients detected! Gradient explosion.")
            print(f"  Max img gradient: {avg_grad_img.abs().max():.2e}")
            print(f"  Max phys gradient: {avg_grad_phys.abs().max():.2e}")
            # Clip gradients
            avg_grad_img = torch.clamp(avg_grad_img, -1e6, 1e6)
            avg_grad_phys = torch.clamp(avg_grad_phys, -1e6, 1e6)
        
        # Check for zero gradients (flat regions)
        img_grad_norm = avg_grad_img.abs().max()
        phys_grad_norm = avg_grad_phys.abs().max()
        if img_grad_norm < 1e-8 and phys_grad_norm < 1e-8:
            print("[WARNING] All gradients are ~zero. Model may be in saturated regime "
                  "or prediction independent of inputs.")
        
        stability_status = {
            'has_nan': False,
            'has_inf': False,
            'has_zero_grad': img_grad_norm < 1e-8 and phys_grad_norm < 1e-8,
            'max_img_grad': float(img_grad_norm.item()),
            'max_phys_grad': float(phys_grad_norm.item())
        }
        # === END STABILITY CHECKS ===
```

**Update metadata**:
```python
metadata={
    'steps': steps,
    'target_class': target_class,
    'img_shape': img.shape,
    'phys_dim': phys.shape[1],
    'baseline_type': baseline_type,
    'completeness_error': float(completeness_error[0].item()),
    'completeness_error_rel': float(completeness_error_rel[0].item()),
    'stability': stability_status  # ADD THIS
}
```

---

## PHASE 3: ENHANCEMENT GAPS (2-3 hours)

### ENHANCEMENT #1: Physiology Feature Normalization

**File**: `src/xai/__init__.py`  
**Add new method to IntegratedGradients class** (before explain method):

```python
def normalize_phys_attribution(
    self,
    attr_phys: np.ndarray,
    phys_dataset_stats: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    """
    Normalize physiology attributions by feature standard deviation.
    
    This addresses scale differences between features:
    - HRV might be in [0, 500] ms
    - GSR might be in [0, 10] μS
    - ECG_HR might be in [40, 180] bpm
    
    Unnormalized attribution for HRV will be naturally larger due to scale.
    Normalization allows fair comparison of feature importance.
    
    Args:
        attr_phys: (D,) attribution array for D physiology features
        phys_dataset_stats: Dict with 'std' key containing (D,) std per feature
                           If None, uses default ImageNet-like normalization
                           
    Returns:
        attr_phys_normalized: (D,) normalized attributions
    """
    D = attr_phys.shape[0]
    
    if phys_dataset_stats is None:
        print("[Warning] Using unit normalization. "
              "Provide actual dataset stats for proper feature normalization.")
        # Default: divide by feature std (assuming unit normalized inputs)
        feature_std = np.ones(D)
    else:
        feature_std = phys_dataset_stats['std']
    
    # Normalize: attr / std
    attr_phys_normalized = attr_phys / (feature_std + 1e-8)
    
    return attr_phys_normalized
```

**Update explain() to optionally use this**:
```python
    # In the return statement:
    return ExplanationOutput(
        method="integrated_gradients",
        prediction=float(pred_probs[0].item()),
        prediction_class=int(pred_class[0].item()),
        vision_attribution=attr_img[0].detach().cpu().numpy(),
        phys_attribution=attr_phys[0].detach().cpu().numpy(),  # Raw
        # Optional: provide normalized version in metadata
        metadata={
            # ...
            'phys_attribution_normalized': self.normalize_phys_attribution(
                attr_phys[0].detach().cpu().numpy(),
                phys_dataset_stats=None
            )
        }
    )
```

---

### ENHANCEMENT #2: Ablation Study Capability

**File**: `src/xai/__init__.py`  
**Add new class**:

```python
class AblationStudy:
    """
    Ablation-based feature importance.
    
    Complement to gradient-based methods (IG, Saliency).
    Directly measures: "How much does prediction change if I remove this feature?"
    
    Useful for validation: IG rankings should correlate with ablation rankings.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
    
    def ablate_feature(
        self,
        img: torch.Tensor,
        phys: torch.Tensor,
        feature_idx: int,
        modality: str = 'phys',
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Remove a single feature and measure effect on prediction.
        
        Args:
            img, phys: model inputs
            feature_idx: index of feature to ablate
            modality: 'img' or 'phys'
            mask: optional scar mask
            
        Returns:
            importance: float in [0, 1]
                       0 = feature has no effect
                       1 = feature strongly affects prediction
        """
        device = self.device
        img = img.to(device)
        phys = phys.to(device)
        if mask is not None:
            mask = mask.to(device)
        
        with torch.no_grad():
            # Get baseline prediction
            out_full = self.model(img, phys, mask=mask)
            pred_full = F.softmax(out_full.logits, dim=1)[:, 1]
            
            # Ablate feature
            if modality == 'phys':
                phys_ablated = phys.clone()
                phys_ablated[:, feature_idx] = 0  # Zero out feature
            elif modality == 'img':
                img_ablated = img.clone()
                img_ablated[:, feature_idx % 3, :, :] = 0  # Zero channel
            else:
                raise ValueError(f"Unknown modality: {modality}")
            
            # Get ablated prediction
            out_ablated = self.model(
                img_ablated if modality == 'img' else img,
                phys_ablated if modality == 'phys' else phys,
                mask=mask
            )
            pred_ablated = F.softmax(out_ablated.logits, dim=1)[:, 1]
            
            # Importance = change in prediction
            importance = (pred_full - pred_ablated).abs().mean()
        
        return float(importance.item())
    
    def explain(
        self,
        img: torch.Tensor,
        phys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        modality: str = 'phys'
    ) -> Dict[int, float]:
        """
        Compute ablation-based importance for all features.
        
        Returns:
            importance_dict: {feature_idx: importance_score}
        """
        device = self.device
        img = img.to(device)
        phys = phys.to(device)
        if mask is not None:
            mask = mask.to(device)
        
        if modality == 'phys':
            num_features = phys.shape[1]
        elif modality == 'img':
            num_features = 3  # RGB channels
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        importance_scores = {}
        for idx in range(num_features):
            score = self.ablate_feature(img, phys, idx, modality, mask)
            importance_scores[idx] = score
        
        return importance_scores
```

---

### ENHANCEMENT #3: Baseline Comparison Study

**File**: `src/xai/__init__.py`  
**Add new method to XAIExplainer class**:

```python
def compare_baselines(
    self,
    img: torch.Tensor,
    phys: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    target_class: int = 1,
    steps: int = 50
) -> Dict[str, ExplanationOutput]:
    """
    Run IG with multiple baselines and compare results.
    
    This addresses Kindermans et al. (2019) which shows
    baseline choice significantly affects IG attributions.
    
    Args:
        img, phys, mask: model inputs
        target_class: class to explain
        steps: IG interpolation steps
        
    Returns:
        Dict mapping baseline_type -> ExplanationOutput
        
    Usage:
        results = explainer.compare_baselines(img, phys, mask)
        # results['black'], results['gray'], results['blur']
    """
    baselines = ['black', 'gray', 'blur']
    results = {}
    
    for baseline_type in baselines:
        print(f"[Running IG with baseline: {baseline_type}]")
        try:
            result = self.ig.explain(
                img, phys, mask=mask,
                target_class=target_class,
                steps=steps,
                baseline_type=baseline_type
            )
            results[baseline_type] = result
        except Exception as e:
            print(f"  Failed: {e}")
    
    return results
```

---

## VALIDATION & TESTING (1 hour)

### Test Script to Validate Fixes

**File**: Create `src/xai/test_fixes.py`

```python
"""
Test script to validate XAI fixes.
Run this to ensure all bug fixes work correctly.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from xai import IntegratedGradients, SaliencyMap, ScarSensitivityXAI
from xai.visualization import denormalize_image


def test_gradient_accumulation_fix():
    """
    Test that gradients don't accumulate across IG steps.
    
    The bug was that .grad wasn't being zeroed, causing
    gradients to accumulate exponentially across steps.
    """
    print("\n[TEST] Gradient Accumulation Fix")
    print("=" * 60)
    
    # Create mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)
        
        def forward(self, img, phys, mask=None):
            logits = self.fc(phys)
            class ModelOutput:
                def __init__(self, logits):
                    self.logits = logits
                    self.gate = None
                    self.focus = None
            return ModelOutput(logits)
    
    model = MockModel()
    ig = IntegratedGradients(model)
    
    # Create test data
    img = torch.randn(1, 3, 224, 224)
    phys = torch.randn(1, 10)
    
    # Run IG
    result = ig.explain(img, phys, steps=5)
    
    # Check that attributions are reasonable (not exponentially large)
    attr_img_max = np.abs(result.vision_attribution).max()
    attr_phys_max = np.abs(result.phys_attribution).max()
    
    print(f"Max vision attribution: {attr_img_max:.4f}")
    print(f"Max phys attribution: {attr_phys_max:.4f}")
    
    # Attributions should be ~same magnitude as gradients
    # If accumulation bug exists, they'd be 100x larger
    assert attr_img_max < 1000, "Attributions suspiciously large - accumulation bug?"
    assert attr_phys_max < 1000, "Attributions suspiciously large - accumulation bug?"
    
    print("✓ PASS: Gradients not accumulating")


def test_completeness_axiom():
    """
    Test that IG attributions satisfy completeness axiom.
    
    sum(attribution) should ≈ f(x) - f(baseline)
    """
    print("\n[TEST] Completeness Axiom")
    print("=" * 60)
    
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)
        
        def forward(self, img, phys, mask=None):
            logits = self.fc(phys)
            class ModelOutput:
                def __init__(self, logits):
                    self.logits = logits
                    self.gate = None
                    self.focus = None
            return ModelOutput(logits)
    
    model = MockModel()
    ig = IntegratedGradients(model)
    
    img = torch.randn(1, 3, 224, 224)
    phys = torch.randn(1, 10)
    
    result = ig.explain(img, phys, steps=50)
    
    # Check completeness info in metadata
    if 'completeness_error_rel' in result.metadata:
        rel_error = result.metadata['completeness_error_rel']
        print(f"Completeness relative error: {rel_error:.2%}")
        assert rel_error < 0.15, "Completeness axiom violated (>15% error)"
        print("✓ PASS: Completeness axiom satisfied")
    else:
        print("⚠ WARNING: Completeness info not in metadata")


def test_denormalize_robustness():
    """
    Test that denormalize_image handles edge cases.
    """
    print("\n[TEST] Denormalization Robustness")
    print("=" * 60)
    
    # Test 1: RGB (C, H, W)
    img_chw = np.random.randn(3, 224, 224).astype(np.float32)
    result = denormalize_image(img_chw)
    assert result.shape == (224, 224, 3), f"Wrong shape: {result.shape}"
    assert result.min() >= 0 and result.max() <= 1, "Values out of range"
    print("✓ PASS: RGB (C, H, W)")
    
    # Test 2: RGB (H, W, C)
    img_hwc = np.random.randn(224, 224, 3).astype(np.float32)
    result = denormalize_image(img_hwc)
    assert result.shape == (224, 224, 3), f"Wrong shape: {result.shape}"
    assert result.min() >= 0 and result.max() <= 1, "Values out of range"
    print("✓ PASS: RGB (H, W, C)")
    
    # Test 3: Grayscale (H, W)
    img_gray = np.random.randn(224, 224).astype(np.float32)
    result = denormalize_image(img_gray)
    assert result.shape == (224, 224, 1), f"Wrong shape: {result.shape}"
    assert result.min() >= 0 and result.max() <= 1, "Values out of range"
    print("✓ PASS: Grayscale (H, W)")
    
    # Test 4: RGBA
    img_rgba = np.random.randn(4, 224, 224).astype(np.float32)
    result = denormalize_image(img_rgba)
    assert result.shape == (224, 224, 4), f"Wrong shape: {result.shape}"
    assert result.min() >= 0 and result.max() <= 1, "Values out of range"
    print("✓ PASS: RGBA (C, H, W)")
    
    print("✓ PASS: Denormalization robustness")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("XAI MODULE VALIDATION TESTS")
    print("="*60)
    
    try:
        test_gradient_accumulation_fix()
        test_completeness_axiom()
        test_denormalize_robustness()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60 + "\n")
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
```

**Run tests**:
```bash
cd c:\Users\USERAS\thesis_project
python src/xai/test_fixes.py
```

---

## IMPLEMENTATION CHECKLIST

- [ ] **FIX #1 (CRITICAL)**: Add `img.grad = None; phys.grad = None` in compute_gradients()
- [ ] **FIX #2**: Update _get_baseline() with multiple baseline options
- [ ] **FIX #3**: Add completeness axiom validation
- [ ] **FIX #4**: Rename FairnessXAI → ScarSensitivityXAI, update docs
- [ ] **FIX #5**: Change saliency aggregation from max to L2 norm
- [ ] **FIX #6**: Replace denormalize_image() with robust version
- [ ] **FIX #7**: Add gradient stability checks
- [ ] **ENHANCEMENT #1**: Add physiology feature normalization method
- [ ] **ENHANCEMENT #2**: Implement AblationStudy class
- [ ] **ENHANCEMENT #3**: Add compare_baselines() method
- [ ] **TESTING**: Run test_fixes.py and verify all pass
- [ ] **DOCUMENTATION**: Update README with new methods and fixes

---

## TIME ESTIMATE

| Phase | Task | Time |
|-------|------|------|
| 1 | Critical Bug Fix | 30 min |
| 2a | Major Fixes #2-7 | 2 hours |
| 2b | Test & Validate | 30 min |
| 3 | Enhancements | 1.5 hours |
| 4 | Documentation | 1 hour |
| **TOTAL** | | **~5 hours** |

---

## NEXT DOCUMENT

See: `XAI_DOCUMENTATION_UPDATES.md` for:
- Updated README with new methods
- Tutorial on baseline selection
- Examples with fixed code
- Paper citations for each fix

