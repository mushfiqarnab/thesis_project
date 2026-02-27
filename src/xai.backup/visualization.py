"""
XAI Visualization Utilities

Provides functions to visualize:
- Saliency maps overlaid on original images
- Attribution heatmaps
- Gate/focus activation values
- Fairness risk indicators
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Optional, Tuple
import torch


def normalize_attribution(attr: np.ndarray) -> np.ndarray:
    """
    Normalize attribution map to [0, 1] for visualization
    
    Args:
        attr: (H, W) or (C, H, W) attribution array
        
    Returns:
        Normalized array in [0, 1]
    """
    if attr.ndim == 3:
        # For (C, H, W), compute magnitude across channels
        attr = np.sqrt((attr ** 2).sum(axis=0))
    
    attr_min = attr.min()
    attr_max = attr.max()
    
    if attr_max - attr_min < 1e-6:
        return np.ones_like(attr)
    
    return (attr - attr_min) / (attr_max - attr_min)


def denormalize_image(img: np.ndarray) -> np.ndarray:
    """
    Denormalize image from ImageNet normalization.
    
    Handles:
    - (C, H, W) and (H, W, C) formats
    - 1-channel (grayscale), 3-channel (RGB), 4-channel (RGBA)
    - Validates input format and range
    
    Args:
        img: (C, H, W) or (H, W, C) or (H, W) normalized image
             Expected to be float32/float64 in range [-3, 5]
        
    Returns:
        (H, W, C) image in [0, 1] with denormalization applied
        
    Raises:
        ValueError: If input format cannot be determined
        AssertionError: If input is outside expected range
    """
    # Input validation - FIX (2025-01-XX): Add robustness checks
    assert isinstance(img, np.ndarray), f"Expected ndarray, got {type(img)}"
    assert img.dtype in [np.float32, np.float64], \
        f"Expected float array, got {img.dtype}. " \
        f"If input is uint8, convert first: img = img.astype(np.float32) / 255.0"
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


def visualize_saliency(
    img: np.ndarray,
    saliency: np.ndarray,
    mask: Optional[np.ndarray] = None,
    title: str = "Saliency Map",
    cmap: str = "jet",
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize saliency map overlaid on image
    
    Args:
        img: (3, H, W) or (H, W, 3) image
        saliency: (H, W) saliency map
        mask: (H, W) binary mask (optional, for scar region)
        title: plot title
        cmap: colormap name
        save_path: if provided, save to this path
    """
    img = denormalize_image(img)
    saliency_norm = normalize_attribution(saliency)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Saliency map
    im = axes[1].imshow(saliency_norm, cmap=cmap)
    axes[1].set_title("Saliency Map")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay saliency on image
    axes[2].imshow(img)
    overlay = axes[2].imshow(saliency_norm, cmap=cmap, alpha=0.5)
    
    if mask is not None:
        # Mark mask region with contour
        from scipy import ndimage
        mask_binary = (mask > 0.5).astype(float)
        contours = ndimage.binary_dilation(mask_binary) - mask_binary
        axes[2].contour(contours, colors='white', linewidths=2)
    
    axes[2].set_title("Saliency Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_integrated_gradients(
    img: np.ndarray,
    vision_attr: np.ndarray,
    phys_attr: Optional[np.ndarray] = None,
    phys_names: Optional[list] = None,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize integrated gradients attribution
    
    Args:
        img: (3, H, W) or (H, W, 3) image
        vision_attr: (C, H, W) or (H, W) attribution
        phys_attr: (D,) physiology attribution (optional)
        phys_names: list of D physiology feature names
        mask: (H, W) scar mask (optional)
        save_path: if provided, save to this path
    """
    img = denormalize_image(img)
    
    # Compute attribution magnitude
    if vision_attr.ndim == 3:
        attr_magnitude = np.sqrt((vision_attr ** 2).sum(axis=0))
    else:
        attr_magnitude = vision_attr
    
    attr_norm = normalize_attribution(attr_magnitude)
    
    n_plots = 2 + (1 if phys_attr is not None else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Attribution heatmap
    im = axes[1].imshow(img)
    im2 = axes[1].imshow(attr_norm, cmap='hot', alpha=0.6)
    axes[1].set_title("Integrated Gradients Attribution")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], label='Attribution Magnitude')
    
    # Physiology attribution (if available)
    if phys_attr is not None and n_plots > 2:
        phys_names = phys_names or [f"Phys{i}" for i in range(len(phys_attr))]
        phys_sorted_idx = np.argsort(-np.abs(phys_attr))
        
        axes[2].barh(range(len(phys_attr)), phys_attr[phys_sorted_idx])
        axes[2].set_yticks(range(len(phys_attr)))
        axes[2].set_yticklabels([phys_names[i] for i in phys_sorted_idx])
        axes[2].set_xlabel("Attribution Value")
        axes[2].set_title("Physiology Attribution")
        axes[2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_attention(
    img: np.ndarray,
    gate_value: float,
    focus_value: float,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize attention mechanisms (gate and focus)
    
    Args:
        img: (3, H, W) or (H, W, 3) image
        gate_value: float in [0, 1]
        focus_value: float
        mask: (H, W) scar mask (optional)
        save_path: if provided, save to this path
    """
    img = denormalize_image(img)
    
    fig = plt.figure(figsize=(14, 5))
    
    # Image with mask overlay
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(img)
    if mask is not None:
        mask_binary = (mask > 0.5).astype(float)
        ax1.contourf(mask_binary, levels=[0.5, 1.5], colors='red', alpha=0.3)
    ax1.set_title("Input Image + Scar Region")
    ax1.axis('off')
    
    # Gate visualization
    ax2 = plt.subplot(1, 3, 2)
    gate_colors = {
        'Vision': gate_value,
        'Physiology': 1.0 - gate_value
    }
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax2.bar(gate_colors.keys(), gate_colors.values(), color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel("Gate Weight", fontsize=12)
    ax2.set_ylim([0, 1])
    ax2.set_title(f"CGF Gate (Vision Trust={gate_value:.3f})", fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, gate_colors.values()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Focus visualization
    ax3 = plt.subplot(1, 3, 3)
    focus_normalized = min(max(focus_value / 2.0, 0), 1)  # Normalize to [0, 1]
    ax3.barh(['Energy in\nScar Region'], [focus_normalized], color='#95E1D3', 
             edgecolor='black', linewidth=2, height=0.5)
    ax3.set_xlim([0, 1])
    ax3.set_xlabel("Normalized Focus", fontsize=12)
    ax3.set_title(f"CGF Focus (Raw={focus_value:.3f})", fontsize=12)
    ax3.text(focus_normalized + 0.05, 0, f'{focus_normalized:.3f}', 
            va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_fairness_xai(
    img: np.ndarray,
    scar_influence: float,
    fairness_risk: str,
    prediction: float,
    scar_present: bool,
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize fairness-aware explanation
    
    Args:
        img: (3, H, W) or (H, W, 3) image
        scar_influence: float in [0, 1] (how much scar matters)
        fairness_risk: "low", "medium", "high"
        prediction: P(threat=1)
        scar_present: whether scar=1
        save_path: if provided, save to this path
    """
    img = denormalize_image(img)
    
    fig = plt.figure(figsize=(14, 5))
    
    # Image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(img)
    scar_text = "With Scar" if scar_present else "No Scar"
    ax1.set_title(f"Input Image\n({scar_text})", fontsize=12)
    ax1.axis('off')
    
    # Risk assessment
    ax2 = plt.subplot(1, 3, 2)
    risk_colors = {'low': '#2ECC71', 'medium': '#F39C12', 'high': '#E74C3C'}
    risk_color = risk_colors.get(fairness_risk, '#95A5A6')
    
    # Vertical bar for scar influence
    ax2.barh(['Scar\nInfluence'], [scar_influence], color=risk_color, 
             edgecolor='black', linewidth=2, height=0.5, alpha=0.7)
    ax2.set_xlim([0, 1])
    ax2.set_xlabel("Influence Score", fontsize=12)
    ax2.set_title(f"Fairness Risk: {fairness_risk.upper()}", fontsize=12, 
                 color=risk_color, fontweight='bold')
    ax2.text(scar_influence + 0.05, 0, f'{scar_influence:.3f}', 
            va='center', fontweight='bold')
    
    # Prediction and risk summary
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    summary_text = f"""
EXPLANATION SUMMARY

Prediction: {prediction:.3f}
Class: {'THREAT' if prediction >= 0.5 else 'SAFE'}

Fairness Risk: {fairness_risk.upper()}
Scar Influence: {scar_influence:.3f}

Risk Interpretation:
"""
    
    if fairness_risk == 'low':
        summary_text += "✓ Model is fair\n  Scar has minimal\n  influence on decision"
    elif fairness_risk == 'medium':
        summary_text += "⚠ Moderate fairness\n  concern. Scar\n  somewhat influences\n  prediction"
    else:  # high
        summary_text += "✗ High fairness risk\n  Scar heavily\n  influences decision"
    
    ax3.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def create_explanation_report(
    explanations: dict,
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None
) -> str:
    """
    Create a comprehensive explanation report with all visualizations
    
    Args:
        explanations: dict of method_name -> ExplanationOutput
        img: (3, H, W) or (H, W, 3) image
        mask: (H, W) scar mask (optional)
        save_dir: directory to save visualizations (optional)
        
    Returns:
        Report text summary
    """
    report = "="*80 + "\n"
    report += "MULTIMODAL THREAT MODEL - EXPLANATION REPORT\n"
    report += "="*80 + "\n\n"
    
    for method_name, expl in explanations.items():
        report += f"\n{method_name.upper()}\n"
        report += "-" * 40 + "\n"
        report += f"Prediction: {expl.prediction:.4f} ({expl.prediction_class})\n"
        
        if expl.scar_influence_score is not None:
            report += f"Scar Influence: {expl.scar_influence_score:.4f}\n"
            report += f"Fairness Risk: {expl.fairness_risk}\n"
        
        if expl.gate_activation is not None:
            report += f"Gate Activation: {expl.gate_activation:.4f}\n"
        
        if expl.focus_activation is not None:
            report += f"Focus Activation: {expl.focus_activation:.4f}\n"
    
    report += "\n" + "="*80 + "\n"
    
    return report
