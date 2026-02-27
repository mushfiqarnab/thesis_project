"""
XAI Demo and Tutorial

This script demonstrates how to use the XAI module to explain predictions.

Usage:
    python src/xai/demo.py --ckpt <checkpoint_path> --csv <csv_path> --sample_idx <index>
"""

import sys
from pathlib import Path
import argparse
import json
import torch
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import MultimodalThreatModel
from dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from xai import XAIExplainer, ExplanationOutput
from xai.visualization import (
    visualize_saliency, visualize_integrated_gradients,
    visualize_attention, visualize_fairness_xai,
    create_explanation_report
)


def load_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    """Load checkpoint state dict safely"""
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    # Clean module prefix
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="XAI Demo: Explain model predictions")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--csv", type=str, required=True, help="CSV dataset path")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to explain")
    parser.add_argument("--fusion", type=str, default="cgf", choices=["concat", "cgf"])
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--ig_steps", type=int, default=50, help="IG interpolation steps")
    parser.add_argument("--save_dir", type=str, default="", help="Save visualizations to this dir")
    parser.add_argument("--method", type=str, default="all", 
                       choices=["all", "integrated_gradients", "saliency_map", "attention", "fairness_xai"])
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[XAI Demo] Using device: {device}")
    
    # Load dataset
    print(f"\n[Loading] Dataset from {args.csv}")
    ds = MultimodalCSVDatasetWithCF(args.csv)
    print(f"  Loaded {len(ds)} samples, physiology dim = {ds[0].phys.numel()}")
    
    # Load model
    print(f"\n[Loading] Checkpoint from {args.ckpt}")
    state = load_checkpoint(Path(args.ckpt), device)
    
    phys_dim = ds[0].phys.numel()
    model = MultimodalThreatModel(
        phys_dim=phys_dim,
        vision_backbone=args.backbone,
        fusion=args.fusion,
        num_classes=2
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  Model loaded: {args.backbone} + {args.fusion} fusion")
    
    # Load sample
    print(f"\n[Sample] Loading sample {args.sample_idx}")
    sample = ds[args.sample_idx]
    
    img = sample.img.unsqueeze(0).to(device)
    phys = sample.phys.unsqueeze(0).to(device)
    mask = sample.mask.unsqueeze(0).to(device)
    scar = sample.scar.unsqueeze(0).to(device)
    y_true = sample.y.item()
    
    print(f"  Scar: {sample.scar.item()}")
    print(f"  True Label: {y_true} ({'THREAT' if y_true else 'SAFE'})")
    print(f"  Has CF: {sample.has_cf.item()}")
    
    # Get prediction
    with torch.no_grad():
        out = model(img, phys, mask=mask)
        probs = torch.softmax(out.logits, dim=1)
        pred = probs[0, 1].item()
    
    pred_class = 1 if pred >= args.threshold else 0
    print(f"\n[Prediction] P(threat=1) = {pred:.4f} → Class {pred_class} "
          f"({'THREAT' if pred_class else 'SAFE'})")
    
    # Initialize XAI
    print(f"\n[XAI] Initializing explainer...")
    explainer = XAIExplainer(model, device=device)
    
    # Generate explanations
    save_dir = Path(args.save_dir) if args.save_dir else None
    save_dir_actual = save_dir or (PROJECT_ROOT / "outputs" / "xai_explanations")
    save_dir_actual.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Generating] Explanations (method={args.method})...")
    
    if args.method == "all":
        print("  Running: Integrated Gradients, Saliency Map, Attention, Fairness XAI")
        explanations = explainer.explain_all(
            img, phys, mask=mask, scar=scar, ig_steps=args.ig_steps
        )
    else:
        print(f"  Running: {args.method}")
        explanations = {
            args.method: explainer.explain_single_method(
                args.method, img, phys, mask=mask, scar=scar, steps=args.ig_steps
            )
        }
    
    # Print report
    report = create_explanation_report(explanations, img[0].detach().cpu().numpy(), 
                                       mask[0, 0].detach().cpu().numpy() if mask is not None else None)
    print("\n" + report)
    
    # Save explanations to JSON
    explanations_json = {}
    for method_name, expl in explanations.items():
        explanations_json[method_name] = {
            'prediction': expl.prediction,
            'prediction_class': expl.prediction_class,
            'gate_activation': expl.gate_activation,
            'focus_activation': expl.focus_activation,
            'scar_influence_score': expl.scar_influence_score,
            'fairness_risk': expl.fairness_risk,
            'metadata': expl.metadata
        }
    
    json_path = save_dir_actual / f"explanations_sample{args.sample_idx}.json"
    json_path.write_text(json.dumps(explanations_json, indent=2), encoding='utf-8')
    print(f"\n[Saved] Explanations to {json_path}")
    
    # Generate visualizations
    print(f"\n[Visualizing] Creating explanation visualizations...")
    
    img_np = img[0].detach().cpu().numpy()
    mask_np = mask[0, 0].detach().cpu().numpy() if mask is not None else None
    
    for method_name, expl in explanations.items():
        try:
            if method_name == "saliency_map" and expl.vision_attribution is not None:
                save_path = save_dir_actual / f"saliency_sample{args.sample_idx}.png"
                visualize_saliency(img_np, expl.vision_attribution, mask=mask_np, 
                                 title=f"Saliency (pred={expl.prediction:.3f})",
                                 save_path=save_path)
                
            elif method_name == "integrated_gradients" and expl.vision_attribution is not None:
                save_path = save_dir_actual / f"integrated_gradients_sample{args.sample_idx}.png"
                visualize_integrated_gradients(
                    img_np, expl.vision_attribution, 
                    phys_attr=expl.phys_attribution,
                    phys_names=ds.phys_cols if hasattr(ds, 'phys_cols') else None,
                    mask=mask_np,
                    save_path=save_path
                )
                
            elif method_name == "attention" and expl.gate_activation is not None:
                save_path = save_dir_actual / f"attention_sample{args.sample_idx}.png"
                visualize_attention(img_np, expl.gate_activation, expl.focus_activation,
                                  mask=mask_np, save_path=save_path)
                
            elif method_name == "fairness_xai" and expl.scar_influence_score is not None:
                save_path = save_dir_actual / f"fairness_xai_sample{args.sample_idx}.png"
                visualize_fairness_xai(
                    img_np, expl.scar_influence_score, expl.fairness_risk,
                    expl.prediction, sample.scar.item(),
                    save_path=save_path
                )
                
            print(f"  ✓ {method_name}")
            
        except Exception as e:
            print(f"  ✗ {method_name}: {e}")
    
    print(f"\n[Done] All visualizations saved to {save_dir_actual}")
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
