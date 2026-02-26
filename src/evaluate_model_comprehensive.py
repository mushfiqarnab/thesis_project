"""
Comprehensive Model Evaluation Script

This script evaluates a trained model and generates:
- Accuracy, Precision, Recall, F1 Score, AUC-ROC
- Confusion Matrix
- ROC Curve
- All visualizations
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
from src.models import MultimodalThreatModel
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.comprehensive_analysis import ModelEvaluator, OUTPUT_DIR

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_model(checkpoint_path: Path, csv_path: Path, device: torch.device):
    """Load model from checkpoint"""
    print(f"📂 Loading model from: {checkpoint_path}")
    
    # Load dataset to get phys_dim
    dataset = MultimodalCSVDatasetWithCF(str(csv_path), verbose=False)
    phys_dim = len(dataset.phys_cols)
    
    # Try to infer model architecture from checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    # Check for fusion type and backbone
    has_cgf = any("gate_mlp" in k for k in state.keys())
    has_mobilenet = any("vision.features" in k for k in state.keys())
    has_vit = any("vision.vit" in k for k in state.keys())
    
    if has_cgf:
        fusion = "cgf"
    else:
        fusion = "concat"
    
    if has_mobilenet:
        backbone = "mobilenet_v3_small"
    elif has_vit:
        backbone = "vit_b_16"
    else:
        backbone = "mobilenet_v3_small"  # default
    
    print(f"   Detected: backbone={backbone}, fusion={fusion}")
    
    model = MultimodalThreatModel(
        phys_dim=phys_dim,
        vision_backbone=backbone,
        fusion=fusion,
        num_classes=2
    ).to(device)
    
    # Load weights
    model.load_state_dict(state, strict=False)
    model.eval()
    print("✅ Model loaded successfully")
    
    return model, fusion, backbone


def evaluate_model(model, loader, device, fusion_type: str):
    """Evaluate model and return predictions"""
    print("\n🔄 Running inference...")
    
    all_probs = []
    all_preds = []
    all_labels = []
    all_scars = []
    
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device)
            phys = batch["phys"].to(device)
            y = batch["y"].cpu().numpy()
            scar = batch["scar"].cpu().numpy()
            mask = batch["mask"].to(device) if fusion_type == "cgf" else None
            
            if fusion_type == "cgf":
                out = model(img, phys, mask=mask)
            else:
                out = model(img, phys)
            
            probs = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y)
            all_scars.extend(scar)
    
    return np.array(all_probs), np.array(all_preds), np.array(all_labels), np.array(all_scars)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--csv", type=str, default="data/csv/multimodal_10k_unbiased.csv", help="Path to CSV file")
    parser.add_argument("--split", type=str, help="Path to split JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    csv_path = Path(args.csv)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load model
    model, fusion_type, backbone = load_model(checkpoint_path, csv_path, device)
    
    # Load dataset and split
    dataset = MultimodalCSVDatasetWithCF(str(csv_path), verbose=False)
    
    if args.split:
        split_path = Path(args.split)
    else:
        # Use dataset-specific split file name (matches project convention)
        csv_stem = csv_path.stem
        split_path = csv_path.parent / f"split_seed{args.seed}_{csv_stem}.json"
    
    if split_path.exists():
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        val_idx = split_data.get("val_idx", [])
    else:
        print("⚠️  Split file not found. Using full dataset for evaluation.")
        val_idx = list(range(len(dataset)))
    
    val_dataset = Subset(dataset, val_idx)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples
    )
    
    # Evaluate
    probs, preds, labels, scars = evaluate_model(model, val_loader, device, fusion_type)
    
    # Calculate metrics
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_with_predictions(labels, preds, probs)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    evaluator.plot_confusion_matrix(labels, preds)
    evaluator.plot_roc_curve(labels, probs)
    evaluator.plot_metrics_summary(metrics)
    
    # Save metrics to JSON
    report = {
        "checkpoint": str(checkpoint_path),
        "csv": str(csv_path),
        "backbone": backbone,
        "fusion": fusion_type,
        "num_samples": len(labels),
        "metrics": metrics
    }
    
    report_path = output_dir / f"evaluation_report_{checkpoint_path.stem}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n💾 Saved evaluation report: {report_path}")
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
