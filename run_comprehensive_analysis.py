"""
Main script to run comprehensive dataset analysis and model evaluation

Usage:
    python run_comprehensive_analysis.py                    # Dataset analysis only
    python run_comprehensive_analysis.py --checkpoint <path>  # Include model evaluation
"""

import argparse
from pathlib import Path
import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        import io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass  # If it fails, continue without emoji support

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.comprehensive_analysis import DatasetAnalyzer, ModelEvaluator, OUTPUT_DIR
from src.evaluate_model_comprehensive import load_model, evaluate_model
from torch.utils.data import DataLoader, Subset
from src.dataset_fair import MultimodalCSVDatasetWithCF, collate_samples
import torch
import numpy as np
import json


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Dataset Analysis and Model Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (optional)")
    parser.add_argument("--csv", type=str, default="data/csv/multimodal_10k_unbiased.csv", help="Path to CSV file")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for model evaluation")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print("="*70)
    print("🚀 COMPREHENSIVE DATASET ANALYSIS & MODEL EVALUATION")
    print("="*70)
    
    # ========== DATASET ANALYSIS ==========
    print("\n" + "="*70)
    print("📊 PHASE 1: DATASET ANALYSIS")
    print("="*70)
    
    analyzer = DatasetAnalyzer(csv_path, split_seed=args.split_seed, val_ratio=args.val_ratio)
    
    # Load data
    df_raw = analyzer.load_raw_data()
    df_processed = analyzer.load_processed_data()
    
    # Create split
    train_idx, val_idx = analyzer.create_split()
    
    # Analyze features
    feature_analysis = analyzer.analyze_features()
    
    # Generate visualizations
    print("\n📊 Generating dataset visualizations...")
    analyzer.plot_class_distribution_before_after()
    analyzer.plot_train_test_split()
    analyzer.plot_feature_statistics()
    
    # Save analysis report
    report = {
        "dataset_info": {
            "csv_path": str(csv_path),
            "total_samples_raw": len(df_raw),
            "total_samples_processed": len(df_processed),
            "samples_dropped": len(df_raw) - len(df_processed)
        },
        "features": feature_analysis["features"],
        "physiology_features": feature_analysis["physiology_features"],
        "class_distribution": {
            "before_preprocessing": feature_analysis["target_distribution"],
            "after_preprocessing": {
                "threat": df_processed["threat"].value_counts().to_dict(),
                "scar": df_processed["scar"].value_counts().to_dict()
            }
        },
        "train_test_split": {
            "seed": analyzer.split_seed,
            "val_ratio": analyzer.val_ratio,
            "train_samples": len(train_idx),
            "train_percentage": round(len(train_idx) / len(df_processed) * 100, 2),
            "validation_samples": len(val_idx),
            "validation_percentage": round(len(val_idx) / len(df_processed) * 100, 2)
        }
    }
    
    report_path = OUTPUT_DIR / "dataset_analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Saved analysis report: {report_path}")
    
    # ========== MODEL EVALUATION ==========
    if args.checkpoint:
        print("\n" + "="*70)
        print("🤖 PHASE 2: MODEL EVALUATION")
        print("="*70)
        
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("   Skipping model evaluation...")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🖥️  Using device: {device}")
            
            # Load model
            model, fusion_type, backbone = load_model(checkpoint_path, csv_path, device)
            
            # Load dataset and split
            dataset = MultimodalCSVDatasetWithCF(str(csv_path), verbose=False)
            
            # Use dataset-specific split file name (matches project convention)
            csv_stem = csv_path.stem
            split_path = csv_path.parent / f"split_seed{args.split_seed}_{csv_stem}.json"
            if split_path.exists():
                split_data = json.loads(split_path.read_text(encoding="utf-8"))
                val_idx = split_data.get("val_idx", list(range(len(dataset))))
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
            print("\n🔄 Running model inference...")
            probs, preds, labels, scars = evaluate_model(model, val_loader, device, fusion_type)
            
            # Calculate metrics
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_with_predictions(labels, preds, probs)
            
            # Generate visualizations
            print("\n📊 Generating evaluation visualizations...")
            evaluator.plot_confusion_matrix(labels, preds)
            evaluator.plot_roc_curve(labels, probs)
            evaluator.plot_metrics_summary(metrics)
            
            # Save evaluation report
            eval_report = {
                "checkpoint": str(checkpoint_path),
                "csv": str(csv_path),
                "backbone": backbone,
                "fusion": fusion_type,
                "num_samples": len(labels),
                "metrics": metrics
            }
            
            eval_report_path = OUTPUT_DIR / f"evaluation_report_{checkpoint_path.stem}.json"
            eval_report_path.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
            print(f"💾 Saved evaluation report: {eval_report_path}")
    else:
        print("\n" + "="*70)
        print("ℹ️  MODEL EVALUATION SKIPPED")
        print("="*70)
        print("   To evaluate a model, run with --checkpoint <path>")
        print("   Example: python run_comprehensive_analysis.py --checkpoint outputs/checkpoints/model.pt")
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"📁 All outputs saved to: {OUTPUT_DIR}")
    print("\n📊 Generated files:")
    for file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"   ✓ {file.name}")
    json_files = list(OUTPUT_DIR.glob("*.json"))
    if json_files:
        print("\n📄 Generated reports:")
        for file in sorted(json_files):
            print(f"   ✓ {file.name}")
    
    print("\n" + "="*70)
    print("📋 KEY FINDINGS SUMMARY")
    print("="*70)
    print(f"   • Total Samples (Raw): {len(df_raw):,}")
    print(f"   • Total Samples (Processed): {len(df_processed):,}")
    print(f"   • Samples Dropped: {len(df_raw) - len(df_processed):,}")
    print(f"   • Physiology Features: {len(feature_analysis['physiology_features'])}")
    print(f"   • Train Split: {len(train_idx):,} ({len(train_idx)/len(df_processed)*100:.1f}%)")
    print(f"   • Validation Split: {len(val_idx):,} ({len(val_idx)/len(df_processed)*100:.1f}%)")
    
    if args.checkpoint and checkpoint_path.exists():
        print(f"\n🤖 Model Evaluation:")
        print(f"   • Backbone: {backbone}")
        print(f"   • Fusion: {fusion_type}")
        if 'metrics' in locals():
            print(f"   • Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"   • Precision: {metrics.get('precision', 0):.4f}")
            print(f"   • Recall: {metrics.get('recall', 0):.4f}")
            print(f"   • F1 Score: {metrics.get('f1_score', 0):.4f}")
            if metrics.get('auc_roc'):
                print(f"   • AUC-ROC: {metrics['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
