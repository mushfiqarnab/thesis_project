"""
Comprehensive Dataset Analysis and Results Visualization Script

This script provides:
1. Dataset analysis (features, samples, class distribution)
2. Before/After preprocessing comparison
3. Train/Test split visualization
4. Model evaluation metrics (AUC-ROC, F1, Precision, Accuracy)
5. Comprehensive visualizations (graphs, charts, diagrams)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "csv" / "multimodal_10k_unbiased.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DatasetAnalyzer:
    """Comprehensive dataset analysis and visualization"""
    
    def __init__(self, csv_path: Path, split_seed: int = 42, val_ratio: float = 0.2):
        self.csv_path = csv_path
        self.split_seed = split_seed
        self.val_ratio = val_ratio
        self.df_raw = None
        self.df_processed = None
        self.train_idx = None
        self.val_idx = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw CSV data (before preprocessing)"""
        print(f"📂 Loading raw dataset from: {self.csv_path}")
        self.df_raw = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(self.df_raw)} samples")
        return self.df_raw
    
    def load_processed_data(self) -> pd.DataFrame:
        """Load and preprocess data (simulating preprocessing steps)"""
        from src.dataset_fair import MultimodalCSVDatasetWithCF
        
        print(f"📂 Loading and preprocessing dataset...")
        dataset = MultimodalCSVDatasetWithCF(str(self.csv_path), verbose=False)
        
        # Get processed dataframe (after dataset preprocessing)
        self.df_processed = dataset.df.copy()
        
        # Apply preprocessing steps (matching dataset.py)
        self.df_processed["threat"] = pd.to_numeric(
            self.df_processed["threat"], errors="coerce"
        ).fillna(0).astype(int)
        self.df_processed["scar"] = pd.to_numeric(
            self.df_processed["scar"], errors="coerce"
        ).fillna(0).astype(int)
        
        # Fill missing physiology values
        phys_cols = dataset.phys_cols
        for col in phys_cols:
            if col in self.df_processed.columns:
                self.df_processed[col] = pd.to_numeric(
                    self.df_processed[col], errors="coerce"
                ).fillna(self.df_processed[col].median())
        
        # Drop rows with missing required fields
        required = ["image_path", "scar", "threat"] + phys_cols
        before = len(self.df_processed)
        self.df_processed = self.df_processed.dropna(subset=required).reset_index(drop=True)
        after = len(self.df_processed)
        
        if after < before:
            print(f"⚠️  Dropped {before - after} rows due to missing data")
        
        print(f"✅ Processed dataset: {len(self.df_processed)} samples")
        return self.df_processed
    
    def create_split(self) -> Tuple[list, list]:
        """Create or load train/validation split"""
        # Use dataset-specific split file name (matches project convention from train_cgf_fair.py)
        csv_stem = Path(self.csv_path).stem
        split_path = Path(self.csv_path).parent / f"split_seed{self.split_seed}_{csv_stem}.json"
        
        if split_path.exists():
            print(f"📂 Loading existing split from: {split_path}")
            split_data = json.loads(split_path.read_text(encoding="utf-8"))
            self.train_idx = split_data["train_idx"]
            self.val_idx = split_data["val_idx"]
        else:
            print(f"📂 Creating new split (seed={self.split_seed}, val_ratio={self.val_ratio})")
            n = len(self.df_processed)
            rng = np.random.default_rng(self.split_seed)
            idx = np.arange(n)
            rng.shuffle(idx)
            val_n = int(self.val_ratio * n)
            self.val_idx = idx[:val_n].tolist()
            self.train_idx = idx[val_n:].tolist()
            
            split_data = {
                "seed": self.split_seed,
                "val_ratio": self.val_ratio,
                "train_idx": self.train_idx,
                "val_idx": self.val_idx
            }
            split_path.write_text(json.dumps(split_data, indent=2), encoding="utf-8")
            print(f"💾 Saved split to: {split_path}")
        
        train_pct = len(self.train_idx) / len(self.df_processed) * 100
        val_pct = len(self.val_idx) / len(self.df_processed) * 100
        print(f"✅ Split: Train={len(self.train_idx)} ({train_pct:.1f}%), Val={len(self.val_idx)} ({val_pct:.1f}%)")
        
        return self.train_idx, self.val_idx
    
    def analyze_features(self) -> Dict:
        """Analyze dataset features"""
        print("\n" + "="*60)
        print("📊 DATASET FEATURE ANALYSIS")
        print("="*60)
        
        analysis = {
            "total_samples": len(self.df_raw),
            "features": {},
            "target_distribution": {},
            "physiology_features": []
        }
        
        # Feature names
        print(f"\n📋 Feature Names:")
        print(f"   Total columns: {len(self.df_raw.columns)}")
        for i, col in enumerate(self.df_raw.columns, 1):
            print(f"   {i}. {col}")
            analysis["features"][col] = {
                "dtype": str(self.df_raw[col].dtype),
                "missing": int(self.df_raw[col].isna().sum()),
                "unique": int(self.df_raw[col].nunique())
            }
        
        # Identify physiology features
        phys_prefixes = ("hrv", "gsr", "eda", "ecg", "bvp")
        phys_cols = [c for c in self.df_raw.columns 
                    if c.lower().startswith(phys_prefixes) 
                    and c.lower() not in ["image_path", "mask_path", "scar", "threat", "label"]]
        analysis["physiology_features"] = phys_cols
        
        print(f"\n🔬 Physiology Features ({len(phys_cols)}):")
        for col in phys_cols:
            print(f"   - {col}")
            if col in self.df_raw.columns:
                stats = self.df_raw[col].describe()
                print(f"     Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, "
                      f"Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
        
        # Target distribution
        if "threat" in self.df_raw.columns:
            threat_dist = self.df_raw["threat"].value_counts().to_dict()
            analysis["target_distribution"]["threat"] = threat_dist
            print(f"\n🎯 Target Distribution (Threat):")
            for label, count in sorted(threat_dist.items()):
                pct = count / len(self.df_raw) * 100
                print(f"   Class {label}: {count} samples ({pct:.2f}%)")
        
        if "scar" in self.df_raw.columns:
            scar_dist = self.df_raw["scar"].value_counts().to_dict()
            analysis["target_distribution"]["scar"] = scar_dist
            print(f"\n🔍 Scar Distribution:")
            for label, count in sorted(scar_dist.items()):
                pct = count / len(self.df_raw) * 100
                print(f"   Class {label}: {count} samples ({pct:.2f}%)")
        
        return analysis
    
    def plot_class_distribution_before_after(self):
        """Plot class distribution before and after preprocessing"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Class Distribution: Before vs After Preprocessing', fontsize=16, fontweight='bold')
        
        # Before preprocessing - Threat
        ax1 = axes[0, 0]
        threat_before = self.df_raw["threat"].value_counts().sort_index()
        colors_before = ['#3498db', '#e74c3c']
        bars1 = ax1.bar(threat_before.index.astype(str), threat_before.values, color=colors_before)
        ax1.set_title('Before Preprocessing: Threat Distribution', fontweight='bold')
        ax1.set_xlabel('Threat Label')
        ax1.set_ylabel('Number of Samples')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Safe (0)', 'Threat (1)'])
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(self.df_raw)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # After preprocessing - Threat
        ax2 = axes[0, 1]
        threat_after = self.df_processed["threat"].value_counts().sort_index()
        bars2 = ax2.bar(threat_after.index.astype(str), threat_after.values, color=colors_before)
        ax2.set_title('After Preprocessing: Threat Distribution', fontweight='bold')
        ax2.set_xlabel('Threat Label')
        ax2.set_ylabel('Number of Samples')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Safe (0)', 'Threat (1)'])
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(self.df_processed)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Before preprocessing - Scar
        ax3 = axes[1, 0]
        scar_before = self.df_raw["scar"].value_counts().sort_index()
        colors_scar = ['#2ecc71', '#f39c12']
        bars3 = ax3.bar(scar_before.index.astype(str), scar_before.values, color=colors_scar)
        ax3.set_title('Before Preprocessing: Scar Distribution', fontweight='bold')
        ax3.set_xlabel('Scar Label')
        ax3.set_ylabel('Number of Samples')
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['No Scar (0)', 'Scar (1)'])
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(self.df_raw)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # After preprocessing - Scar
        ax4 = axes[1, 1]
        scar_after = self.df_processed["scar"].value_counts().sort_index()
        bars4 = ax4.bar(scar_after.index.astype(str), scar_after.values, color=colors_scar)
        ax4.set_title('After Preprocessing: Scar Distribution', fontweight='bold')
        ax4.set_xlabel('Scar Label')
        ax4.set_ylabel('Number of Samples')
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['No Scar (0)', 'Scar (1)'])
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(self.df_processed)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = OUTPUT_DIR / "class_distribution_before_after.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        plt.close()
    
    def plot_train_test_split(self):
        """Visualize train/test split"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Train/Test Split Distribution', fontsize=16, fontweight='bold')
        
        # Pie chart
        ax1 = axes[0]
        sizes = [len(self.train_idx), len(self.val_idx)]
        labels = [f'Train\n{len(self.train_idx)} samples\n({len(self.train_idx)/len(self.df_processed)*100:.1f}%)',
                 f'Validation\n{len(self.val_idx)} samples\n({len(self.val_idx)/len(self.df_processed)*100:.1f}%)']
        colors = ['#3498db', '#e74c3c']
        explode = (0.05, 0.05)
        wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                           autopct='', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax1.set_title('Split Percentage', fontweight='bold', fontsize=14)
        
        # Bar chart with class distribution in each split
        ax2 = axes[1]
        train_threat = self.df_processed.iloc[self.train_idx]["threat"].value_counts().sort_index()
        val_threat = self.df_processed.iloc[self.val_idx]["threat"].value_counts().sort_index()
        
        x = np.arange(len(train_threat))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, train_threat.values, width, label='Train', color='#3498db', alpha=0.8)
        bars2 = ax2.bar(x + width/2, val_threat.values, width, label='Validation', color='#e74c3c', alpha=0.8)
        
        ax2.set_xlabel('Threat Label', fontweight='bold')
        ax2.set_ylabel('Number of Samples', fontweight='bold')
        ax2.set_title('Class Distribution in Train/Validation Split', fontweight='bold', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Safe (0)', 'Threat (1)'])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        save_path = OUTPUT_DIR / "train_test_split.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        plt.close()
    
    def plot_feature_statistics(self):
        """Plot statistics of physiology features"""
        from src.dataset_fair import MultimodalCSVDatasetWithCF
        dataset = MultimodalCSVDatasetWithCF(str(self.csv_path), verbose=False)
        phys_cols = dataset.phys_cols
        
        if len(phys_cols) == 0:
            print("⚠️  No physiology features found")
            return
        
        n_features = len(phys_cols)
        n_cols = min(2, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('Physiology Feature Distributions', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(phys_cols):
            ax = axes[idx]
            data = self.df_processed[col].dropna()
            
            # Histogram
            ax.hist(data, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.4f}')
            ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.4f}')
            
            ax.set_title(f'{col}', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = OUTPUT_DIR / "feature_statistics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        plt.close()


class ModelEvaluator:
    """Evaluate model and generate metrics"""
    
    def __init__(self, model_path: Optional[Path] = None, csv_path: Path = CSV_PATH):
        self.model_path = model_path
        self.csv_path = csv_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load trained model"""
        if self.model_path is None or not self.model_path.exists():
            print("⚠️  No model checkpoint provided. Skipping model evaluation.")
            return None
        
        print(f"📂 Loading model from: {self.model_path}")
        # Model loading logic would go here
        # For now, we'll create a placeholder
        print("⚠️  Model loading not implemented. Please provide predictions manually.")
        return None
    
    def evaluate_with_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_proba: Optional[np.ndarray] = None) -> Dict:
        """Calculate evaluation metrics"""
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION METRICS")
        print("="*60)
        
        metrics = {}
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        metrics["accuracy"] = accuracy
        print(f"\n✅ Accuracy: {accuracy:.4f}")
        
        # Precision
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics["precision"] = precision
        print(f"✅ Precision: {precision:.4f}")
        
        # Recall
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics["recall"] = recall
        print(f"✅ Recall: {recall:.4f}")
        
        # F1 Score
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        metrics["f1_score"] = f1
        print(f"✅ F1 Score: {f1:.4f}")
        
        # AUC-ROC
        if y_proba is not None:
            try:
                auc_roc = roc_auc_score(y_true, y_proba)
                metrics["auc_roc"] = auc_roc
                print(f"✅ AUC-ROC: {auc_roc:.4f}")
            except ValueError as e:
                print(f"⚠️  AUC-ROC calculation failed: {e}")
                metrics["auc_roc"] = None
        else:
            print("⚠️  No probability predictions provided. AUC-ROC cannot be calculated.")
            metrics["auc_roc"] = None
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        print(f"\n📋 Confusion Matrix:")
        print(f"                Predicted")
        print(f"              Safe  Threat")
        print(f"Actual Safe    {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"      Threat    {cm[1,0]:4d}   {cm[1,1]:4d}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Safe (0)', 'Threat (1)'],
                   yticklabels=['Safe (0)', 'Threat (1)'],
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
        ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
        ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        save_path = OUTPUT_DIR / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray):
        """Plot ROC curve"""
        if y_proba is None:
            print("⚠️  No probability predictions. Skipping ROC curve.")
            return
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC Curve (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
        ax.set_title('ROC Curve', fontweight='bold', fontsize=14)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = OUTPUT_DIR / "roc_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        plt.close()
    
    def plot_metrics_summary(self, metrics: Dict):
        """Plot metrics summary bar chart"""
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0)
        ]
        
        # Filter out None values
        filtered_names = []
        filtered_values = []
        for name, value in zip(metric_names, metric_values):
            if value is not None:
                filtered_names.append(name)
                filtered_values.append(value)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(filtered_names, filtered_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title('Model Performance Metrics', fontweight='bold', fontsize=14)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        save_path = OUTPUT_DIR / "metrics_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        plt.close()
        
        # Add AUC-ROC if available
        if metrics.get('auc_roc') is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(['AUC-ROC'], [metrics['auc_roc']], color='#9b59b6', alpha=0.8)
            ax.set_ylabel('Score', fontweight='bold', fontsize=12)
            ax.set_title('AUC-ROC Score', fontweight='bold', fontsize=14)
            ax.set_ylim([0, 1])
            ax.text(0, metrics['auc_roc'],
                   f'{metrics["auc_roc"]:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            save_path = OUTPUT_DIR / "auc_roc_score.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")
            plt.close()


def main():
    """Main analysis pipeline"""
    print("="*60)
    print("🚀 COMPREHENSIVE DATASET ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(CSV_PATH)
    
    # 1. Load raw data
    df_raw = analyzer.load_raw_data()
    
    # 2. Load processed data
    df_processed = analyzer.load_processed_data()
    
    # 3. Create split
    train_idx, val_idx = analyzer.create_split()
    
    # 4. Analyze features
    feature_analysis = analyzer.analyze_features()
    
    # 5. Generate visualizations
    print("\n" + "="*60)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*60)
    
    analyzer.plot_class_distribution_before_after()
    analyzer.plot_train_test_split()
    analyzer.plot_feature_statistics()
    
    # 6. Save analysis report
    report = {
        "dataset_info": {
            "csv_path": str(CSV_PATH),
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
            "train_percentage": len(train_idx) / len(df_processed) * 100,
            "validation_samples": len(val_idx),
            "validation_percentage": len(val_idx) / len(df_processed) * 100
        }
    }
    
    report_path = OUTPUT_DIR / "dataset_analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n💾 Saved analysis report: {report_path}")
    
    # 7. Model evaluation (if model provided)
    print("\n" + "="*60)
    print("🤖 MODEL EVALUATION")
    print("="*60)
    print("ℹ️  To evaluate a model, provide predictions using evaluate_with_predictions()")
    print("   Example:")
    print("   evaluator = ModelEvaluator()")
    print("   metrics = evaluator.evaluate_with_predictions(y_true, y_pred, y_proba)")
    print("   evaluator.plot_confusion_matrix(y_true, y_pred)")
    print("   evaluator.plot_roc_curve(y_true, y_proba)")
    print("   evaluator.plot_metrics_summary(metrics)")
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print(f"📁 All outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"   - {file.name}")
    print(f"   - dataset_analysis_report.json")


if __name__ == "__main__":
    main()
