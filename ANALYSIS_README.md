# Comprehensive Dataset Analysis & Model Evaluation

This guide explains how to generate comprehensive analysis reports, visualizations, and model evaluation metrics for your thesis project.

## 📋 What This Script Provides

### Dataset Analysis:
1. **Feature Analysis**
   - Complete list of all features/columns
   - Number of samples
   - Feature statistics (mean, std, min, max)
   - Missing value analysis

2. **Class Distribution**
   - Before preprocessing vs After preprocessing
   - Threat label distribution
   - Scar label distribution
   - Visual comparisons

3. **Train/Test Split**
   - Split percentages
   - Class distribution in each split
   - Visual representations

4. **Feature Statistics**
   - Physiology feature distributions
   - Histograms and statistics

### Model Evaluation (if checkpoint provided):
1. **Metrics**
   - ✅ Accuracy
   - ✅ Precision
   - ✅ Recall (Sensitivity)
   - ✅ F1 Score
   - ✅ AUC-ROC

2. **Visualizations**
   - Confusion Matrix
   - ROC Curve
   - Metrics Summary Bar Chart
   - AUC-ROC Score Visualization

## 🚀 Quick Start

### 1. Dataset Analysis Only

```bash
python run_comprehensive_analysis.py
```

This will:
- Analyze the dataset (`data/csv/multimodal_10k_unbiased.csv`)
- Generate all dataset visualizations
- Create a comprehensive analysis report
- Save everything to `outputs/analysis/`

### 2. Dataset Analysis + Model Evaluation

```bash
python run_comprehensive_analysis.py --checkpoint outputs/checkpoints/your_model.pt
```

This will:
- Perform all dataset analysis
- Load and evaluate the model
- Calculate all metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
- Generate evaluation visualizations
- Save evaluation report

### 3. Custom Options

```bash
python run_comprehensive_analysis.py \
    --csv data/csv/your_dataset.csv \
    --checkpoint outputs/checkpoints/model.pt \
    --split-seed 42 \
    --val-ratio 0.2 \
    --batch-size 32
```

## 📊 Generated Outputs

All outputs are saved to `outputs/analysis/`:

### Visualizations (PNG files):
- `class_distribution_before_after.png` - Class distribution comparison
- `train_test_split.png` - Train/validation split visualization
- `feature_statistics.png` - Physiology feature distributions
- `confusion_matrix.png` - Model confusion matrix (if model evaluated)
- `roc_curve.png` - ROC curve with AUC score (if model evaluated)
- `metrics_summary.png` - Bar chart of all metrics (if model evaluated)
- `auc_roc_score.png` - AUC-ROC visualization (if model evaluated)

### Reports (JSON files):
- `dataset_analysis_report.json` - Complete dataset analysis
- `evaluation_report_<model_name>.json` - Model evaluation metrics (if model evaluated)

## 📈 Understanding the Results

### Dataset Analysis Report Structure:

```json
{
  "dataset_info": {
    "total_samples_raw": 10001,
    "total_samples_processed": 10000,
    "samples_dropped": 1
  },
  "features": {
    "image_path": {...},
    "hrv": {...},
    "gsr": {...},
    "scar": {...},
    "threat": {...}
  },
  "physiology_features": ["hrv", "gsr"],
  "class_distribution": {
    "before_preprocessing": {...},
    "after_preprocessing": {...}
  },
  "train_test_split": {
    "train_samples": 8000,
    "train_percentage": 80.0,
    "validation_samples": 2000,
    "validation_percentage": 20.0
  }
}
```

### Model Evaluation Report Structure:

```json
{
  "checkpoint": "path/to/model.pt",
  "backbone": "mobilenet_v3_small",
  "fusion": "cgf",
  "num_samples": 2000,
  "metrics": {
    "accuracy": 0.8500,
    "precision": 0.8200,
    "recall": 0.8800,
    "f1_score": 0.8490,
    "auc_roc": 0.9200,
    "confusion_matrix": [[...], [...]]
  }
}
```

## 🔍 Key Metrics Explained

### Accuracy
- Overall correctness: (TP + TN) / (TP + TN + FP + FN)
- Range: 0 to 1 (higher is better)

### Precision
- Positive predictive value: TP / (TP + FP)
- Range: 0 to 1 (higher is better)
- Measures: "Of all predicted threats, how many were actually threats?"

### Recall (Sensitivity)
- True positive rate: TP / (TP + FN)
- Range: 0 to 1 (higher is better)
- Measures: "Of all actual threats, how many did we catch?"

### F1 Score
- Harmonic mean of precision and recall: 2 × (Precision × Recall) / (Precision + Recall)
- Range: 0 to 1 (higher is better)
- Balanced metric for imbalanced datasets

### AUC-ROC
- Area Under the ROC Curve
- Range: 0 to 1 (higher is better, 0.5 = random)
- Measures: Overall classification performance across all thresholds

## 📝 Example Output

```
======================================================================
🚀 COMPREHENSIVE DATASET ANALYSIS & MODEL EVALUATION
======================================================================

📊 PHASE 1: DATASET ANALYSIS
======================================================================
📂 Loading raw dataset from: data/csv/multimodal_10k_unbiased.csv
✅ Loaded 10001 samples
📂 Loading and preprocessing dataset...
✅ Processed dataset: 10000 samples
📂 Creating new split (seed=42, val_ratio=0.2)
✅ Split: Train=8000 (80.0%), Val=2000 (20.0%)

📊 DATASET FEATURE ANALYSIS
======================================================================
📋 Feature Names:
   Total columns: 6
   1. image_path
   2. hrv
   3. gsr
   4. scar
   5. threat
   6. mask_path

🔬 Physiology Features (2):
   - hrv
   - gsr

🎯 Target Distribution (Threat):
   Class 0: 5000 samples (50.00%)
   Class 1: 5000 samples (50.00%)

📊 Generating dataset visualizations...
💾 Saved: outputs/analysis/class_distribution_before_after.png
💾 Saved: outputs/analysis/train_test_split.png
💾 Saved: outputs/analysis/feature_statistics.png

======================================================================
✅ ANALYSIS COMPLETE!
======================================================================
📁 All outputs saved to: outputs/analysis
```

## 🛠️ Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch torchvision
```

### Issue: Checkpoint not found
**Solution**: Make sure the checkpoint path is correct. Check available checkpoints:
```bash
ls outputs/checkpoints/
```

### Issue: Split file not found
**Solution**: The script will automatically create a split file. If you want to use a specific split:
```bash
python run_comprehensive_analysis.py --split data/csv/split_seed42.json
```

## 📚 Additional Information

### Dataset Features:
- **image_path**: Path to face image
- **hrv**: Heart Rate Variability feature
- **gsr**: Galvanic Skin Response feature
- **scar**: Binary label (0=no scar, 1=scar)
- **threat**: Binary label (0=safe, 1=threat)
- **mask_path**: Path to scar mask (if scar=1)

### Preprocessing Steps:
1. Convert numeric columns to proper types
2. Fill missing physiology values with median
3. Drop rows with missing required fields
4. Clamp labels to {0, 1}

### Train/Test Split:
- Default: 80% train, 20% validation
- Seed: 42 (for reproducibility)
- Stratified by default (maintains class distribution)

## 📞 Support

For issues or questions, check:
1. The generated JSON reports for detailed information
2. The visualization PNG files for visual insights
3. The console output for warnings and errors

---

**Note**: This script is designed to work with the multimodal threat detection dataset structure. If your dataset has a different structure, you may need to modify the `DatasetAnalyzer` class in `src/comprehensive_analysis.py`.
