# DATASET ANALYSIS - COMPREHENSIVE GUIDE

## ✅ What's Needed for Dataset Analysis - COMPLETE CHECKLIST

### 1. **DATASET INFORMATION** ✅
| Item | Status | Details |
|------|--------|---------|
| Dataset Path | ✅ | `data/csv/multimodal_10k_unbiased.csv` |
| Total Samples | ✅ | **10,000 samples** |
| Samples Dropped | ✅ | 0 (100% clean) |
| Data Quality | ✅ | No missing values in required fields |

### 2. **FEATURES & FEATURE NAMES** ✅
| # | Feature Name | Type | Missing | Unique | Role |
|---|---|---|---|---|---|
| 1 | `image_path` | String | 0 | 10,000 | Image file path (vision modality) |
| 2 | `hrv` | Float64 | 0 | 1,741 | Heart Rate Variability (physiology) |
| 3 | `gsr` | Float64 | 0 | 1,832 | Galvanic Skin Response (physiology) |
| 4 | `scar` | Int64 | 0 | 2 | Sensitive attribute (0=No scar, 1=Scar) |
| 5 | `threat` | Int64 | 0 | 2 | **PRIMARY TARGET** (0=Safe, 1=Threat) |
| 6 | `mask_path` | String | 4,982 | 5,018 | Scar mask path (optional, for scar samples) |
| 7 | `subject` | String | 0 | 15 | Subject ID (15 different subjects) |

**Total Columns: 7**  
**Physiology Features: 2** (HRV, GSR)

---

## 📊 FEATURES STATISTICS

### Physiology Features - Detailed Statistics

#### **1. HRV (Heart Rate Variability)**
```
Mean:  0.0341
Std:   0.0233
Min:   0.0014
Max:   0.1469
```

#### **2. GSR (Galvanic Skin Response)**
```
Mean:  4.6173
Std:   3.4910
Min:   0.7367
Max:   20.2059
```

---

## 📈 CLASS DISTRIBUTION ANALYSIS

### **BEFORE vs AFTER PREPROCESSING**

#### Threat Label (Primary Target)
| Class | Before | After | Count | % |
|-------|--------|-------|-------|---|
| 0 (Safe) | 6,377 | 6,377 | 6,377 | 63.77% |
| 1 (Threat) | 3,623 | 3,623 | 3,623 | 36.23% |
| **Total** | **10,000** | **10,000** | **10,000** | **100%** |

**Status:** No samples dropped during preprocessing ✅

#### Scar Label (Sensitive Attribute)
| Class | Before | After | Count | % |
|-------|--------|-------|-------|---|
| 0 (No Scar) | 4,982 | 4,982 | 4,982 | 49.82% |
| 1 (Scar) | 5,018 | 5,018 | 5,018 | 50.18% |
| **Total** | **10,000** | **10,000** | **10,000** | **100%** |

**Observation:** Well-balanced scar distribution (nearly 50-50 split)

---

## 🔄 TRAIN/TEST SPLIT INFORMATION

### Split Configuration
| Parameter | Value |
|-----------|-------|
| **Split Seed** | 42 |
| **Validation Ratio** | 0.2 (20%) |
| **Split File** | `data/csv/split_seed42_multimodal_10k_unbiased.json` |

### Split Distribution
| Set | Samples | Percentage |
|-----|---------|-----------|
| **Train** | 8,000 | **80.0%** |
| **Validation** | 2,000 | **20.0%** |
| **Total** | 10,000 | 100% |

### Class Distribution in Each Split
**Training Set (8,000 samples):**
- Threat=0 (Safe): ~5,101 samples (63.77%)
- Threat=1 (Threat): ~2,899 samples (36.23%)

**Validation Set (2,000 samples):**
- Threat=0 (Safe): ~1,276 samples (63.77%)
- Threat=1 (Threat): ~724 samples (36.23%)

---

## 🎯 RESULTS & METRICS

### How to Generate Model Evaluation Metrics

To evaluate a trained model and generate the following metrics:

```bash
python run_comprehensive_analysis.py --checkpoint <path_to_model.pt>
```

#### **Metrics Generated:**
- ✅ **Accuracy** - Overall correctness
- ✅ **Precision** - True positives / (True positives + False positives)
- ✅ **Recall (Sensitivity)** - True positives / (True positives + False negatives)
- ✅ **F1 Score** - Harmonic mean of precision and recall
- ✅ **AUC-ROC** - Area under the Receiver Operating Characteristic curve

#### **Example Command:**
```bash
# Evaluate a specific model
python run_comprehensive_analysis.py --checkpoint outputs/checkpoints/counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.pt

# Custom options
python run_comprehensive_analysis.py \
    --csv data/csv/multimodal_10k_unbiased.csv \
    --checkpoint outputs/checkpoints/model.pt \
    --split-seed 42 \
    --val-ratio 0.2 \
    --batch-size 32
```

---

## 📊 OUTPUTS & VISUALIZATIONS

### Generated Files (In `outputs/analysis/`)

#### **Visualizations (PNG)**
1. ✅ **`class_distribution_before_after.png`**
   - 4-panel visualization comparing threat and scar distributions
   - Before vs after preprocessing
   - Shows counts and percentages

2. ✅ **`train_test_split.png`**
   - Pie chart of train/validation split percentages
   - Bar chart showing class distribution in each split

3. ✅ **`feature_statistics.png`**
   - Histograms of HRV and GSR distributions
   - Statistical summaries

4. **`confusion_matrix.png`** (Generated when model is evaluated)
   - Confusion matrix heatmap
   - True positives, false positives, false negatives, true negatives

5. **`roc_curve.png`** (Generated when model is evaluated)
   - ROC curve with AUC score
   - Shows classifier performance across thresholds

6. **`metrics_summary.png`** (Generated when model is evaluated)
   - Bar chart of all metrics (Accuracy, Precision, Recall, F1, AUC)

7. **`auc_roc_score.png`** (Generated when model is evaluated)
   - Detailed AUC-ROC visualization

#### **Reports (JSON)**
1. ✅ **`dataset_analysis_report.json`**
   - Complete dataset statistics
   - Feature information
   - Class distributions
   - Train/test split details

2. **`evaluation_report_<model_name>.json`** (Generated when model is evaluated)
   - Model metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
   - Number of samples evaluated
   - Backbone and fusion information

---

## 📈 MULTIMODAL ARCHITECTURE DETAILS

### Vision Modality
- **Input:** Facial images (224×224 pixels)
- **Backbone:** Vision Transformer (ViT) or MobileNet V3
- **Features:** Extracted from pretrained models

### Physiological Modality
- **Features:** HRV, GSR (2 features)
- **Processing:** MLP (Multi-Layer Perceptron)
- **Normalization:** Applied before fusion

### Fusion Strategy
- **Method:** Concatenation or late fusion
- **Output:** Binary classification (0=Safe, 1=Threat)

### Sensitive Attribute
- **Attribute:** Scar presence (0=No scar, 1=Scar)
- **Purpose:** Fairness evaluation
- **Methods:** Demographic parity, equalized odds

---

## 🚀 QUICK START COMMANDS

### Generate Dataset Analysis Only
```bash
python run_comprehensive_analysis.py
```
✅ Generates: Visualizations + Dataset analysis report

### Generate Dataset + Model Evaluation
```bash
python run_comprehensive_analysis.py --checkpoint outputs/checkpoints/model_name.pt
```
✅ Generates: All visualizations + Metrics + Evaluation report

### Available Checkpoint Models
```
outputs/checkpoints/
├── counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.pt
├── counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_best.pt
├── baseline_best.pt
└── [other models...]
```

---

## 📋 DATASET PREPROCESSING APPLIED

### Steps Performed
1. ✅ **Load CSV** - Read multimodal_10k_unbiased.csv
2. ✅ **Type Conversion** - Convert threat/scar to integers
3. ✅ **Missing Value Handling** - Fill physiology NaNs with median
4. ✅ **Required Field Validation** - Drop rows missing: image_path, scar, threat, or physiology
5. ✅ **Data Standardization** - Normalize features for model input

### Result
- **Samples retained:** 10,000 (100%)
- **No data loss** - All samples passed quality checks

---

## 🎯 KEY INSIGHTS

### Dataset Balance
- **Threat:** Imbalanced (63.77% Safe vs 36.23% Threat)
  - Action: Use weighted loss or class weights in training
- **Scar:** Well-balanced (49.82% vs 50.18%)
  - Ideal for fairness evaluation

### Data Quality
- **Missing values:** Only mask_path (optional, for visualization)
- **Data types:** Correctly inferred
- **Unique values:** Sufficient diversity in physiology features

### Multimodal Representation
- **Vision:** 10,000 unique facial images
- **Physiology:** 2 features per sample (HRV, GSR)
- **Metadata:** 15 different subjects (good diversity)

---

## 📊 HOW TO INTERPRET VISUALIZATIONS

### 1. Class Distribution Before/After
- **Use:** Verify data quality and preprocessing effectiveness
- **Read:** Bar heights show sample counts; percentages indicate class proportions
- **Action:** If significant drops occur → investigate missing data

### 2. Train/Test Split
- **Use:** Confirm proper data partitioning
- **Read:** Pie chart shows percentage split; bar chart confirms class balance maintained
- **Action:** Verify both train and validation have similar class distributions

### 3. Feature Statistics
- **Use:** Understand feature distributions and potential outliers
- **Read:** Histogram shapes indicate if features are normally distributed
- **Action:** Look for bimodal distributions or heavy outliers

### 4. Confusion Matrix (Model Evaluation)
- **Use:** Understand which classes are confused
- **Diagonal:** Correct predictions
- **Off-diagonal:** Misclassifications
- **Action:** High off-diagonal values → model struggles with that class

### 5. ROC Curve (Model Evaluation)
- **Use:** Evaluate classifier performance across thresholds
- **AUC Score:** Higher is better (0.5 = random, 1.0 = perfect)
- **Action:** Curve closer to top-left → better classifier

### 6. Metrics Summary (Model Evaluation)
- **Use:** Compare all metrics at once
- **Target:** Maximize all metrics (ideally >0.85)
- **Trade-off:** Often precision vs recall trade-off exists

---

## 🔍 FURTHER ANALYSIS OPTIONS

### 1. Fairness Evaluation (Scar Bias)
Check if model predictions are fair across scar groups:
```bash
python src/eval_fairness.py --checkpoint <model_path>
```

### 2. Feature Importance
Understand which features matter most:
- SHAP values
- Gradient-based attribution
- Ablation studies

### 3. Subject-wise Analysis
Verify model performance across different subjects:
- Per-subject metrics
- Variance across subjects

### 4. Threshold Optimization
Find optimal decision threshold for your use case:
- ROC curve analysis
- Cost-benefit analysis

---

## ✅ CHECKLIST FOR THESIS SUBMISSION

- [ ] Dataset analysis report generated ✅
- [ ] Feature names documented ✅
- [ ] Number of samples confirmed (10,000) ✅
- [ ] Class distribution visualized ✅
- [ ] Train/test split percentages verified (80%/20%) ✅
- [ ] Before/after preprocessing comparison ✅
- [ ] Model metrics calculated (Accuracy, Precision, Recall, F1, AUC)
- [ ] Visualizations created for all metrics
- [ ] Fairness metrics evaluated (DP gap, EO gap)
- [ ] Results documented in thesis

---

**Generated:** February 6, 2026  
**Dataset:** multimodal_10k_unbiased.csv  
**Status:** READY FOR ANALYSIS ✅
