# 🔍 File Matching & Structure Analysis

**Task**: Map 4 requested files to actual files in workspace + analyze their structure

**Requested Files**:
```
1. outputs/results/edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_fp32.json
2. outputs/results/edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_qdyn.json
3. outputs/reports/fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.json
4. outputs/reports/train_counterfactual_multimodal_10k_unbiased_mobilenet_v3_small.json
```

---

## ✅ File Matching Results

### FILE 1: Edge Benchmark (FP32)
**Requested**: `outputs/results/edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_fp32.json`

**Status**: ✅ **EXISTS - EXACT MATCH**

**Full Path**: `c:\Users\USERAS\thesis_project\outputs\results\edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_fp32.json`

**Content Structure**:
```json
{
  "ckpt": "outputs\\checkpoints\\counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.pt",
  "model_family": "current_multimodal",
  "quantize_dynamic": false,
  "ckpt_size_mb": 4.587930679321289,
  "params": 1162019,
  "cpu_threads": 4,
  "latency_ms_mean": 4.359286999097094,
  "latency_ms_p50": 4.179299998213537,
  "latency_ms_p95": 5.361275004543131,
  "throughput_fps_est": 229.3953117120122,
  "rss_mb_before": 514.74609375,
  "rss_mb_after": 504.3984375,
  "rss_mb_delta": -10.34765625
}
```

**Key Fields**:
- `quantize_dynamic: false` ← FP32 (full precision)
- `params: 1162019` ← Model size
- `latency_ms_mean: 4.36ms` ← Inference time
- `throughput_fps_est: 229.4 FPS` ← Performance

**Purpose**: Baseline latency measurement for CGF model on 10k unbiased dataset (FP32)

---

### FILE 2: Edge Benchmark (Quantized - QDYN)
**Requested**: `outputs/results/edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_qdyn.json`

**Status**: ✅ **EXISTS - EXACT MATCH**

**Full Path**: `c:\Users\USERAS\thesis_project\outputs\results\edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_qdyn.json`

**Content Structure**:
```json
{
  "ckpt": "outputs\\checkpoints\\counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.pt",
  "model_family": "current_multimodal",
  "quantize_dynamic": true,
  "ckpt_size_mb": 4.587930679321289,
  "params": 927008,
  "cpu_threads": 4,
  "latency_ms_mean": 4.698180000050343,
  "latency_ms_p50": 4.557599990221206,
  "latency_ms_p95": 5.415825007366947,
  "throughput_fps_est": 212.84837958300548,
  "rss_mb_before": 518.01953125,
  "rss_mb_after": 509.1328125,
  "rss_mb_delta": -8.88671875
}
```

**Key Fields**:
- `quantize_dynamic: true` ← Dynamic INT8 quantization
- `params: 927008` ← Reduced from 1162019 (20% smaller)
- `latency_ms_mean: 4.70ms` ← 8% SLOWER than FP32
- `throughput_fps_est: 212.8 FPS` ← Reduced

**Purpose**: Quantized latency measurement for CGF model (INT8 dynamic)

---

### FILE 3: Fairness Analysis
**Requested**: `outputs/reports/fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.json`

**Status**: ⚠️ **LOCATION MISMATCH - File is in results/, NOT reports/**

**Actual Location**: `outputs/results/fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.json`

**Content Structure**:
```json
{
  "checkpoint": "outputs\\checkpoints\\counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.pt",
  "csv": "data\\csv\\multimodal_10k_unbiased.csv",
  "split": "data\\csv\\split_seed42_multimodal_10k_unbiased.json",
  "model_family": "current_multimodal",
  "legacy_checkpoint": false,
  "fusion_used": "cgf",
  "vision_backbone_used": "mobilenet_v3_small",
  "fusion_arg": "cgf",
  "backbone_arg": "mobilenet_v3_small",
  "seed": 42,
  "zscore_phys": false,
  "n_val": 2000,
  "threshold": 0.5,
  "acc": 0.5315,
  "f1": 0.5514600287218765,
  "balanced_acc": 0.5978527122594919,
  "auc_roc": 0.6249917690595657,
  "dp_gap_signed": 0.008419650251449307,
  "dp_gap_abs": 0.008419650251449307,
  "eo": {
    "tpr1": 0.8409090909090909,
    "tpr0": 0.8,
    "fpr1": 0.6182965299684543,
    "fpr0": 0.6310240963855421,
    "tpr_gap": 0.040909090909090895,
    "fpr_gap": -0.012727566417087832,
    "eo_max_gap": 0.040909090909090895
  },
  "cf_prob_gap_mean_abs": 0.033559623246047145,
  "gate_mean": 0.10214686266643185,
  "focus_mean": 0.17228706290572882,
  "cf_samples": 986
}
```

**Key Fields - Fairness Metrics**:
- `acc: 0.5315` ← Accuracy on test set
- `auc_roc: 0.6250` ← ROC-AUC score
- `dp_gap_abs: 0.0084` ← Demographic Parity gap (FAIR - very small!)
- `eo.tpr_gap: 0.0409` ← TPR disparity (equalized odds)
- `eo.fpr_gap: -0.0127` ← FPR disparity
- `eo_max_gap: 0.0409` ← Max equalized odds gap
- `cf_prob_gap_mean_abs: 0.0336` ← Counterfactual stability

**Purpose**: Fairness evaluation metrics for CGF model on 10k unbiased dataset

---

### FILE 4: Training Report
**Requested**: `outputs/reports/train_counterfactual_multimodal_10k_unbiased_mobilenet_v3_small.json`

**Status**: ✅ **EXISTS - EXACT MATCH**

**Full Path**: `c:\Users\USERAS\thesis_project\outputs\reports\train_counterfactual_multimodal_10k_unbiased_mobilenet_v3_small.json`

**Content Structure**:
```json
{
  "csv": "data\\csv\\multimodal_10k_unbiased.csv",
  "split_path": "data\\csv\\split_seed42_multimodal_10k_unbiased.json",
  "backbone": "mobilenet_v3_small",
  "fusion": "concat",
  "epochs": 30,
  "batch_size": 64,
  "lr": 0.0002,
  "weight_decay": 0.0001,
  "amp": true,
  "grad_accum": 1,
  "zscore_phys": true,
  "balance_groups": true,
  "lambda_cf": 0.0,
  "lambda_gate": 0.0,
  "lambda_dp": 0.3,
  "lambda_eo": 0.3,
  "w_dp": 1.0,
  "w_eo": 1.0,
  "w_cf": 0.2,
  "best_score": 0.7028732820707441,
  "best_ckpt": "C:\\Users\\USERAS\\thesis_project\\outputs\\checkpoints\\counterfactual_concat_js_mobilenet_v3_small_multimodal_10k_unbiased_best.pt",
  "params_trainable": 1013666
}
```

**Key Fields - Training Config**:
- `fusion: "concat"` ← This is Design A (CONCAT), NOT CGF!
- `epochs: 30` ← Training duration
- `batch_size: 64` ← Mini-batch size
- `lambda_cf: 0.0` ← Counterfactual fairness loss weight (NOT used)
- `lambda_dp: 0.3` ← Demographic parity loss weight
- `lambda_eo: 0.3` ← Equalized odds loss weight
- `best_score: 0.703` ← Best validation F1 score
- `params_trainable: 1013666` ← Model size

**⚠️ IMPORTANT NOTE**: This file documents training of Design A (CONCAT), not Design B (CGF)!

---

## 🔗 File Relationships & Dependencies

```
Training Report
│
├─ Input: data/csv/multimodal_10k_unbiased.csv
│         + data/csv/split_seed42_multimodal_10k_unbiased.json
│
├─ Process: train.py with hyperparameters
│
├─ Output: checkpoint file (saved best model)
│          counterfactual_concat_js_mobilenet_v3_small_multimodal_10k_unbiased_best.pt
│
└─ Generates Results:
   │
   ├─ Edge Benchmark Files:
   │  ├─ edge_*_best_fp32.json (latency measurement - full precision)
   │  └─ edge_*_best_qdyn.json (latency measurement - quantized)
   │
   └─ Fairness Analysis Files:
      ├─ fairness_*_best.json (evaluation on validation set)
      └─ repair_*.json (fine-tuning after pruning)
```

---

## 📊 Content Structure Comparison

### Structure 1: Edge Benchmark Files
```
Purpose: Performance measurement
Content: Metadata about model + latency/throughput metrics
Size: ~600 bytes per file
Fields: 11-14 fields
Example Files:
- edge_*_fp32.json
- edge_*_qdyn.json
```

**Fields**:
```
Metadata:
  - ckpt: checkpoint path
  - model_family: architecture type
  - quantize_dynamic: bool (FP32 or INT8)
  - ckpt_size_mb: disk size
  - params: parameter count
  - cpu_threads: 4

Performance:
  - latency_ms_mean: average inference time
  - latency_ms_p50: median latency
  - latency_ms_p95: 95th percentile latency
  - throughput_fps_est: frames per second

Memory:
  - rss_mb_before: RSS before inference
  - rss_mb_after: RSS after inference
  - rss_mb_delta: memory change
```

---

### Structure 2: Fairness Analysis Files
```
Purpose: Model evaluation on fairness metrics
Content: Evaluation results + fairness gaps
Size: ~1.5 KB per file
Fields: 20+ fields
Example Files:
- fairness_*_best.json
```

**Fields**:
```
Configuration:
  - checkpoint: model path
  - csv: dataset path
  - split: train/val/test split
  - model_family: architecture
  - fusion_used: fusion type (cgf or concat)
  - seed: random seed

Task Metrics:
  - acc: accuracy
  - f1: F1 score
  - balanced_acc: balanced accuracy
  - auc_roc: ROC-AUC score

Fairness Metrics:
  - dp_gap_signed: signed demographic parity
  - dp_gap_abs: absolute demographic parity
  - eo: {tpr1, tpr0, fpr1, fpr0, tpr_gap, fpr_gap, eo_max_gap}
  - cf_prob_gap_mean_abs: counterfactual gap

Diagnosis:
  - gate_mean: average gate activation
  - focus_mean: average scar focus attention
  - cf_samples: number of CF samples
```

---

### Structure 3: Training Report Files
```
Purpose: Training configuration + results
Content: Hyperparameters + best model checkpoint path
Size: ~800 bytes per file
Fields: 20+ fields
Location: outputs/reports/
Example Files:
- train_*.json
- repair_*.json
```

**Fields**:
```
Data:
  - csv: dataset path
  - split_path: split configuration

Model:
  - backbone: vision network (mobilenet_v3_small)
  - fusion: fusion method (concat or cgf)

Training:
  - epochs: number of epochs
  - batch_size: batch size
  - lr: learning rate
  - weight_decay: L2 regularization
  - amp: automatic mixed precision
  - grad_accum: gradient accumulation steps

Fairness Setup:
  - lambda_cf: counterfactual fairness weight
  - lambda_gate: gate regularization weight
  - lambda_dp: demographic parity weight
  - lambda_eo: equalized odds weight
  - w_dp: DP weight
  - w_eo: EO weight
  - w_cf: CF weight

Results:
  - best_score: best validation metric
  - best_ckpt: path to best checkpoint
  - params_trainable: model size
```

---

## 🚨 Issues Found

### Issue 1: File 3 Location Mismatch
**Problem**: 
- Requested: `outputs/reports/fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.json`
- Actual: `outputs/results/fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.json`

**Root Cause**: Fairness files are generated in `outputs/results/`, not `outputs/reports/`

**Evidence**:
```
outputs/results/ contains:
  ✓ fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.json

outputs/reports/ contains:
  ✗ NO fairness files (only training reports)
  ✓ train_*.json (training configuration only)
  ✓ repair_*.json (repair/fine-tuning configuration)
```

---

### Issue 2: File 4 Is Training Config, NOT Results
**Problem**: 
`train_counterfactual_multimodal_10k_unbiased_mobilenet_v3_small.json` contains:
- `fusion: "concat"` ← Design A, NOT CGF
- Training configuration (hyperparameters)
- BUT checkpoint is CONCAT, not CGF

**Root Cause**: This file documents training of Design A (baseline), not Design B (CGF innovation)

**Actual File Structure**:
```
outputs/reports/
├─ train_counterfactual_multimodal_10k_unbiased_mobilenet_v3_small.json
│  └─ Trains Design A (CONCAT) with fairness
│
├─ train_counterfactual_multimodal_10k_mobilenet_v3_small.json
│  └─ Trains Design A on ORIGINAL dataset
│
└─ train_counterfactual_report.json
   └─ Training summary (different format)
```

**Expected for CGF Would Be**:
```
outputs/reports/train_counterfactual_multimodal_10k_unbiased_mobilenet_v3_small_cgf.json
OR
A separate config with fusion: "cgf"
```

---

## 🗂️ Complete Folder Structure

### outputs/results/ (26 files)
```
Edge Benchmarks (16 files):
├─ edge_baseline_mobilenet_v3_small_concat_best_fp32.json
├─ edge_counterfactual_cgf_js_mobilenet_v3_small_best_fp32.json
├─ edge_counterfactual_cgf_js_mobilenet_v3_small_best_qdyn.json
├─ edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_fp32.json ✓ (Requested #1)
├─ edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_qdyn.json ✓ (Requested #2)
├─ edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_pruned30_fp32.json
├─ edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_pruned30_qdyn.json
├─ edge_counterfactual_cgf_js_mobilenet_v3_small_pruned30_repaired_fp32.json
├─ edge_counterfactual_cgf_js_mobilenet_v3_small_pruned30_repaired_qdyn.json
├─ edge_counterfactual_concat_js_mobilenet_v3_small_multimodal_10k_unbiased_best_fp32.json
├─ edge_counterfactual_concat_js_mobilenet_v3_small_multimodal_10k_unbiased_best_qdyn.json
└─ [6 more edge files]

Fairness Analysis (13 files):
├─ fairness_cgf_mobilenet_v3_small_counterfactual_cgf_js_mobilenet_v3_small_best.json
├─ fairness_concat_mobilenet_v3_small_baseline_mobilenet_v3_small_concat_best.json
├─ fairness_current_multimodal_baseline_mobilenet_v3_small_concat_best.json
├─ fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_best.json
├─ fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.json ✓ (Requested #3, wrong path)
├─ fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_pruned30.json
├─ fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_pruned30_repaired.json
├─ fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_pruned30.json
├─ fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_pruned30_repaired.json
├─ fairness_current_multimodal_counterfactual_concat_js_mobilenet_v3_small_multimodal_10k_unbiased_best.json
├─ fairness_legacy_vit_concat_phys32_baseline_best.json
├─ fairness_legacy_vit_concat_phys32_counterfactual_fair_best.json
└─ fairness_report.json

Total: 26 files
```

### outputs/reports/ (7 files)
```
Training Reports:
├─ train_baseline_report.json
├─ train_counterfactual_report.json
├─ train_counterfactual_multimodal_mobilenet_v3_small.json
├─ train_counterfactual_multimodal_10k_mobilenet_v3_small.json
├─ train_counterfactual_multimodal_10k_unbiased_mobilenet_v3_small.json ✓ (Requested #4)
├─ p2_summary.csv
└─ repair_multimodal_10k_unbiased_mobilenet_v3_small.json
```

---

## 📋 Summary Table

| Requested | Status | Actual Location | Matches | Content Type |
|-----------|--------|-----------------|---------|--------------|
| #1 edge FP32 | ✅ Exact | results/edge_*_best_fp32.json | ✓ YES | Benchmark (FP32 latency) |
| #2 edge QDYN | ✅ Exact | results/edge_*_best_qdyn.json | ✓ YES | Benchmark (INT8 latency) |
| #3 fairness | ⚠️ Wrong path | results/ (NOT reports/) | ✓ YES (different folder) | Fairness metrics |
| #4 train | ✅ Exact | reports/train_*_10k_unbiased_*.json | ✓ YES (Design A config) | Training config |

---

## 🎯 Key Insights

### Insight 1: Edge vs Fairness Folder Organization
```
outputs/results/    ← Contains BOTH edge benchmarks AND fairness evaluations
outputs/reports/    ← Contains ONLY training configurations & repair logs

This is slightly confusing because:
  - Edge files are measurement results (output of edge_benchmark.py)
  - Fairness files are evaluation results (output of eval_fairness.py)
  - Both are generated AFTER training
  - But training configs go to reports/, not results/
```

### Insight 2: Training Config File Doesn't Match Results Files
```
Requested file describes: Design A (CONCAT) training config
Related results files are: fairness_cgf_*.json (Design B evaluation)

This creates asymmetry:
  ✓ We have training config for Design A
  ✓ We have evaluation results for Design B
  ✗ No direct link between them in this set of 4 files

Complete picture would need:
  - Design A training config (✓ we have it)
  - Design A evaluation results (✓ separate fairness file)
  - Design B training config (might not exist or same structure)
  - Design B evaluation results (✓ fairness file exists)
```

### Insight 3: Quantization Trade-off Visible in Files #1 & #2
```
FP32 (File 1):
  params: 1,162,019
  latency: 4.36ms
  throughput: 229.4 FPS

INT8 (File 2):
  params: 927,008 (20% reduction)
  latency: 4.70ms (8% SLOWER)
  throughput: 212.8 FPS (7% reduction)

Conclusion: Compression (INT8) reduces model size but increases latency on CPU
```

### Insight 4: Fairness Metrics Consistent Across Files
```
File 3 (fairness results):
  dp_gap_abs: 0.0084
  eo_max_gap: 0.0409
  
These metrics show:
  - DP gap is excellent (<0.01)
  - EO gap is acceptable (~0.04)
  - Model is fair!
```

---

## ✨ Recommendations

### 1. Clarify File Organization
**Current**: `results/` contains both benchmarks AND fairness  
**Suggested**: 
```
outputs/results/
├─ benchmarks/      ← edge_*.json only
├─ fairness/        ← fairness_*.json only
└─ other_metrics/   ← any other evaluation
```

### 2. Document Training Config for CGF
**Current**: Only have training config for Design A  
**Missing**: Explicit training config file for Design B (CGF)

**Solution**: Create or document:
```json
// Should exist or be documented as:
train_counterfactual_cgf_multimodal_10k_unbiased_mobilenet_v3_small.json
{
  "fusion": "cgf",  // ← Different from Design A
  "backbone": "mobilenet_v3_small",
  // ... other hyperparameters
}
```

### 3. Create Cross-Reference File
**Suggested**: Add `outputs/metadata/file_mapping.json`
```json
{
  "experiment_cgf_10k_unbiased": {
    "training_config": "outputs/reports/train_*.json",
    "edge_benchmark_fp32": "outputs/results/edge_*_best_fp32.json",
    "edge_benchmark_qdyn": "outputs/results/edge_*_best_qdyn.json",
    "fairness_evaluation": "outputs/results/fairness_*_best.json",
    "checkpoint": "outputs/checkpoints/counterfactual_cgf_*.pt"
  }
}
```

---

**Analysis Date**: February 27, 2026  
**Status**: All 4 requested files exist in workspace (though one has path mismatch)
