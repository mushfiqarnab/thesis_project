# 📊 Results Folder - Quick Reference

**Total Files**: 26 JSON files  
**Total Size**: ~150 KB  
**Purpose**: Benchmark and fairness evaluation results

---

## 🎯 Why 26 Files?

You tested **multiple configurations**:

```
Model Variants:
├─ Design A (CONCAT baseline)
├─ Design B (CGF innovation)
└─ Legacy ViT (older architecture)

Datasets:
├─ Original multimodal
└─ 10k unbiased (balanced)

Optimizations:
├─ Full (no compression)
├─ Pruned 30%
└─ Pruned + Repaired

Quantization:
├─ FP32 (full precision)
└─ QDyn (INT8 dynamic)

Evaluation:
├─ Edge benchmarking (latency, throughput, size)
└─ Fairness analysis (DP gap, EO gap, CF stability)

Result: Multiple combinations × 2 evaluation types = 26 files
```

---

## 📁 File Categories

### Category A: Edge Benchmarking (16 files)
Measure: Inference latency, throughput, model size on CPU

**Naming**: `edge_[model]_[dataset]_[optimization]_[quantization].json`

```
1. Baseline Comparison (1)
   └─ edge_baseline_mobilenet_v3_small_concat_best_fp32.json

2. CGF Models (3 variants)
   ├─ Original multimodal
   ├─ 10k unbiased
   └─ + pruned

3. CGF Quantized (3 variants × FP32/QDyn)
   ├─ Original FP32
   ├─ Original QDyn
   ├─ 10k unbiased FP32/QDyn
   └─ Pruned FP32/QDyn

4. Pruned Models (2 FP32, 2 QDyn)
   ├─ Pruned only
   └─ Pruned + repaired

5. Design A Variants (2)
   ├─ FP32
   └─ QDyn
```

### Category B: Fairness Analysis (10 files)
Measure: DP gap, EO gap, counterfactual fairness stability

**Naming**: `fairness_[model_type]_[variant].json`

```
1. Legacy ViT (2)
   ├─ Baseline (no fairness training)
   └─ With fairness training

2. Design A (CONCAT) (2)
   ├─ Unfair baseline (DP=0.042)
   └─ With fairness training

3. Design B (CGF) (6)
   ├─ Original FP32
   ├─ 10k unbiased FP32
   ├─ + pruning
   ├─ + pruning + repair
   ├─ + pruning + quantized + repair
   └─ Direct A vs B comparison
```

---

## 📈 Key Findings Across Files

### Latency Progression
```
Baseline (CONCAT):                4.15ms
CGF Full:                         4.36ms
CGF Pruned 30%:                   4.10ms ← Better!
CGF Pruned + Quantized:           4.50ms (overhead)
CGF Pruned + Repaired + QDyn:     4.50ms

Insight: Pruning helps on CPU (reduces compute), 
         quantization adds overhead on single-sample CPU inference
```

### Fairness Improvement
```
Design A (CONCAT) baseline:       DP gap = 0.042
Design B (CGF) trained fairly:    DP gap = 0.012 (71% reduction!)

After pruning:                    DP gap = 0.012 (preserved)
After repair:                     DP gap = 0.010 (even better)

Insight: CGF learning to ignore scars works!
         Compression doesn't hurt fairness!
```

### Size Reduction
```
FP32 (full precision):            4.6 MB (1,162K params)
Pruned 30%:                       3.2 MB (817K params)
Quantized:                        3.7 MB (dynamic INT8)
Pruned + Quantized:               2.9 MB (combined)

Insight: Combined optimization = 35% size reduction
         while maintaining fairness
```

---

## 🔍 File Naming Patterns

### Edge Benchmark Pattern
```
edge_[FUSION]_[ARCHITECTURE]_[DATASET]_[OPTIMIZATION]_[QUANTIZATION].json

Example: edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_pruned30_qdyn.json
├─ edge_                    ← This is a benchmark file
├─ counterfactual_cgf_js    ← Model: CGF with JS-divergence loss
├─ mobilenet_v3_small       ← Vision backbone
├─ multimodal_10k_unbiased  ← Dataset: balanced 10k samples
├─ pruned30                 ← 30% pruning applied
└─ qdyn                     ← Dynamic INT8 quantization
```

### Fairness Analysis Pattern
```
fairness_[MODEL_TYPE]_[FUSION]_[CHECKPOINT].json

Example: fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_pruned30_repaired.json
├─ fairness_                         ← Fairness metrics
├─ current_multimodal                ← Modern multimodal architecture
├─ counterfactual_cgf_js             ← CGF with JS loss
├─ mobilenet_v3_small                ← Backbone
├─ multimodal_10k_unbiased           ← Dataset
├─ pruned30                          ← Pruning level
└─ repaired                          ← Fine-tuned after compression
```

---

## 💡 What Each File Contains

### Benchmark Files (edge_*.json)
```json
{
  "ckpt": "path/to/model.pt",
  "model_family": "current_multimodal",
  "ckpt_size_mb": 4.587,
  "params": 1162019,
  "latency_ms_mean": 4.359,
  "latency_ms_p50": 4.179,      ← Median latency
  "latency_ms_p95": 5.361,      ← 95th percentile
  "throughput_fps_est": 229.4,
  "rss_mb_before": 514.7,       ← Memory before
  "rss_mb_after": 504.4,        ← Memory after
  "rss_mb_delta": -10.3         ← Memory freed
}
```

### Fairness Files (fairness_*.json)
```json
{
  "model": "CGF",
  "fusion": "cgf",
  "dataset": "multimodal_10k_unbiased",
  "checkpoint": "...best.pt",
  "metrics": {
    "accuracy": 0.7785,
    "precision": 0.7654,
    "recall": 0.7789,
    "f1": 0.7721,
    "auc_roc": 0.8456
  },
  "fairness": {
    "dp_gap": 0.0054,           ← Demographic parity
    "eo_gap": 0.0035,           ← Equalized odds
    "cf_stability": 0.042,      ← Counterfactual
    "results_by_group": {
      "scar_0": {...},
      "scar_1": {...}
    }
  }
}
```

---

## ✅ Do You Need All 26?

### Keep Them Because:
- **Reproducibility**: Document every configuration tested
- **Thesis Evidence**: Reference specific files for different claims
- **Comparison**: Show why CGF + pruning + repair is best
- **Version Control**: Git tracks each result separately
- **No Storage Cost**: Only 150 KB total

### Could Delete Later:
- **After submission**: Archive instead (compress to .zip)
- **Duplicates**: Files with identical results could be deduplicated
- **Legacy**: Old ViT results if not mentioned in thesis

---

## 📋 For Your Thesis

### Results Section Reference
```
"Table 3.1 summarizes inference latency across model variants.
Detailed results stored in outputs/results/:

Edge benchmarking:
  - Design A baseline: edge_baseline_mobilenet_v3_small_concat_best_fp32.json
  - Design B (CGF): edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_fp32.json
  - Optimized variant: edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_pruned30_qdyn.json

Fairness validation:
  - Baseline fairness: fairness_concat_mobilenet_v3_small_baseline_mobilenet_v3_small_concat_best.json
  - CGF fairness: fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best.json
  - Post-compression: fairness_current_multimodal_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_pruned30_repaired.json"
```

---

## 🎯 Bottom Line

| Aspect | Answer |
|--------|--------|
| **Why 26 files?** | Comprehensive evaluation of design choices, optimizations, and fairness |
| **Are they needed?** | Yes - each documents one important experiment |
| **Storage issue?** | No - 150 KB total |
| **Delete any?** | No - keep all for thesis submission |
| **Organize them?** | Optional - create summary file or subfolders |

**In short**: You did thorough experiments and saved the results. Keep everything!

---

*Quick reference completed: February 27, 2026*
