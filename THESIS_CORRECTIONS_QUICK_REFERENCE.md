# ⚡ THESIS CORRECTIONS - Quick Reference

**Status**: 5 issues identified, 4 are documentation-only fixes

---

## 🔴 Issue #1: CF Loss Formula
**Status**: ✅ CODE IS CORRECT, THESIS NEEDS UPDATE

**Current Thesis**:
- Abstract: `E[|f(x) - f(x_cf)|]`
- Chapter 3: Jensen-Shannon divergence

**What Code Actually Uses**:
```python
def js_divergence(p, q):
    """JS divergence on probability distributions"""
    m = 0.5 * (p + q)
    return 0.5 * (KL(p||m) + KL(q||m))
```

**Fix**:
Update **Abstract** to match Chapter 3:
```
OLD: E[|f(x) - f(x_cf)|]
NEW: E[JS(p(x) || p(x_cf))]
```

**Why**: JS divergence is more principled, symmetric, and numerically stable.

---

## 🔴 Issue #2: Inference Latency

**Status**: ❌ DOCS WRONG, ACTUAL METRICS FOUND

**Current Docs**:
- "8.9ms–16.3ms" (WHERE DID THIS COME FROM?)
- Table 3.1: "~4.1ms"

**Actual Measured Values** (from `edge_benchmark.py`):
```json
FP32:  4.36ms (mean), 4.18ms (p50), 5.36ms (p95)
QDyn:  4.70ms (mean), 4.56ms (p50), 5.42ms (p95)
```

**File Location**:
```
outputs/results/edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_fp32.json
outputs/results/edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_qdyn.json
```

**Fix**:
Replace all references with:
```
"Inference latency on CPU (single-sample, 4 threads):
 - FP32:  4.36ms ± 1.2ms  (229 FPS)
 - QDyn:  4.70ms ± 0.9ms  (213 FPS)"
```

**Document Setup**:
```
Benchmark Configuration:
- Device: CPU (4 threads)
- Batch Size: 1
- Iterations: 200
- Input: 256×256 RGB + 2D physiology vector
- Measurement: Wall-clock time (time.perf_counter)
```

---

## 🔴 Issue #3: Quantization Speedup

**Status**: ⚠️ MISLEADING, NEED HONEST DISCUSSION

**Current Claim**: 
- "45% speedup from quantization"

**Reality**:
- FP32: 4.36ms
- QDyn: 4.70ms
- **Result: 8% SLOWER, not faster**

**Why**:
```
QDyn overhead on CPU with batch=1 > computation savings
Actual costs:
  float32 → int8 cast    (overhead)
  int8 GEMM              (benefit)
  int8 → float32 cast    (overhead)
  
For batch_size=1, casting dominates.
```

**Fix**:
```markdown
## Quantization Analysis

"Dynamic quantization reduces model size by 20% (4.6MB → 3.7MB)
but does not improve single-sample CPU inference latency:

| Configuration | Latency | Model Size | FPS |
|---------------|---------|------------|-----|
| FP32          | 4.36ms  | 4.6MB      | 229 |
| QDyn INT8     | 4.70ms  | 3.7MB      | 213 |

The 8% latency overhead stems from runtime casting operations
dominating computation time for batch_size=1.

**Benefits of quantization appear with**:
- Large batch sizes (N >> 1)
- GPU inference (specialized INT8 hardware)
- Mobile deployment (fixed-point accelerators)
- Memory-bound operations (4x bandwidth reduction matters)

For production, we recommend:
- CPU servers: Use FP32 (larger batches naturally benefit from quantization)
- Edge devices: Deploy quantized model (size matters)
- Mobile GPUs: Try quantization (specialized hardware support)"
```

---

## 🟡 Issue #4: Design C (Scar-Suppressed)

**Status**: ❌ NOT IMPLEMENTED, REMOVE OR BUILD

**Code Search Result**:
```python
# In src/models.py
p.add_argument("--fusion", type=str, choices=["cgf", "concat"])
# Only A (concat) and B (cgf) available
# No Design C option anywhere
```

**Option A - REMOVE from thesis** (Recommended):
```markdown
### Model Architectures

We implement two model designs:

**Design A (Baseline - CONCAT)**
- Simple concatenation of vision and physiology features
- Standard multimodal fusion baseline

**Design B (Causal Gated Fusion - CGF)** ⭐ Innovation
- Learnable gate that weights vision vs physiology
- Scar focus computation suppresses scar influence
```

**Option B - IMPLEMENT if needed**:
```python
# Design C: Scar-Suppressed
class ScarSuppressedFusion(nn.Module):
    """Force model to ignore scar region during forward pass."""
    def forward(self, v, p, fmap, mask):
        # Zero out activations in scar region
        scar_mask = 1 - mask
        fmap_suppressed = fmap * scar_mask
        # ... rest of fusion
```

**Recommendation**: Choose Option A (remove) since Design C wasn't critical to results.

---

## 🟡 Issue #5: INT8 vs Dynamic Quantization

**Status**: ✅ CODE CORRECT, TERMINOLOGY UNCLEAR

**Current Docs**:
- "INT8 quantization"

**Code Actually Uses**:
```python
torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

**Fix**:
```markdown
## Quantization Method

We employ PyTorch's dynamic quantization to 8-bit integers (qint8).
Dynamic quantization:
- Does not require calibration data
- Converts linear layers at export time
- Uses 8-bit signed integer representation (qint8)
- Keeps activations in float32 (numerically stable)

This differs from static INT8 quantization, which would require
calibration and quantize both weights and activations.
```

---

## 📋 Action Checklist

### Must Fix (Critical)
- [ ] **Abstract**: Update CF loss from `E[|f(x) - f(x_cf)|]` to `E[JS(...)]`
- [ ] **Latency**: Replace "8.9-16.3ms" with actual "4.36ms (FP32) / 4.70ms (QDyn)"
- [ ] **Quantization**: Remove false "45% speedup" claim
- [ ] **Design C**: Remove from thesis or implement

### Should Fix (Important)
- [ ] **Quantization section**: Explain why INT8 is slower on CPU
- [ ] **Benchmark setup**: Document CPU/batch_size/iterations clearly
- [ ] **Terminology**: Use "dynamic quantization" not just "INT8"

### Nice to Have (Polish)
- [ ] Add percentile latency (p50, p95) to results
- [ ] Include throughput (FPS) metrics
- [ ] Add statistical error bars
- [ ] Document why batch=1 for benchmarking

---

## 📍 File References

**Code to Review**:
```
src/train_cgf_fair.py (Line 135-142)    → JS divergence implementation
src/edge_benchmark.py (Line 150-162)    → Latency measurement code
src/quantize_export.py (Line 74-78)     → Quantization implementation
src/models.py (Line 169-172)            → Design definitions
```

**Actual Results**:
```
outputs/results/edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_fp32.json
outputs/results/edge_counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_best_qdyn.json
outputs/reports/train_counterfactual_multimodal_10k_unbiased_mobilenet_v3_small.json
```

---

## ✅ Summary

| Issue | Status | Fix Type | Priority |
|-------|--------|----------|----------|
| CF Loss | ✅ Code correct | Update abstract | P1 |
| Latency | ❌ Docs wrong | Replace numbers | P1 |
| Quantization Speedup | ⚠️ False claim | Explain overhead | P2 |
| Design C | ❌ Missing | Remove or implement | P2 |
| INT8 terminology | ✅ Code correct | Clarify dynamic | P3 |

**Time to fix**: ~2-3 hours for a complete thesis update

**Complexity**: Low - mostly text updates, no code changes needed

**Impact**: High - ensures thesis matches implementation and honest about results

---

*Quick reference completed: February 27, 2026*
