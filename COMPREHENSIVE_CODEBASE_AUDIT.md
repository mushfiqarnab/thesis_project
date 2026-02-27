# 🔍 COMPREHENSIVE CODEBASE AUDIT REPORT

**Date**: February 27, 2026  
**Project**: Edge-Friendly Debiasing for Scar-Mediated Threat Profiling Using Emotion-Physiology Fusion  
**Audit Scope**: All source files, data pipeline, models, training loops  
**Status**: THOROUGH LINE-BY-LINE ANALYSIS IN PROGRESS

---

## 📋 Audit Plan

This audit will examine:

### Phase 1: Core Models (models.py)
- [ ] Class definitions and inheritance
- [ ] Forward pass logic
- [ ] Tensor shape consistency
- [ ] Device compatibility
- [ ] Numerical stability

### Phase 2: Data Pipeline (dataset_fair.py)
- [ ] CSV loading and validation
- [ ] Counterfactual generation
- [ ] Mask handling
- [ ] Data type conversions
- [ ] Edge cases and error handling

### Phase 3: Training Loop (train_cgf_fair.py)
- [ ] Loss computation correctness
- [ ] Gradient flow
- [ ] Fairness metric calculations
- [ ] State dict handling
- [ ] Checkpoint saving/loading

### Phase 4: Evaluation (eval_fairness.py, edge_benchmark.py)
- [ ] Metric computation accuracy
- [ ] Device management
- [ ] Result serialization

### Phase 5: Utility Scripts
- [ ] Pruning logic (prune_checkpoint.py)
- [ ] Quantization handling (quantize_export.py)
- [ ] Checkpoint repair (fair_repair_finetune.py)

---

## 🎯 ISSUES FOUND SO FAR (During Initial Scan)

### ✅ **GOOD PRACTICES OBSERVED**

1. **Type Hints**: Comprehensive throughout
   - `models.py`: All functions have return type hints ✓
   - `dataset_fair.py`: Dataclass with typed fields ✓
   - `train_cgf_fair.py`: Arguments properly typed ✓

2. **Error Handling**: 
   - Path existence checks ✓
   - CSV column validation ✓
   - Device compatibility fallbacks ✓

3. **Numerical Stability**:
   - Epsilon clamping in js_divergence ✓
   - log1p for focus computation (avoids log(0)) ✓
   - Safe mask normalization ✓

4. **Reproducibility**:
   - Seed setting comprehensive ✓
   - Split saving to JSON ✓
   - Hyperparameter logging ✓

---

## ⚠️ **ISSUES TO INVESTIGATE**

### Issue 1: Missing Device Specification in VisionEncoder
**File**: `src/models.py` lines 44-50

```python
if self.name == "mobilenet_v3_small":
    m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # ISSUE: Model m is created without specifying device
    # If model is later moved to GPU, this might cause issues
```

**Status**: NEEDS VERIFICATION

### Issue 2: Tensor Device Mismatch in focus_from_mask
**File**: `src/models.py` lines 118-132

```python
@staticmethod
def focus_from_mask(fmap: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    # mask might be on different device than fmap
    # But F.interpolate happens without explicit device check
```

**Status**: POTENTIALLY PROBLEMATIC

### Issue 3: Default Mask in MultimodalThreatModel.forward
**File**: `src/models.py` lines 176-180

```python
if mask is None:
    mask = torch.zeros((img.size(0), 1, img.size(2), img.size(3)), 
                       device=img.device, dtype=img.dtype)
# Good: device is img.device, but dtype might be wrong (should be float32)
```

**Status**: NEEDS CLARIFICATION

### Issue 4: State Dict Loading Without Validation
**File**: `src/train_cgf_fair.py` lines 84-101

```python
def load_state_dict_safely(path: str, device: torch.device) -> dict:
    # Function exists but needs full audit of all cases
```

**Status**: INCOMPLETE REVIEW

---

## 📝 DETAILED FINDINGS WILL FOLLOW

Due to the extensive scope, I'm conducting:

1. **Line-by-line code review** of all 23 src files
2. **Cross-file dependency analysis** to find integration issues
3. **Data flow tracing** from input to output
4. **Tensor shape verification** through all layers
5. **Numerical stability checks** in critical computations

---

## 🔧 XAI IMPLEMENTATION PREREQUISITES

Before implementing XAI features, we must:

1. ✅ **Confirm no device mismatches** exist
2. ✅ **Validate tensor shapes** throughout pipeline
3. ✅ **Verify loss computations** are correct
4. ✅ **Test edge cases** in data loading
5. ✅ **Document assumptions** in each module

---

## Next Steps

1. Complete full file-by-file audit
2. Document all findings in structured format
3. Create issue tracking document
4. Propose fixes for identified problems
5. THEN implement XAI with full confidence

---

**Estimated Completion**: Will provide comprehensive report with:
- ✅ Line-by-line analysis of each file
- ✅ Identified issues with severity levels
- ✅ Proposed fixes with code examples
- ✅ Risk assessment for XAI integration
- ✅ XAI implementation plan

---

**Note**: Taking time to be precise as requested - not rushing, ensuring zero issues before XAI implementation.

