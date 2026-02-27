# 🔬 PROPER SECOND-PASS AUDIT FRAMEWORK
**Approach**: Rigorous academic review with domain expertise  
**Target**: Thesis-defensible XAI implementation  

---

## METHODOLOGY

### Three-Layer Verification

**Layer 1: Mathematical Correctness**
- Review against original papers (line by line)
- Verify formulas, not just claims
- Check edge cases mathematically

**Layer 2: Implementation Fidelity**
- Does code match the math?
- Are there discretization errors?
- How do hyperparameters affect results?

**Layer 3: Domain Appropriateness**
- Does this make sense for your problem?
- Are assumptions valid for face+physiology?
- What could go wrong in practice?

---

## ISSUES REQUIRING DETAILED INVESTIGATION

### Issue A: Integrated Gradients Baseline Selection

**The Original Paper** (Sundararajan et al. 2017):
- Recommends baseline = "input that represents absence of signal"
- For images: could be black image, blur, or random noise
- Choice affects attributions (known issue in literature)

**What You Need To Understand**:
```
For YOUR specific problem (faces + threat detection):

1. BLACK IMAGE (zeros in normalized space)
   - Represents: A completely featureless image
   - Question: Is this "absence of signal"?
   - Answer: Sort of - but normalized means ≠ pixel-space black
   
2. AVERAGE FACE (mean image from your training set)
   - Represents: A neutral/default face
   - Question: Is this meaningful?
   - Answer: Maybe - if model learns relative to this
   
3. GAUSSIAN NOISE
   - Represents: Random texture
   - Question: Is this interpretable?
   - Answer: No - but mathematically clean

4. BLURRED IMAGE
   - Represents: Image with high-frequency removed
   - Question: What frequency cutoff?
   - Answer: Arbitrary - depends on blur kernel
```

**For Your Defense**:
> "We used X baseline because [SPECIFIC REASON FOR YOUR DOMAIN]. 
> We validated against alternatives and found consistent results."

**What To Do**:
1. Pick ONE baseline with justification
2. OR test all three and show they give consistent rankings
3. Report which one in your paper

---

### Issue B: Completeness Axiom Verification

**The Correct Axiom** (from paper):
```
sum_i(IG_i) = f(x) - f(x_baseline)

where IG_i = (x_i - x_baseline_i) * integral_0^1[df/dx_i at step t]
```

**Why My Check Was Wrong**:
- I summed attributions (which have units of "pixel-space change")
- Compared to prediction (which has units of "probability change")
- These don't have the same units!

**The REAL Check Should Be**:
```
# For each feature i, verify:
# partial_f/partial_x_i (at baseline) * (x_i - baseline_i) ≈ IG_i

# But you can't easily compute this without numerical differentiation
# Better approach: Just verify implementation visually
```

**For Your Defense**:
- Don't claim to verify completeness axiom if you can't do it properly
- Instead show: "IG attributions make intuitive sense"
- Example: "Scar region has high attribution when model predicts threat"

---

### Issue C: Which Saliency Aggregation Is Best?

**The Options**:
```python
# 1. Maximum across channels
saliency_max = grad.abs().max(dim=1)[0]

# 2. L2 norm across channels  
saliency_l2 = (grad.abs()**2).sum(dim=1)**0.5

# 3. Mean across channels
saliency_mean = grad.abs().mean(dim=1)

# 4. Keep all channels (no aggregation)
saliency_per_channel = grad.abs()  # (B, 3, H, W)
```

**Which to Use?**
- L2 norm is "most principled" (captures all channels)
- Max is "most interpretable" (shows strongest signal)
- Mean is "balanced"
- Per-channel best for visualization

**For Your Defense**:
Choose ONE based on your use case:
- For paper results: Use L2 norm (standard)
- For visualization: Show per-channel (richer information)
- For ablation: Use all three and compare

---

### Issue D: Gate Mechanism Impact on Gradients

**The Question**: Does IG work well with a gating mechanism?

**The Code**:
```python
class CausalGatedFusion(nn.Module):
    def forward(self, v, p, mask):
        focus = log(inside_mean / overall_mean)
        gate = sigmoid(MLP([p_proj, focus]))
        fused = gate * v + (1-gate) * p
        return fused
```

**Gradient Flow**:
```
d(output)/d(v) = gate * I + d(gate)/d(...) * v
d(output)/d(p) = (1-gate) * I + d(gate)/d(...) * (-p)
```

**Potential Issues**:
1. Gate acts as "soft switch" - could make some features completely invisible
2. If gate ≈ 0 always, vision branch has near-zero gradient
3. IG attributions for vision would be near-zero even if important
4. **Question**: Is this a feature or a bug?

**For Your Defense**:
Analyze this:
```python
# After training, check:
gate_values = model(img, phys, mask)  # Get all gate values
print("Gate statistics:")
print(f"  Mean: {gate_values.gate.mean()}")
print(f"  Std: {gate_values.gate.std()}")
print(f"  Min: {gate_values.gate.min()}")
print(f"  Max: {gate_values.gate.max()}")

# If gate is near 0 or 1 always, that's suspicious
# If gate varies [0.2, 0.8], that's healthy
```

---

### Issue E: XAI Metric Interpretation

**Your Current Metric**:
```python
def compute_scar_influence_score(self, img, phys, mask, scar):
    p_with_scar = model(img, phys, mask=mask)
    p_without_scar = model(img, phys, mask=zeros)
    influence = (p_with_scar - p_without_scar).abs()
```

**What This Actually Tells You**:
```
High influence could mean:
1. ✓ Model correctly uses scar information
2. ✗ Model relies on scar due to confounding
3. ? Model is confused by scar
```

**Better Metric**:
```python
def scar_usage_analysis(self, img, phys, mask, scar, y_true):
    p_with = self.model(img, phys, mask=mask)
    p_without = self.model(img, phys, mask=zeros)
    influence = (p_with - p_without).abs()
    
    # Disaggregate
    mask_scar_present = (scar == 1)
    mask_scar_absent = (scar == 0)
    
    # When should scar matter?
    # Only when it's actually informative for threat
    
    # Return detailed analysis
    return {
        'total_influence': influence.mean(),
        'influence_when_scar_present': influence[mask_scar_present].mean(),
        'influence_when_scar_absent': influence[mask_scar_absent].mean(),
        'correlation_with_accuracy': correlation(influence, accuracy),
    }
```

---

## WHAT NEEDS ACTUAL TESTING

### Test 1: IG Stability Across Runs

```python
# Do you get same attributions with:
# - Different random seeds?
# - Different step counts (50 vs 100)?
# - Different baselines?

result_1 = ig.explain(img, phys, steps=50, seed=42)
result_2 = ig.explain(img, phys, steps=50, seed=43)
result_3 = ig.explain(img, phys, steps=100, seed=42)

# Compute correlation
corr = np.corrcoef(result_1.vision_attribution.flatten(),
                   result_2.vision_attribution.flatten())
print(f"Correlation between runs: {corr}")

# Should be very high (>0.95)
```

### Test 2: Model Gradient Sanity

```python
# Manually check gradients make sense
img_perturbation = torch.randn_like(img) * 0.01
output_change = (
    self.model(img + img_perturbation, phys, mask).logits -
    self.model(img, phys, mask).logits
)

estimated_change = (img_perturbation * ig_grad).sum()
actual_change = output_change.item()

error = (estimated_change - actual_change) / (actual_change + 1e-8)
print(f"Gradient approximation error: {error:.2%}")

# Should be <5% for good IG
```

### Test 3: Saliency Aggregation Consistency

```python
# Compare different aggregations
attr = ig.explain(img, phys)  # (3, H, W)

sal_max = attr.max(axis=0)
sal_l2 = np.linalg.norm(attr, axis=0)
sal_mean = attr.mean(axis=0)

# Compute ranking correlation
from scipy.stats import spearmanr

corr_max_l2, _ = spearmanr(sal_max.flatten(), sal_l2.flatten())
corr_max_mean, _ = spearmanr(sal_max.flatten(), sal_mean.flatten())

print(f"Correlation max-vs-L2: {corr_max_l2:.3f}")
print(f"Correlation max-vs-mean: {corr_max_mean:.3f}")

# Should all be >0.8 (similar rankings)
```

---

## DECISION POINTS FOR YOUR THESIS

### Decision 1: Baseline Selection
**Choose one:**
- [ ] Black baseline (simplest)
- [ ] Average face baseline (if you compute it)
- [ ] Test all and report robustness

### Decision 2: Completeness Axiom
**Choose one:**
- [ ] Remove completeness check (it's wrong)
- [ ] Fix it properly (implement correctly)
- [ ] Don't claim it in paper

### Decision 3: Saliency Aggregation
**Choose one:**
- [ ] Keep L2 norm (academic standard)
- [ ] Show all three in appendix
- [ ] Use per-channel visualization

### Decision 4: Stability Handling
**Choose one:**
- [ ] Raise error on NaN (proper error handling)
- [ ] Log warning and skip sample (skip bad data)
- [ ] Document limitation

---

## TIMELINE FOR PROPER IMPLEMENTATION

### Phase 1 (2-3 hours): Fix Critical Issues
- [ ] Remove broken completeness check
- [ ] Fix NaN handling to raise error
- [ ] Choose baseline with justification

### Phase 2 (2-3 hours): Validation Testing
- [ ] Test IG stability (multiple runs)
- [ ] Verify gradient sanity (perturbation test)
- [ ] Check aggregation consistency

### Phase 3 (1-2 hours): Documentation
- [ ] Document choices (why you picked this)
- [ ] Document limitations (what could be wrong)
- [ ] Document validation results

---

## FINAL RECOMMENDATION

**Don't submit current XAI module with:**
- ❌ Broken completeness check
- ❌ Silent NaN replacement
- ❌ Hardcoded ImageNet normalization
- ❌ Unsupported "gray" baseline

**DO submit with:**
- ✅ One well-justified baseline
- ✅ Proper error handling
- ✅ Documented aggregation choice
- ✅ Test results showing stability

**This takes 6-8 hours to do properly.**

