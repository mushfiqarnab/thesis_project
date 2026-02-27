# ✅ Counterfactual Fairness Loss - Formula Correction

**Status**: CONFIRMED & CORRECTED  
**Date**: February 27, 2026  
**Source**: Direct code analysis from `src/train_cgf_fair.py` lines 135-142 and 364-371

---

## 🔍 CONFIRMED: Actual Implementation

### The js_divergence() Function
**File**: `src/train_cgf_fair.py` lines 135-142

```python
def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p,q: (B,2) probabilities -> (B,)
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)
```

### Mathematical Formula

**The code implements**:
```
JS(p||q) = 0.5 × KL(p||m) + 0.5 × KL(q||m)
where m = 0.5 × (p + q)
```

**Expanded**:
```
JS(p||q) = 0.5 × Σ p_i × log(p_i / m_i) + 0.5 × Σ q_i × log(q_i / m_i)
```

**In context of counterfactual fairness**:
```
p = softmax(f(x))           ← prediction on original image (with scar)
q = softmax(f(x_cf))        ← prediction on counterfactual (scar removed)

L_cf = E[JS(p(x) || p(x_cf))]
```

---

## ❌ What Was Claimed (WRONG)

**Current Abstract/Thesis States**:
```
The counterfactual fairness loss is defined as E[|f(x) - f(x_cf)|]
```

**Problems with this**:
1. ❌ It's L1 distance on **raw logits**, not probabilities
2. ❌ It's not **symmetric** in the original/counterfactual
3. ❌ It's unbounded (can grow infinitely)
4. ❌ It's not **probability-based** (doesn't measure distributional similarity)
5. ❌ It contradicts the actual code implementation

---

## ✅ Corrected Abstract Paragraph

### Original Thesis Paragraph (with error highlighted)
```
[Your current abstract paragraph here with the incorrect CF loss formula]
...the counterfactual fairness loss is defined as E[|f(x) - f(x_cf)|], 
which ensures that model predictions remain robust to changes in scar-related 
attributes...
```

### CORRECTED Paragraph
```
[Keep all surrounding text identical, replace ONLY the CF loss definition]

...the counterfactual fairness loss is measured using Jensen-Shannon 
divergence between softmax probability distributions: E[JS(p(x) || p(x_cf))], 
where p denotes the model's output probability distribution. This choice is 
principled because Jensen-Shannon divergence is symmetric, bounded to [0, ln(2)], 
and directly measures the difference in model confidence—ensuring that the model 
makes similar predictions whether the scar attribute is present or counterfactually 
removed...
```

---

## 📐 LaTeX Formulas for Thesis

### One-Line Formula (for abstract/summary)

```latex
L_{\text{cf}} = \mathbb{E}[\text{JS}(p(x) \| p(x_{\text{cf}}))]
```

### Extended Formula (for methods section)

```latex
\text{JS}(p \| q) = \frac{1}{2} \text{KL}(p \| m) + \frac{1}{2} \text{KL}(q \| m)
\quad \text{where} \quad m = \frac{p + q}{2}
```

### Full Training Loss (for methods section)

```latex
L_{\text{total}} = L_{\text{task}} + \lambda_{\text{cf}} \cdot \text{JS}(p(x) \| p(x_{\text{cf}})) 
                  + \lambda_{\text{gate}} L_{\text{gate}} 
                  + \lambda_{\text{dp}} L_{\text{dp}} 
                  + \lambda_{\text{eo}} L_{\text{eo}}
```

---

## 🎯 Why Jensen-Shannon Divergence?

### Advantages over L1 distance:

| Property | L1 Distance | Jensen-Shannon |
|----------|-----------|-----------------|
| Symmetric | ❌ No | ✅ Yes: JS(p\|\|q) = JS(q\|\|p) |
| Bounded | ❌ No (unbounded) | ✅ Yes: [0, ln(2)] |
| Probability-based | ❌ No (raw logits) | ✅ Yes (normalized distributions) |
| Numerically stable | ⚠️ Risky | ✅ With clamping (eps=1e-8) |
| Fairness interpretation | ❌ Unclear | ✅ Clear: measures confidence similarity |

### When/Why This Loss Works

**JS divergence in counterfactual fairness**:
- Low JS(p\|\|p_cf) → Model predictions same regardless of scar
- High JS(p\|\|p_cf) → Model depends heavily on scar (unfair)
- Training minimizes JS divergence → Scar influence removed
- Applies to both classes (symmetric)

---

## 📝 Exact Correction Instructions

### Step 1: Find the Abstract Section
Search your thesis for text containing:
```
E[|f(x) - f(x_cf)|]
```
or similar L1 distance formula

### Step 2: Replace with This Paragraph

**Find**:
```
[the entire sentence/paragraph describing CF loss with E[|f(x) - f(x_cf)|] formula]
```

**Replace With**:
```
the counterfactual fairness loss is measured using Jensen-Shannon divergence 
between softmax probability distributions: E[JS(p(x) || p(x_cf))], where p 
denotes the model's output probability distribution. This choice is principled 
because Jensen-Shannon divergence is symmetric, bounded to [0, ln(2)], and 
directly measures the difference in model confidence—ensuring that the model 
makes similar predictions whether the scar attribute is present or counterfactually 
removed.
```

### Step 3: Update Methods Section (if present)

If you have a detailed methods section, add:

**New Paragraph**:
```
The counterfactual fairness objective is defined as:

  L_cf = E_x[JS(p(x) || p(x_cf))]

where JS(p || q) = 1/2 · KL(p || m) + 1/2 · KL(q || m) with m = (p+q)/2. 
The Jensen-Shannon divergence is symmetric and bounded, making it suitable 
for fairness constraints where we want identical model behavior regardless 
of counterfactual attribute status.
```

---

## 🔗 Supporting Evidence

**Code Reference**:
- `src/train_cgf_fair.py` line 135-142: `js_divergence()` function definition
- `src/train_cgf_fair.py` line 364-371: Using JS divergence in training loop

**Measurement Verification**:
- Actual fairness results show DP gap = 0.0084 (excellent fairness)
- Confirms that JS divergence training produces fair models

**Mathematical Verification**:
- JS(p||q) ∈ [0, ln(2)] (bounded)
- JS(p||q) = JS(q||p) (symmetric)
- JS → 0 when p ≈ q (predictions similar)

---

## ✨ Quick Checklist for Thesis Update

```
Before submission, ensure:

[ ] Abstract: Replace E[|f(x) - f(x_cf)|] with E[JS(p(x) || p(x_cf))]
[ ] Methods: Add detailed JS divergence definition if section exists
[ ] Chapter 3: Verify it already has correct JS formula (it likely does)
[ ] All references: Search for "E[|f" and ensure no other instances
[ ] LaTeX rendering: Test that LaTeX formulas render correctly in PDF
[ ] Consistency: Verify JS divergence mentioned same way throughout
```

---

## 📚 Citation Format (if needed)

For your thesis, you can cite:

**For Jensen-Shannon divergence**:
```
The Jensen-Shannon divergence (JS(p||q) = 1/2·KL(p||m) + 1/2·KL(q||m)) 
is a symmetric, bounded measure of distributional difference [Lin 1991].
```

**For counterfactual fairness**:
```
We implement counterfactual fairness by minimizing Jensen-Shannon divergence 
between model outputs on original and counterfactually perturbed inputs, 
ensuring fair behavior regardless of scar presence [Kusner et al. 2017].
```

---

## 🚀 Final Recommendations

1. **Update Abstract** - Use the corrected paragraph above
2. **Use LaTeX formula** - `L_{\text{cf}} = \mathbb{E}[\text{JS}(p(x) \| p(x_{\text{cf}}))]`
3. **Explain in Methods** - Add why JS divergence is better than L1
4. **Verify Chapter 3** - Ensure it's already correct (likely is)
5. **No code changes needed** - Implementation is already correct!

---

**Status**: Ready for implementation  
**Risk**: Low (this is factual correction, not code change)  
**Impact**: High (prevents examiner questions about loss formula)  
**Timeline**: 10 minutes to update thesis
