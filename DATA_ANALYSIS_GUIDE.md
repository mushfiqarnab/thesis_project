# 📊 DATA ANALYSIS PHASE: Getting Started

**Status**: Ready to begin analysis  
**Your Tools**: TIER 3 Enhanced XAI Module  
**Timeline**: 8+ weeks available

---

## 🎯 What You'll Do Now

### Phase 1: Data Preparation (Week 1)
- Load your dataset
- Verify data format
- Split train/validation/test
- Check model checkpoint

### Phase 2: Explanation Generation (Week 2-3)
- Generate explanations for test samples
- Use multiple XAI methods (IG, Saliency, Attention)
- Validate results with IGValidator
- Profile performance with PerformanceProfiler

### Phase 3: Analysis & Comparison (Week 4-5)
- Compare XAI methods with ExplanationComparator
- Analyze gradient flow with GradientFlowAnalyzer
- Create visualizations
- Generate comparison reports

### Phase 4: Thesis Writing (Week 6-12)
- Write methodology with code examples
- Include XAI results as figures
- Show validation evidence
- Defend design choices

---

## 🚀 Quick Start

### Option 1: With Your Own Data

```bash
# Generate explanations with TIER 3 enhancements
python analysis_with_tier3.py \
    --ckpt your_checkpoint.pth \
    --csv your_dataset.csv \
    --num_samples 100 \
    --ig_steps 50

# Expected output:
# [STEP 1] Generating Explanations (with profiling)...
#   ✅ integrated_gradients: 0.7234 (2.1234s)
#   ✅ saliency_map: 0.7210 (0.0432s)
#   ✅ attention: 0.7198 (0.0098s)
# 
# [STEP 2] Validating Results...
#   ✅ PASS: Attribution validation
# 
# [STEP 3] Comparing Methods for Consistency...
#   ✅ AGREE: Method predictions
#   - Mean: 0.7214
#   - Spread: 0.0036
```

### Option 2: With Dummy Data (Testing)

```bash
# Run with dummy data to test pipeline
python analysis_with_tier3.py \
    --num_samples 10 \
    --ig_steps 25

# Good for:
# - Testing your pipeline
# - Verifying all components work
# - Understanding expected output
# - Estimating timing on full dataset
```

---

## 📋 Understanding the Output

### STEP 1: Explanation Generation

Shows predictions from each method:
```
integrated_gradients: 0.7234 (2.1234s)  ← Method name, prediction, time
saliency_map: 0.7210 (0.0432s)
attention: 0.7198 (0.0098s)
```

**What it means**:
- All methods return similar predictions → Good ✅
- Large difference between methods → Investigate ⚠️
- Some methods much slower → Expected (IG is slower)

### STEP 2: Validation

```
✅ PASS: Attribution validation
  - Has NaN: False
  - All zero: False
  - Gate varies: True
```

**What to look for**:
- ✅ Has NaN: False (no numerical errors)
- ✅ All zero: False (attributions not constant)
- ✅ Gate varies: True (gate is learning)

### STEP 3: Method Comparison

```
✅ AGREE: Method predictions
  - Mean: 0.7214
  - Spread: 0.0036
  - Attribution correlation: r=0.9711
```

**Interpretation**:
- Spread < 0.1 → Methods agree strongly ✅
- Spread > 0.2 → Methods disagree, investigate ⚠️
- Correlation > 0.7 → Attributions similar ✅
- Correlation < 0.5 → Methods different ⚠️

### BATCH SUMMARY

```
Processed: 100 successful, 0 failed

Predictions:
  Mean: 0.6234
  Std: 0.2145
  Min: 0.0234
  Max: 0.9876

Validation: 98/100 (98%)
Method Consistency: 97/100 (97%)
```

**What this tells you**:
- Validation pass rate → Confidence in results
- Method consistency → Robustness of explanations
- Prediction distribution → Model behavior

---

## 🧪 Using TIER 3 Tools Directly

### 1. GradientFlowAnalyzer - Monitor Gradient Health

```python
from src.xai import GradientFlowAnalyzer

analyzer = GradientFlowAnalyzer()

# Check single layer
results = analyzer.analyze_gradients(gradients, "conv5")
print(f"Gradient health: {results['is_healthy']}")

# Check full network
layer_grads = {'conv1': grad1, 'conv2': grad2, 'fc1': grad3}
results = GradientFlowAnalyzer.analyze_multiple_layers(layer_grads)
print(f"Overall health: {results['_overall_healthy']}")

# Use case: Verify IG computation isn't affected by vanishing gradients
```

### 2. ExplanationComparator - Compare Methods

```python
from src.xai import ExplanationComparator

comparator = ExplanationComparator()

# Compare attributions from different methods
results = comparator.compare_attributions({
    'integrated_gradients': attr_ig,
    'saliency_map': attr_saliency,
    'attention': attr_attention
})

print(f"Methods agree with r={results['mean_pearson']:.3f}")

# Compare predictions
pred_results = comparator.compare_predictions({
    'method_a': 0.72,
    'method_b': 0.73,
    'method_c': 0.71
})
print(f"Predictions agree: {pred_results['predictions_agree']}")

# Use case: Validate that different XAI methods give consistent results
```

### 3. PerformanceProfiler - Benchmark Methods

```python
from src.xai import PerformanceProfiler

profiler = PerformanceProfiler()

# Profile single method
result, timing = profiler.profile_method(
    ig.explain, img, phys,
    method_name="integrated_gradients"
)
print(f"Time: {timing['wall_time_sec']:.4f}s")
print(f"GPU memory: {timing['gpu_memory_mb']:.2f} MB")

# Profile multiple methods
methods = {
    'ig': ig.explain,
    'saliency': saliency.explain,
    'attention': attention.explain
}
timings = profiler.profile_multiple_methods(methods, img, phys)

# Estimate complexity
complexity = profiler.estimate_complexity(batch_size=32)
print(f"IG is {complexity['ig_complexity_ratio']:.0f}x more expensive than Saliency")

# Use case: Estimate computational requirements for full dataset
```

---

## 📊 Analysis Workflow

### Step 1: Prepare Your Data

```python
from torch.utils.data import DataLoader

# Load your dataset
train_dataset = YourDataset('path/to/train.csv')
test_dataset = YourDataset('path/to/test.csv')

train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=1)
```

### Step 2: Load Model and Explainer

```python
from src.models import MultimodalThreatModel
from src.xai import XAIExplainer

model = MultimodalThreatModel(...)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

explainer = XAIExplainer(model, num_steps=50)
```

### Step 3: Generate Explanations

```python
with torch.no_grad():
    for img, phys, mask in test_loader:
        # Generate all explanations at once
        results = explainer.explain_batch(img, phys, mask)
        
        # Or individual methods
        ig_result = explainer.ig.explain(img, phys, mask)
        sal_result = explainer.saliency.explain(img, phys, mask)
        att_result = explainer.attention.explain(img, phys, mask)
```

### Step 4: Validate Results

```python
from src.xai import IGValidator

validator = IGValidator()

validation = validator.full_validation(
    attr_img=ig_result.vision_attribution,
    attr_phys=ig_result.phys_attribution,
    gate_values=[ig_result.gate_activation]
)

if validation['all_pass']:
    print("✅ Results are valid")
else:
    print("⚠️ Issues detected:", validation)
```

### Step 5: Compare Methods

```python
from src.xai import ExplanationComparator

comparator = ExplanationComparator()

# Compare predictions across methods
comparison = comparator.compare_predictions({
    'ig': float(ig_result.prediction),
    'saliency': float(sal_result.prediction),
    'attention': float(att_result.prediction)
})

print(f"Methods agree: {comparison['predictions_agree']}")
```

### Step 6: Profile Performance

```python
from src.xai import PerformanceProfiler

profiler = PerformanceProfiler()

# Time how long each method takes
for method_name, method_fn in [('ig', ig.explain), ('saliency', sal.explain)]:
    _, timing = profiler.profile_method(method_fn, img, phys)
    print(f"{method_name}: {timing['wall_time_sec']:.4f}s")
```

---

## 💾 Saving Results

### Save Individual Explanations

```python
import json
import numpy as np

# Convert to JSON-serializable format
explanation_dict = {
    'prediction': float(result.prediction),
    'prediction_class': int(result.prediction_class),
    'gate_activation': float(result.gate_activation) if result.gate_activation else None,
    'focus_activation': float(result.focus_activation) if result.focus_activation else None,
    'metadata': result.metadata
}

with open('explanation.json', 'w') as f:
    json.dump(explanation_dict, f)

# Save attributions
if result.vision_attribution is not None:
    np.save('attribution_vision.npy', result.vision_attribution)
if result.phys_attribution is not None:
    np.save('attribution_phys.npy', result.phys_attribution)
```

### Save Analysis Report

```python
import json
from datetime import datetime

report = {
    'timestamp': datetime.now().isoformat(),
    'num_samples': 100,
    'model_name': 'mobilenet_v3_small',
    'fusion_type': 'cgf',
    'ig_steps': 50,
    'results': {
        'validation_pass_rate': 0.98,
        'method_agreement_rate': 0.97,
        'mean_prediction': 0.62,
        'method_timings': {
            'ig': 2.12,
            'saliency': 0.043,
            'attention': 0.0098
        }
    }
}

with open('analysis_report.json', 'w') as f:
    json.dump(report, f, indent=2)
```

---

## 📈 Creating Visualizations

### Visualize Attributions

```python
import matplotlib.pyplot as plt
from src.xai.visualization import visualize_saliency, visualize_integrated_gradients

# Create figure with multiple explanations
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Original image
axes[0, 0].imshow(denormalize(img))
axes[0, 0].set_title("Original Image")

# Saliency map
visualize_saliency(saliency_result.vision_attribution, ax=axes[0, 1])
axes[0, 1].set_title("Saliency Map")

# Integrated Gradients
visualize_integrated_gradients(ig_result.vision_attribution, ax=axes[1, 0])
axes[1, 0].set_title("Integrated Gradients")

# Attention/Gate
axes[1, 1].text(0.5, 0.5, f"Gate: {ig_result.gate_activation:.3f}\nFocus: {ig_result.focus_activation:.3f}")
axes[1, 1].set_title("Gate Mechanism")

plt.tight_layout()
plt.savefig('explanation_comparison.png', dpi=150)
```

### Plot Performance Comparison

```python
import matplotlib.pyplot as plt

methods = ['Integrated Gradients', 'Saliency Map', 'Attention']
times = [2.12, 0.043, 0.0098]

plt.figure(figsize=(10, 6))
plt.bar(methods, times, color=['red', 'blue', 'green'])
plt.ylabel('Time (seconds)')
plt.title('XAI Method Performance Comparison')
plt.yscale('log')
plt.tight_layout()
plt.savefig('performance_comparison.png')
```

---

## 🐛 Troubleshooting

### Problem: NaN in attributions

**Cause**: Gradient explosion or numerical instability  
**Solution**: 
```python
# Check gradient health
results = gradient_analyzer.analyze_gradients(gradients)
if not results['is_healthy']:
    # Reduce batch size or increase IG steps
    print("Gradient health:", results)
```

### Problem: Methods disagree significantly

**Cause**: Implementation issue or data problem  
**Solution**:
```python
# Check prediction consistency
comparison = comparator.compare_predictions(predictions)
if not comparison['predictions_agree']:
    print("Spread:", comparison['prediction_spread'])
    # Investigate why methods give different results
```

### Problem: Very slow performance

**Cause**: Too many IG steps or batch size too large  
**Solution**:
```python
# Profile to identify bottleneck
timings = profiler.profile_multiple_methods(methods, img, phys)

# Reduce IG steps if it's the bottleneck
explainer = XAIExplainer(model, num_steps=20)  # Fewer steps
```

---

## ✅ Checklist for Analysis

- [ ] Model checkpoint loads correctly
- [ ] Data loads and shapes are correct
- [ ] Single sample analysis works
- [ ] Batch analysis completes without errors
- [ ] Validation pass rate > 95%
- [ ] Method agreement rate > 90%
- [ ] Performance within expected bounds
- [ ] Results saved and documented
- [ ] Visualizations created
- [ ] Analysis report generated

---

## 📚 Next Steps

### Week 1-2: Get comfortable with the pipeline
1. Run `analysis_with_tier3.py` on small subset
2. Verify outputs are reasonable
3. Create sample visualizations
4. Document your data format

### Week 3-4: Full analysis
1. Run on complete test set
2. Compare all XAI methods
3. Analyze gradient flow
4. Profile on your hardware

### Week 5-6: Synthesis
1. Create analysis report with statistics
2. Generate key figures for thesis
3. Write methodology section
4. Prepare defense explanations

### Week 7-12: Thesis writing
1. Integrate results into chapters
2. Discuss findings
3. Compare to related work
4. Conclude with contributions

---

## 💡 Pro Tips

1. **Start small**: Test on 10 samples first to verify pipeline
2. **Check early**: Validate results after first 50 samples
3. **Profile early**: Know your timing on real data
4. **Save often**: Checkpoint results periodically
5. **Document**: Keep analysis notebook with key findings
6. **Compare methods**: Show your explanations are consistent
7. **Validate thoroughly**: Use IGValidator on every batch

---

## 🎯 Your Goal

By end of data analysis phase, you should have:

✅ Explanations for 100+ test samples  
✅ Validation evidence (>95% pass rate)  
✅ Method comparison results  
✅ Performance profiles  
✅ Visualization figures  
✅ Analysis report  
✅ Thesis-ready results  

**Then**: Write it all up in your thesis! 📝

---

**Ready?** Run `python analysis_with_tier3.py --help` to get started! 🚀
