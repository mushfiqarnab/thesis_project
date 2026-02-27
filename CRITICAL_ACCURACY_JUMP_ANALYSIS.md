# 🔴 CRITICAL ANALYSIS: Why Pruning "Improves" Accuracy

## The Discovery

**Base CGF training** (src/train_cgf_fair.py):
```
Loss = CE + 1.0*JS_cf + 0.05*GateReg + 0.5*DP + 0.5*EO
       └─ strong fairness penalties ──────────────────┘
```

**Repair training** (src/fair_repair_finetune.py):
```
Loss = CE + 1.0*JS_cf + 0.05*GateReg + 0.3*DP + 0.3*EO
       └─ RELAXED fairness penalties (60% of original) ┘
```

---

## The Truth

The 24pp accuracy jump (53.2% → 77.85%) is achieved by **RELAXING FAIRNESS CONSTRAINTS** during repair training:

| Component | Base CGF | Pruned CGF | Change |
|-----------|----------|-----------|--------|
| lambda_DP | 0.5 | 0.3 | **-40%** |
| lambda_EO | 0.5 | 0.3 | **-40%** |
| Accuracy | 53.2% | 77.85% | **+24.65pp** |

This is **Option B (Honest but Risky)**: not a fair comparison. You're trading fairness regularization for task accuracy.

---

## The Questions This Raises

1. **Is this inherently wrong?** No — relaxing fairness-accuracy tradeoff is a valid design choice.

2. **But is it honest storytelling?** Only if you explicitly acknowledge the tradeoff.

3. **What would a panel attack you on?** 
   - > "You claim fairness improves from 4.1% to 0.35% EO gap. But you also relaxed fairness constraints by 40%. How do I know the improvement isn't just weaker regularization?"

---

## What the Actual Pruned Model Achieved

According to the evaluation JSON:
```json
"eo_max_gap": 0.0035  ← 0.35% EO gap (EXCELLENT)
```

But was this measured with:
- Same test setup as base CGF?  
- Same preprocessing?
- Or did pruning + relaxed training somehow improve fairness anyway?

---

## Your Defense Options

### Option 1: HONEST & DEFENSIBLE
> "The base CGF prioritizes fairness learning under strong constraints (0.5 weights), achieving 4.1% EO gap but lower accuracy (53.2%). The pruned variant relaxes those constraints (0.3 weights) during repair to demonstrate edge-deployment feasibility. Both are valid points on the fairness-accuracy frontier. The pruned model shows 0.35% EO gap, which is even better than base CGF, suggesting compression itself has regularizing effects beyond just capacity redistribution."

**Requires**: Ablation showing `pruned_repair_with_0.5_constraints > 53.2%` to prove pruning alone helps.

---

### Option 2: REFRAME THE STORY  
> "We trained CGF with two objectives: (1) base model learns causal fairness structure under strong constraints, (2) pruned model validates edge-deployment with balanced fairness-accuracy tradeoff. The fact that pruned achieves 0.35% EO gap with relaxed constraints suggests our learned focus mechanism is robust — fairness doesn't depend on heavy regularization alone."

**Requires**: This can work if you own the tradeoff openly.

---

### Option 3: RUN ABLATION IMMEDIATELY
Before viva, check what happens if you repair with `--lambda_dp 0.5 --lambda_eo 0.5`:
```bash
python src/fair_repair_finetune.py \
  --ckpt_in outputs/checkpoints/counterfactual_cgf_js_mobilenet_v3_small_multimodal_10k_unbiased_pruned30.pt \
  --csv data/csv/multimodal_10k_unbiased.csv \
  --split data/csv/split_seed42_multimodal_10k_unbiased.json \
  --lambda_dp 0.5 --lambda_eo 0.5 \
  --epochs 5
```

Then evaluate to see: **Does pruned+repair with SAME fairness constraints as base reach >77%?**

If YES: You can claim legitimate "capacity redistribution"  
If NO: You must acknowledge the constraint relaxation in viva

---

## The Bottom Line for Your Viva

**You CANNOT say**: "Pruning improved accuracy because of capacity redistribution" without evidence that pruning + same fair constraints > 53.2%.

**You CAN say (if ablation done)**:  
- "Pruning actually helps. Base 53.2% with λ=0.5, pruned X% with λ=0.5. The gap is pure compression benefit."

**You SHOULD say (honest framing)**:  
- "Base prioritizes fairness learning (0.5 constraints). Pruned balances fairness-accuracy with relaxed constraints (0.3). Both valid deployment choices."

---

## What Should I Do?

You need ONE of:

1. **Run the ablation** (fair comparison): pruned repair with same λ values
2. **Check evaluation setup**: Did the stored 77.85% use different settings than we expect?
3. **Find training logs**: Do training reports show what λ values were actually used?

Which do you want me to investigate first?
