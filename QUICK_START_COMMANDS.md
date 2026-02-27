# 🚀 Quick Start: Run Your Analysis Now

## Copy-Paste Ready Commands

### Step 1: Quick Test (Do This First!) - 2 minutes

```powershell
cd "c:\Users\USERAS\thesis_project"
.\.venv\Scripts\Activate.ps1
python analysis_with_tier3.py --num_samples 5 --ig_steps 10
```

**What to expect**: All output shows `[OK]` and ends with `[DONE] ANALYSIS COMPLETE`

---

### Step 2: Full Analysis (Tomorrow) - 20-100 minutes

#### Option A: Fast (Good for Thesis)
```powershell
python analysis_with_tier3.py --num_samples 100 --ig_steps 20 --batch_size 8
```
**Time**: ~20 minutes on CPU, ~2 minutes on GPU
**Quality**: Good for thesis results

#### Option B: High Quality (Best Results)
```powershell
python analysis_with_tier3.py `
  --num_samples 100 `
  --ig_steps 50 `
  --batch_size 8 `
  --device cuda
```
**Time**: ~50 minutes total on GPU
**Quality**: Excellent for thesis + publication

#### Option C: Production Scale (Full Dataset)
```powershell
python analysis_with_tier3.py `
  --num_samples 500 `
  --ig_steps 50 `
  --batch_size 16 `
  --device cuda
```
**Time**: ~4 hours on GPU
**Quality**: Most comprehensive

---

### Step 3: After Analysis - Visualize Results

Read this for visualization code:
```
Open file: REAL_DATA_ANALYSIS_GUIDE.md
Section: Step 6: Visualize Results
Copy the matplotlib code to create:
  - attribution_visualization.png
  - method_agreement.png
  - profiling_comparison.png
```

---

### Step 4: Before Your Defense

Read in this order:
1. `VIVA_QUICK_REFERENCE_CHEATSHEET.md` (5 min read)
2. `MODEL_ARCHITECTURE_COMPARISON_FRAMEWORK.md` (10 min read)
3. `00_FINAL_STATUS_READY_FOR_ANALYSIS.md` (10 min read)

---

## What Each Command Does

| Command | Time | Purpose | Use When |
|---------|------|---------|----------|
| `--num_samples 5 --ig_steps 10` | 1 min | Verify setup works | Testing |
| `--num_samples 100 --ig_steps 20` | 20 min | Thesis results | Main analysis |
| `--num_samples 100 --ig_steps 50` | 50 min | High quality | Publication |
| `--num_samples 500 --ig_steps 50` | 4 hours | Complete analysis | Large dataset |

---

## Check GPU Availability (Optional But Recommended)

```powershell
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

If it prints `GPU Available: True`, add `--device cuda` to your command for 5-10× speedup!

---

## What Happens During Analysis

Your script will:

1. **Load data** from `data/` folder
2. **Generate explanations** using 3 methods (IG, Saliency, Attention)
3. **Validate** that explanations are good quality
4. **Compare** methods to ensure they agree
5. **Profile** performance (timing, memory)
6. **Report** results with statistics

**Total output**: `results/` folder with all analysis data

---

## If Something Goes Wrong

### Error: "CUDA out of memory"
```powershell
python analysis_with_tier3.py --num_samples 50 --ig_steps 20 --batch_size 2
```
(Reduce batch_size and num_samples)

### Error: "ModuleNotFoundError"
```powershell
.\.venv\Scripts\Activate.ps1
pip install torch torchvision numpy
```

### Error: Takes too long
```powershell
python analysis_with_tier3.py --num_samples 50 --ig_steps 10
```
(Reduce num_samples and ig_steps for quicker test)

---

## After Your Analysis Is Complete

You'll have files like:
```
tier3_analysis_results.pkl    <- Complete results object
explanations.npy              <- Attribution heatmaps
profiling_summary.csv         <- Performance metrics
comparison_report.txt         <- Method comparison
batch_statistics.json         <- Aggregated metrics
```

Use these for:
- **Thesis figures**: Visualizations and tables
- **Viva defense**: Show results slides
- **Publication**: Detailed metrics in appendix

---

## Your Next 24 Hours

### Today (Now)
- [ ] Run Step 1 (2 min) - verify setup works

### Tomorrow  
- [ ] Run Step 2 Option A or B (20 min to 1 hour)
- [ ] Wait for analysis to complete
- [ ] Check `results/` folder for output files

### Next 2 Days
- [ ] Run Step 3 visualization code
- [ ] Generate 3-4 thesis figures
- [ ] Write Chapter 5 (Results) with those figures

### Week 1
- [ ] Read VIVA_QUICK_REFERENCE_CHEATSHEET.md
- [ ] Write Chapters 3-6 of thesis
- [ ] Prepare defense slides

---

## You're All Set! 🎉

Your XAI analysis pipeline is ready. Just run one of the commands above and wait for your results.

**Good luck with your thesis!** 🚀

