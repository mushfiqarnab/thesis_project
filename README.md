@'
# Multimodal Threat Detection Pipeline (Pre-Thesis 2)

This repository implements a multimodal pipeline for threat classification using:
- Facial images (with synthetic scar generation + mask)
- Physiological signals (HRV + GSR from WESAD)
- Multimodal fusion model (Vision Transformer + physiology MLP)
- Fairness evaluation (DP gap, EO gap) across the sensitive attribute "scar"

> Datasets and generated artifacts are excluded from GitHub (`data/**` is ignored).

## What’s Included
- `src/prepare_faces.py` : generate synthetic scars + masks + faces CSV
- `src/prepare_wesad.py` : extract HRV/GSR features from WESAD windows
- `src/build_multimodal_csv.py` : build balanced multimodal.csv
- `src/train_baseline.py` : baseline multimodal training (ViT + physiology)
- `src/eval_fairness.py` : fairness metrics (DP gap, EO gap)

## What’s Not Included
- Face datasets (CelebA/FFHQ)
- WESAD raw files
- Generated images, CSVs, or model checkpoints

## How to Reproduce (High Level)
1) Prepare faces (requires CelebA/FFHQ set up locally):
   - `python src/prepare_faces.py`

2) Prepare physiology windows (requires WESAD set up locally):
   - `python src/prepare_wesad.py`

3) Build the multimodal dataset:
   - `python src/build_multimodal_csv.py`

4) Train baseline:
   - `python src/train_baseline.py`

5) Evaluate fairness:
   - `python src/eval_fairness.py`

## Status
- Baseline training + fairness evaluation completed
- Counterfactual fairness training is the next stage
'@ | Set-Content -Encoding UTF8 README.md
