# 🗺️ Project Architecture Map

## System Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FAIR MULTIMODAL THREAT DETECTION                         │
│                   (Causal Gated Fusion Architecture)                        │
└─────────────────────────────────────────────────────────────────────────────┘

LAYER 1: DATA SOURCES
═══════════════════════════════════════════════════════════════════════════════
┌─────────────────────┐                    ┌──────────────────────┐
│  FFHQ Faces (70K)   │                    │ WESAD Physiology     │
│  - High quality     │                    │ - ECG (700Hz)        │
│  - Diverse faces    │                    │ - EDA (700Hz)        │
│  - RGB images       │                    │ - Baseline & Stress  │
└──────────┬──────────┘                    └──────────┬───────────┘
           │                                          │
           ▼                                          ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│  FACE PROCESSING         │      │ SIGNAL PROCESSING            │
│  prepare_faces.py        │      │ prepare_wesad.py             │
│  ├─ Resize 256×256       │      │ ├─ Extract 30-sec windows    │
│  ├─ Add synthetic scars  │      │ ├─ Calculate HRV features    │
│  ├─ Generate masks       │      │ ├─ Calculate GSR features    │
│  └─ Save all versions    │      │ └─ Map labels (0/1)          │
└──────────┬───────────────┘      └──────────┬───────────────────┘
           │                                  │
           ▼                                  ▼
      faces.csv                        wesad_windows.csv
    [paths, scars]                   [hrv, gsr, threat]


LAYER 2: DATASET ASSEMBLY
═══════════════════════════════════════════════════════════════════════════════
                ┌─────────────────────────────────┐
                │  build_multimodal_csv.py        │
                │  ├─ Merge faces + physiology    │
                │  ├─ Combine metadata            │
                │  └─ Create final CSV            │
                └─────────────┬───────────────────┘
                              │
                              ▼
          ┌───────────────────────────────────────┐
          │ multimodal_10k_unbiased.csv           │
          │ 10,000 samples                        │
          │ Columns: [image, hrv, gsr, scar,     │
          │           threat, mask, subject]      │
          │ Train: 8,000 | Valid: 2,000           │
          └───────────────────┬───────────────────┘
                              │
           ┌──────────────────┴───────────────────┐
           ▼                                      ▼
      Training Data                        Validation Data
       (80%, 8,000)                         (20%, 2,000)


LAYER 3: NEURAL NETWORK ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

INPUT
─────
[256×256 RGB Image]    [HRV scalar]  [GSR scalar]
       │                    │              │
       │ Vision Encoder     │ Physiology   │
       │                    │ Encoder      │
       ▼                    │              ▼
    MobileNetV3-Small       │         PhysMLP
    (576 channels)          │         (64 dim)
       │                    │           │
       ▼                    │           ▼
  576-dim embedding         │      64-dim embedding
       │                    │           │
       └────────┬───────────┴───────────┘
                │
        CAUSAL GATED FUSION (CGF)
        ════════════════════════════════════════
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
    V_proj          P_proj
  (256-dim)       (256-dim)
        │                │
        │                ▼
        │         [focus computation]
        │         scar activation
        │         ────────────────
        │         overall activation
        │              │
        │         focus (scalar)
        │              │
        │    ┌─────────┴───────┐
        │    │                 │
        │    │  Gate MLP       │
        │    │  [phys, focus]  │
        │    │  → sigmoid(·)   │
        │    │                 │
        │    ▼                 │
        │  gate ∈ [0,1]       │
        │  (0=trust phys)     │
        │  (1=trust vision)   │
        │    │                 │
        └────┼─────────────────┘
             │
        Fusion: gate*v + (1-gate)*p
             │
             ▼
        256-dim fused representation
             │
             ▼
        Classifier Head
        ├─ Linear(256→128) + ReLU + Dropout
        └─ Linear(128→2) → logits
             │
             ▼
        [P(safe), P(threat)]


LAYER 4: TRAINING LOSSES
═══════════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────────────┐
│ Total Loss = L_task + λ_cf·L_cf + λ_gate·L_gate + λ_dp·L_dp + λ_eo·L_eo   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  L_task: Binary cross-entropy                                            │
│          [Ensures model learns to classify threat vs safe]               │
│                                                                            │
│  L_cf: Counterfactual fairness loss                                      │
│        [Prediction unchanged if scar removed]                            │
│        = ||P(threat|img) - P(threat|img_no_scar)||                       │
│                                                                            │
│  L_gate: Gate regularization                                             │
│          [Prevent gate from collapsing to 0 or 1]                        │
│          = regularization of gate values                                 │
│                                                                            │
│  L_dp: Demographic parity loss                                           │
│        [Equal threat rates for scar vs no-scar]                          │
│        = |P(threat|scar=1) - P(threat|scar=0)|                           │
│                                                                            │
│  L_eo: Equalized odds loss                                               │
│        [Equal TPR and FPR across scar groups]                            │
│        = |TPR_scar - TPR_no_scar| + |FPR_scar - FPR_no_scar|            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘


LAYER 5: EVALUATION & ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

             ┌────────────────────────────┐
             │ Trained Model Checkpoint   │
             │ (cgf_full_best.pt)         │
             └────────────┬───────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │ evaluate_model_comprehensive.py  │
        │ ├─ Accuracy, Precision, Recall  │
        │ ├─ F1, AUC-ROC                  │
        │ ├─ Confusion Matrix             │
        │ └─ Fairness Metrics             │
        └────────────┬────────────────────┘
                     │
        ┌────────────┴────────────────┐
        ▼                             ▼
    VISUALIZATIONS              JSON REPORTS
    ──────────────              ────────────
    • confusion_matrix.png      • evaluation_report.json
    • roc_curve.png             • fairness_metrics.json
    • metrics_summary.png       • p2_summary.csv
    • feature_statistics.png
    • train_test_split.png
    • class_distribution.png
    • auc_roc_score.png


LAYER 6: DEPLOYMENT OPTIONS
═══════════════════════════════════════════════════════════════════════════════

Model Compression Pipeline:
════════════════════════════

cgf_full_best.pt (2.25M params, ~9MB)
         │
         ├─→ prune_checkpoint.py (remove 30%)
         │              │
         │              ▼
         │   cgf_pruned_30_best.pt (1.4M params, ~5.6MB)
         │
         └─→ quantize_export.py (8-bit quantization)
                        │
                        ▼
            cgf_quantized.pt (350KB!)
                        │
                        ▼
            [Deploy to mobile/embedded]
            ├─ Android (TensorFlow Lite)
            ├─ iOS (Core ML)
            └─ Embedded Linux (ONNX Runtime)

Performance after compression:
├─ Accuracy: ~77% (maintained)
├─ Latency: ~2.5ms (improved from 4.36ms)
└─ Model size: 350KB (97% reduction!)


LAYER 7: DOCUMENTATION & DEFENSE
═══════════════════════════════════════════════════════════════════════════════

Documentation Pyramid:
                        ▲
                       ╱│╲
                      ╱ │ ╲         COMPLETE_ANALYSIS_MASTER_GUIDE.md
                     ╱  │  ╲        (30 min - All details)
                    ╱───┼───╲
                   ╱    │    ╲      RESULTS_AND_HOW_TO.md
                  ╱     │     ╲     DATASET_ANALYSIS_GUIDE.md
                 ╱──────┼──────╲    (20 min each)
                ╱       │       ╲
               ╱        │        ╲  QUICK_REFERENCE.md
              ╱─────────┼─────────╲ START_HERE.md
             ╱          │          ╲ (5-10 min)
            ╱───────────┴───────────╲
           README.md, PROJECT_HOW_IT_WORKS.md
           (Entry points)

Thesis Defense Materials:
├─ MASTER_VIVA_CHECKLIST.md
├─ EMERGENCY_VIVA_DEFENSE_GUIDE_2HOURS.md
├─ VIVA_BATTLE_CARD.md
├─ SAFA_COMPLETE_PREP_PACKAGE.md
├─ MUGDHA_COMPLETE_PREP_PACKAGE.md
└─ [20+ analysis & verification documents]
```

---

## File Dependency Graph

```
Data Sources:
  FFHQ (70K images) ─────┐
  WESAD (raw ECG/EDA) ───┤
                         │
                         ▼
  ┌─────────────────────────────────┐
  │ prepare_faces.py                │ ──→ faces_clean/, faces_synth_scar/,
  │ prepare_wesad.py                │     scar_masks/, faces.csv,
  │ build_multimodal_csv.py          │     wesad_windows.csv
  └─────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────┐
  │ multimodal_10k_unbiased.csv      │
  │ (10,000 samples)                │
  └──────────┬──────────────────────┘
             │
      ┌──────┴───────┐
      │              │
      ▼              ▼
  ┌─────────────────────────────────┐    ┌──────────────────────┐
  │ train_baseline.py               │    │ train_cgf_fair.py    │
  │ (CONCAT model)                  │    │ (CGF model)          │
  └──────────┬──────────────────────┘    └──────────┬───────────┘
             │                                      │
             └──────────────┬───────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │ evaluate_model_comprehensive.py         │
        │ eval_fairness.py                       │
        │ (evaluation suite)                     │
        └────────────┬─────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
    outputs/analysis/        outputs/reports/
    (7 PNG files)            (2 JSON files)
         │                        │
         └────────────┬───────────┘
                      │
                      ▼
            Thesis & Defense Materials
```

---

## Training Loop Overview

```
Epoch 1 → N:
═══════════════════════════════════════════════════════════════════════════

For each batch of 32 samples:
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Load data                                            │
  │    ├─ Face image (256×256)                             │
  │    ├─ HRV, GSR features (scalars)                      │
  │    ├─ Scar flag (0 or 1)                              │
  │    ├─ Threat label (0 or 1)                           │
  │    └─ Scar mask (binary)                              │
  │                                                        │
  │ 2. Forward pass                                        │
  │    ├─ Vision encoder → 576-dim                         │
  │    ├─ Physiology encoder → 64-dim                      │
  │    ├─ CGF gate → scalar (0-1)                          │
  │    ├─ Fusion → 256-dim                                 │
  │    └─ Classifier → logits (2-dim)                      │
  │                                                        │
  │ 3. Compute losses                                      │
  │    ├─ L_task: BCE(logits, threat)                     │
  │    ├─ L_cf: ||pred - pred_counterfactual||            │
  │    ├─ L_gate: regularize gate values                  │
  │    ├─ L_dp: |P(threat|scar=1) - P(threat|scar=0)|    │
  │    └─ L_eo: |TPR_gap| + |FPR_gap|                    │
  │                                                        │
  │ 4. Total loss = weighted sum of all losses            │
  │                                                        │
  │ 5. Backward pass                                       │
  │    └─ Compute gradients                               │
  │                                                        │
  │ 6. Optimizer step (Adam)                              │
  │    └─ Update all parameters                           │
  │                                                        │
  │ 7. Validation (every N batches)                       │
  │    ├─ Evaluate on validation set                      │
  │    ├─ Compute fairness metrics                        │
  │    └─ Save best checkpoint if improved                │
  └─────────────────────────────────────────────────────────┘

Final Output:
  Best checkpoint saved: cgf_full_best.pt
  Metrics: Accuracy, F1, AUC-ROC, DP Gap, EO Gap
```

---

## Results Comparison Table

```
┌──────────────────┬──────────────────┬──────────────┬────────────┬──────────────┐
│ Aspect           │ Baseline (CONCAT)│ CGF (Full)   │ Pruned 30% │ Quantized    │
├──────────────────┼──────────────────┼──────────────┼────────────┼──────────────┤
│ Accuracy         │ 73.45%           │ 77.85% ↑↑   │ 77.85% ↑↑ │ ~77% ↑↑      │
│ F1 Score         │ 0.55             │ 0.61 ↑      │ 0.61 ↑    │ ~0.61 ↑      │
│ AUC-ROC          │ 0.62             │ 0.68 ↑      │ 0.68 ↑    │ ~0.67 ↑      │
├──────────────────┼──────────────────┼──────────────┼────────────┼──────────────┤
│ DP Gap           │ 0.0142           │ 0.0110 ✓    │ 0.0054 ✓✓ │ ~0.0054 ✓✓   │
│ EO Gap           │ 0.0109           │ 0.0050 ✓    │ 0.0035 ✓✓ │ ~0.0035 ✓✓   │
│ CF Stability     │ 0.15             │ 0.06 ✓      │ 0.04 ✓✓   │ ~0.04 ✓✓     │
├──────────────────┼──────────────────┼──────────────┼────────────┼──────────────┤
│ Parameters       │ 2.1M             │ 2.25M       │ 1.4M ↓    │ ~350K ↓↓     │
│ Model Size       │ ~8.4MB           │ ~9MB        │ ~5.6MB ↓  │ ~350KB ↓↓    │
│ Latency (GPU)    │ 4.15ms           │ 4.36ms      │ 4.16ms    │ ~2.5ms ↓     │
│ Latency (Mobile) │ N/A              │ N/A         │ ~8ms      │ ~3-4ms ↓↓    │
├──────────────────┼──────────────────┼──────────────┼────────────┼──────────────┤
│ Deployment       │ Server only      │ Server only │ Edge-ready│ Mobile ✓✓    │
│ Real-time?       │ Yes              │ Yes         │ Yes       │ Yes          │
└──────────────────┴──────────────────┴──────────────┴────────────┴──────────────┘

Legend: ↑↑ significant improvement | ✓ improved | ✓✓ significantly improved
        ↓ reduced | ↓↓ significantly reduced
```

