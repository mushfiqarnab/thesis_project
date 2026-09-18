"""
experiment_ablation_suite.py
==============================
Complete publication-quality ablation study for GWPACDNet.

Scientific Questions:
    Q1: What is the DP gap when ONLY physiological features are used?
        If low -> visual scar is the sole bias source (strongest paper claim)
        If also high -> dataset is fundamentally irresolvable

    Q2: What does the scar-only oracle achieve?
        This establishes the "shortcut ceiling" — the accuracy a perfectly biased model gets.

    Q3: What is the theoretical fairness achievable from physio signals alone?
        This sets the DR-PS-ZOCR performance target.

Methodology:
    - Identical 85/15 train/val split (seeded, reproducible)
    - Logistic Regression (sklearn) for tabular baselines
    - MLP (PyTorch) for fair comparison with GWPACDNet scale
    - DP Gap = |P(y_hat=1|scar=1) - P(y_hat=1|scar=0)| on validation set
    - All metrics computed on same 1500-sample held-out validation set

References:
    - Creager et al. (2019). Flexibly Fair Representation Learning. ICML.
    - Zhang et al. (2018). Mitigating Unwanted Biases. AIES.
    - Calders & Verwer (2010). Three Naive Bayes Approaches for Discrimination-Free.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, '.')

# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

def compute_dp_gap(y_pred, scar_labels):
    """
    Demographic Parity Gap = |P(y_hat=1|scar=1) - P(y_hat=1|scar=0)|
    Exact formula used in all GWPACDNet training runs for apples-to-apples comparison.
    """
    scar_arr = np.array(scar_labels)
    pred_arr = np.array(y_pred)
    s1_mask = scar_arr == 1
    s0_mask = scar_arr == 0
    if s1_mask.sum() == 0 or s0_mask.sum() == 0:
        return float('nan')
    p_s1 = pred_arr[s1_mask].mean()
    p_s0 = pred_arr[s0_mask].mean()
    return abs(p_s1 - p_s0)

def pareto_score(acc, dp_gap):
    """
    Pareto Score = Acc - DP_Gap (consistent with training objective).
    Higher is better. Used in all checkpoint comparisons.
    """
    return acc - dp_gap

def print_result(name, acc, dp_gap, note=""):
    ps = pareto_score(acc, dp_gap)
    print(f"  {name:<35} | Acc={acc:.4f} | DP Gap={dp_gap:.4f} | Pareto={ps:.4f}  {note}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 75)
print("  EXPERIMENT: MODALITY ABLATION & FAIRNESS BASELINE SUITE")
print("=" * 75)
print()

csv_path = "data/csv/multimodal_10k.csv"
df = pd.read_csv(csv_path)

# Apply same scaling as clinical_dataloader.py for consistency
df["hrv_scaled"] = df["hrv"] * 10.0
df["gsr_scaled"] = df["gsr"] / 10.0

# Feature matrices
X_physio = df[["hrv_scaled", "gsr_scaled"]].values  # 2-dim (same 2 real features in phys)
X_scar   = df[["scar"]].values                       # 1-dim shortcut oracle
X_all    = df[["hrv_scaled", "gsr_scaled", "scar"]].values  # Full tabular features
y        = df["threat"].values
scar_lab = df["scar"].values

# Same 85/15 split, seeded — identical to training pipeline
X_ph_tr, X_ph_val, X_s_tr, X_s_val, X_a_tr, X_a_val, y_tr, y_val, sc_tr, sc_val = \
    train_test_split(
        X_physio, X_scar, X_all, y, scar_lab,
        test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )

print(f"Train samples : {len(y_tr):,}  |  Val samples: {len(y_val):,}")
print(f"Val threat rate: {y_val.mean():.3f}  |  Val scar rate: {sc_val.mean():.3f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 0: MAJORITY CLASS ORACLE
# ─────────────────────────────────────────────────────────────────────────────

print("[BASELINE 0] Majority Class Oracle (lower bound on accuracy)")
majority = int(np.bincount(y_tr).argmax())
y_pred_majority = np.full(len(y_val), majority)
acc_maj = accuracy_score(y_val, y_pred_majority)
dp_maj  = compute_dp_gap(y_pred_majority, sc_val)
print_result("Majority Class", acc_maj, dp_maj, "(predicts all 0)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 1: SCAR-ONLY SHORTCUT ORACLE
# ─────────────────────────────────────────────────────────────────────────────

print("[BASELINE 1] Scar-Only Shortcut Oracle")
print("  Purpose: Upper bound on biased shortcut accuracy. A model that perfectly")
print("  exploits the scar feature. Defines the 'shortcut ceiling'.")
scaler_s = StandardScaler().fit(X_s_tr)
lr_scar = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
lr_scar.fit(scaler_s.transform(X_s_tr), y_tr)
y_pred_scar = lr_scar.predict(scaler_s.transform(X_s_val))
acc_scar = accuracy_score(y_val, y_pred_scar)
dp_scar  = compute_dp_gap(y_pred_scar, sc_val)
print_result("Scar-Only LR", acc_scar, dp_scar, "(pure visual shortcut)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 2: PHYSIO-ONLY LOGISTIC REGRESSION (KEY EXPERIMENT)
# ─────────────────────────────────────────────────────────────────────────────

print("[BASELINE 2] Physio-Only Logistic Regression (KEY SCIENTIFIC QUESTION)")
print("  Purpose: Determines whether physiological signals (HRV, GSR) carry")
print("  genuine, scar-independent threat information at low DP gap.")
print("  If DP Gap << 0.55: visual scar is the SOLE source of GWPACDNet bias.")
print("  If DP Gap ~= 0.55: dataset-level correlation limits all approaches.")

scaler_ph = StandardScaler().fit(X_ph_tr)
lr_physio = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
lr_physio.fit(scaler_ph.transform(X_ph_tr), y_tr)
y_pred_physio = lr_physio.predict(scaler_ph.transform(X_ph_val))
y_prob_physio = lr_physio.predict_proba(scaler_ph.transform(X_ph_val))[:, 1]
acc_physio = accuracy_score(y_val, y_pred_physio)
dp_physio  = compute_dp_gap(y_pred_physio, sc_val)
print_result("Physio-Only LR", acc_physio, dp_physio, "(HRV + GSR only, no visual)")
print()
print(f"  INTERPRETATION:")
if dp_physio < 0.10:
    print(f"  -> DP Gap {dp_physio:.4f} << 0.55. CONFIRMED: Visual scar is the sole bias source.")
    print(f"     GWPACDNet's 0.55 plateau is an architecture capacity failure, NOT dataset collapse.")
    print(f"     This STRONGLY motivates the DR-PS-ZOCR latent disentanglement framework.")
elif dp_physio < 0.30:
    print(f"  -> DP Gap {dp_physio:.4f} < 0.55. PARTIAL: Physio signals reduce but don't eliminate bias.")
    print(f"     Both modality dominance AND representation capacity contribute to the plateau.")
else:
    print(f"  -> DP Gap {dp_physio:.4f} ~= 0.55. Dataset-level correlation affects all modalities.")
print()


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 3: FULL TABULAR FUSION (ALL FEATURES, NO PENALTY)
# ─────────────────────────────────────────────────────────────────────────────

print("[BASELINE 3] Full Tabular Fusion — HRV + GSR + Scar (no fairness penalty)")
print("  Purpose: Upper bound accuracy with all tabular features; establishes")
print("  the accuracy ceiling for a non-image baseline system.")

scaler_a = StandardScaler().fit(X_a_tr)
lr_all = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
lr_all.fit(scaler_a.transform(X_a_tr), y_tr)
y_pred_all = lr_all.predict(scaler_a.transform(X_a_val))
acc_all = accuracy_score(y_val, y_pred_all)
dp_all  = compute_dp_gap(y_pred_all, sc_val)
print_result("Full Tabular LR", acc_all, dp_all, "(HRV + GSR + Scar)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 4: PHYSIO MLP (Scaled to GWPACDNet parameter order)
# ─────────────────────────────────────────────────────────────────────────────

print("[BASELINE 4] Physio-Only MLP (scaled, no fairness penalty)")
print("  Purpose: Tests whether a deeper physio-only network breaks the LR ceiling.")

class PhysioMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

torch.manual_seed(RANDOM_SEED)
mlp = PhysioMLP()
optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

X_ph_tr_t  = torch.tensor(scaler_ph.transform(X_ph_tr), dtype=torch.float32)
y_tr_t     = torch.tensor(y_tr, dtype=torch.long)
sc_tr_t    = torch.tensor(sc_tr, dtype=torch.long)
X_ph_val_t = torch.tensor(scaler_ph.transform(X_ph_val), dtype=torch.float32)

dataset_train = torch.utils.data.TensorDataset(X_ph_tr_t, y_tr_t, sc_tr_t)
loader_train  = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True)

criterion = nn.CrossEntropyLoss()

# Train 50 epochs
mlp.train()
for epoch in range(50):
    for xb, yb, _ in loader_train:
        optimizer.zero_grad()
        loss = criterion(mlp(xb), yb)
        loss.backward()
        optimizer.step()
    scheduler.step()

mlp.eval()
with torch.no_grad():
    logits = mlp(X_ph_val_t)
    preds_mlp = logits.argmax(dim=1).numpy()

acc_mlp = accuracy_score(y_val, preds_mlp)
dp_mlp  = compute_dp_gap(preds_mlp, sc_val)
print_result("Physio-Only MLP", acc_mlp, dp_mlp, "(64-dim, 50 epochs, no penalty)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE (Publication-Ready)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 75)
print("  ABLATION SUMMARY TABLE (Publication-Ready)")
print("=" * 75)
print(f"  {'Model':<35} | {'Acc':>6} | {'DP Gap':>8} | {'Pareto':>8} | Modality")
print("  " + "-" * 73)

rows = [
    ("Majority Class Oracle",          acc_maj,    dp_maj,    "None"),
    ("Scar-Only LR (shortcut oracle)", acc_scar,   dp_scar,   "Visual (scar only)"),
    ("Physio-Only LR  [KEY]",          acc_physio, dp_physio, "Physio (HRV+GSR)"),
    ("Physio-Only MLP",                acc_mlp,    dp_mlp,    "Physio (HRV+GSR)"),
    ("Full Tabular LR",                acc_all,    dp_all,    "Physio+Scar"),
    ("GWPACDNet lam=2.0 (CPU, ep20)",  0.6740,     0.5895,    "Multimodal (full)"),
    ("GWPACDNet lam=5.0 (GPU, ep30)",  0.6787,     0.5675,    "Multimodal (full)"),
]
for name, acc, dp, mod in rows:
    ps = pareto_score(acc, dp)
    print(f"  {name:<35} | {acc:>6.4f} | {dp:>8.4f} | {ps:>8.4f} | {mod}")

print()
print("  KEY FINDING:")
gap = dp_physio
if gap < 0.15:
    print(f"  Physio-only DP Gap = {gap:.4f}. This CONFIRMS that physiological signals")
    print(f"  alone achieve near-demographic-parity. The visual scar feature is the")
    print(f"  DOMINANT and PRIMARY source of all fairness degradation in GWPACDNet.")
    print(f"  This is the foundational empirical evidence for the DR-PS-ZOCR framework.")
elif gap < 0.40:
    print(f"  Physio-only DP Gap = {gap:.4f}. Physiological signals carry bias too,")
    print(f"  but substantially less than the scar-fused model (0.55).")
    print(f"  The visual scar is the dominant (but not sole) bias source.")
else:
    print(f"  Physio-only DP Gap = {gap:.4f}. Both modalities carry similar bias levels.")
    print(f"  The HRV/GSR features themselves correlate with scar-adjacent states.")
print("=" * 75)
