"""Script 2: Stratified Bootstrap DP Gap CI on Real Model Predictions"""
import numpy as np
import pandas as pd

np.random.seed(42)

print("=" * 60)
print("SCRIPT 2: STRATIFIED BOOTSTRAP ON REAL MODEL PREDICTIONS")
print("=" * 60)

# Load all three OOF prediction files
files = [
    ("outputs/pilot_checkpoints/oof_preds_rnd.csv", "Random-init model"),
    ("outputs/pilot_checkpoints/oof_preds_ext.csv", "Extended model"),
    ("outputs/pilot_checkpoints/oof_preds_mod.csv", "Modality model"),
]

for fpath, label in files:
    print()
    print(f"--- {label} ({fpath}) ---")
    try:
        df = pd.read_csv(fpath)
        print(f"Samples: {len(df)}")

        y_true = df['y_true'].values
        y_pred = df['prob'].values
        a_true = (df['a_true'].values >= 0.5).astype(int)

        idx_scar   = np.where(a_true == 1)[0]
        idx_noscar = np.where(a_true == 0)[0]

        print(f"Class distribution: {dict(zip(*np.unique(y_true, return_counts=True)))}")
        scar_ct = int(a_true.sum())
        print(f"Scar group: {scar_ct}, No-scar group: {len(a_true)-scar_ct}")

        e_scar   = np.mean(y_pred[idx_scar])
        e_noscar = np.mean(y_pred[idx_noscar])
        point_dp = abs(e_scar - e_noscar)

        print(f"E[y_hat | A=1 (scar)]:    {e_scar:.5f}")
        print(f"E[y_hat | A=0 (no scar)]: {e_noscar:.5f}")
        print(f"Point Soft DP Gap:        {point_dp:.5f}")

        # 10,000-iteration Stratified Bootstrap
        n_iter = 10000
        gaps = np.zeros(n_iter)
        for i in range(n_iter):
            bs  = np.random.choice(idx_scar,   len(idx_scar),   replace=True)
            bns = np.random.choice(idx_noscar, len(idx_noscar), replace=True)
            gaps[i] = abs(np.mean(y_pred[bs]) - np.mean(y_pred[bns]))

        ci_lo  = np.percentile(gaps, 2.5)
        ci_hi  = np.percentile(gaps, 97.5)
        mean_g = np.mean(gaps)

        print(f"Bootstrap Mean DP Gap:    {mean_g:.5f}")
        print(f"95% CI:                   [{ci_lo:.5f}, {ci_hi:.5f}]")
        print(f"CI Width:                 {ci_hi - ci_lo:.5f}")
        verdict = "PASS (statistically constrained)" if ci_hi < 0.1 else "WIDE (high uncertainty)"
        print(f"Verdict:                  {verdict}")

    except Exception as e:
        print(f"ERROR: {e}")
