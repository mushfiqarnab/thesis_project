import numpy as np
import pandas as pd

def stratified_bootstrap_dp_gap():
    np.random.seed(42)
    df = pd.read_csv(r"C:\Users\USERAS\thesis_project\outputs\pilot_checkpoints\oof_preds_rnd.csv")
    A = df['a_true'].values
    Y_pred = df['prob'].values
    
    idx_scar = np.where(A == 1)[0]
    idx_no_scar = np.where(A == 0)[0]
    
    n_iterations = 10000
    dp_gaps = np.zeros(n_iterations)
    
    print(f"Running {n_iterations} Stratified Bootstrap Iterations on REAL Data...")
    for i in range(n_iterations):
        boot_idx_scar = np.random.choice(idx_scar, size=len(idx_scar), replace=True)
        boot_idx_no_scar = np.random.choice(idx_no_scar, size=len(idx_no_scar), replace=True)
        
        e_y_scar = np.mean(Y_pred[boot_idx_scar])
        e_y_no_scar = np.mean(Y_pred[boot_idx_no_scar])
        
        dp_gaps[i] = np.abs(e_y_scar - e_y_no_scar)
        
    ci_lower = np.percentile(dp_gaps, 2.5)
    ci_upper = np.percentile(dp_gaps, 97.5)
    mean_gap = np.mean(dp_gaps)
    
    print("\n--- Soft Demographic Parity (DP) Gap Analysis ---")
    print(f"Mean Soft DP Gap: {mean_gap:.5f}")
    print(f"95% Confidence Interval: [{ci_lower:.5f}, {ci_upper:.5f}]")
    
    if ci_upper < 0.10:
        print("\nStatistical Proof Successful: The upper bound of the 95% CI is highly constrained.")
    else:
        print("\nStatistical Proof Failed: The upper bound of the 95% CI exceeds 0.10.")

if __name__ == "__main__":
    stratified_bootstrap_dp_gap()
