import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings

# Suppress harmless warnings if we do hit boundaries in CV, but we aim for max_iter=5000
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def run_rigorous_k1_baseline(csv_path):
    print("=== RIGOROUS K1 BASELINE (PHYSIOLOGY ONLY) ===")
    df = pd.read_csv(csv_path)
    
    # Features and Target
    X = df[['hrv_rmssd', 'hrv_sdnn', 'gsr_mean', 'gsr_std']].values
    y = df['threat'].values
    groups = df['subject'].values
    
    logo = LeaveOneGroupOut()
    
    accs, aucs, f1s = [], [], []
    
    fold = 1
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Ensure classes are present in test set to calculate AUC
        if len(np.unique(y_test)) < 2:
            continue
            
        # Z-score standardization (strictly fit on training to avoid leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # We perform an internal grid search to prevent under-optimization (straw-manning)
        # Using a subset of parameters to keep it computationally feasible within minutes
        param_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64), (64, 64, 32)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.005]
        }
        
        base_mlp = MLPClassifier(max_iter=5000, random_state=42, early_stopping=True, n_iter_no_change=20)
        
        # Grouped by nothing inside the grid search since it's just tuning on the current train split
        clf = GridSearchCV(base_mlp, param_grid, cv=3, n_jobs=-1, scoring='roc_auc')
        clf.fit(X_train_scaled, y_train)
        
        best_model = clf.best_estimator_
        
        preds = best_model.predict(X_test_scaled)
        probs = best_model.predict_proba(X_test_scaled)[:, 1]
        
        accs.append(accuracy_score(y_test, preds))
        aucs.append(roc_auc_score(y_test, probs))
        f1s.append(f1_score(y_test, preds))
        
        print(f"Fold {fold} | Subject {groups[test_idx][0]} | Best Params: {clf.best_params_} | AUC: {aucs[-1]:.4f}")
        fold += 1
        
    print("\n=== FINAL RIGOROUS BASELINE RESULTS ===")
    print(f"Evaluated {fold-1} subjects using LOSO cross-validation.")
    print(f"Mean Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"Mean AUC      : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"Mean F1-Score : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print("=======================================")
    
    with open("outputs/k1_baseline_rigorous.txt", "w") as f:
        f.write(f"Mean Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}\n")
        f.write(f"Mean AUC      : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}\n")
        f.write(f"Mean F1-Score : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}\n")

if __name__ == "__main__":
    run_rigorous_k1_baseline("data/csv/wesad_windows.csv")
