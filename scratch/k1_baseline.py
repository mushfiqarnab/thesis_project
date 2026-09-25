import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def run_k1_baseline():
    df = pd.read_csv('data/csv/wesad_windows.csv')
    
    X = df[['hrv_rmssd', 'hrv_sdnn', 'gsr_mean', 'gsr_std']].values
    y = df['threat'].values
    groups = df['subject'].values

    logo = LeaveOneGroupOut()
    
    accuracies = []
    aucs = []
    f1s = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Standard MLP baseline (mimicking a physiological feature extractor branch)
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        
        y_pred = mlp.predict(X_test_scaled)
        y_proba = mlp.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros_like(y_test)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        
        if len(np.unique(y_test)) > 1:
            aucs.append(roc_auc_score(y_test, y_proba))

    print(f"--- K1 Baseline (Physiology-Only) ---")
    print(f"Model: MLP (64, 32) on HRV/GSR features")
    print(f"Validation Strategy: Leave-One-Subject-Out (N={len(np.unique(groups))})")
    print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

if __name__ == '__main__':
    run_k1_baseline()
