import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def run_k1_baseline(csv_path):
    print("=== K1 BASELINE (PHYSIOLOGY ONLY) ===")
    df = pd.read_csv(csv_path)
    
    # Features and Target
    X = df[['hrv_rmssd', 'hrv_sdnn', 'gsr_mean', 'gsr_std']].values
    y = df['threat'].values
    groups = df['subject'].values
    
    logo = LeaveOneGroupOut()
    
    accs = []
    aucs = []
    f1s = []
    
    fold = 1
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Ensure classes are present in test set to calculate AUC (some subjects might only have 1 class? WESAD usually has both)
        if len(np.unique(y_test)) < 2:
            continue
            
        # Z-score standardization (strictly fit on training to avoid leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Standard MLP
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        
        preds = mlp.predict(X_test_scaled)
        probs = mlp.predict_proba(X_test_scaled)[:, 1]
        
        accs.append(accuracy_score(y_test, preds))
        aucs.append(roc_auc_score(y_test, probs))
        f1s.append(f1_score(y_test, preds))
        fold += 1
        
    print(f"Evaluated {fold-1} subjects using LOSO cross-validation.")
    print(f"Mean Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"Mean AUC      : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"Mean F1-Score : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print("=====================================")

if __name__ == "__main__":
    run_k1_baseline("data/csv/wesad_windows.csv")
