"""
Comprehensive ROC, AUC, Confusion Matrix, and Correlation Analysis
Using sklearn with multiple model architectures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 80)
print("COMPREHENSIVE ROC, AUC, AND CONFUSION MATRIX ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/6] Loading dataset...")
df = pd.read_csv('data/csv/multimodal_10k_unbiased.csv')
print(f"Dataset shape: {df.shape}")
print(f"Features: {df.columns.tolist()}")

# Select relevant features for analysis
# Using only numeric features: hrv, gsr, and threat label
feature_cols = ['hrv', 'gsr']
target_col = 'threat'

print(f"\nUsing features: {feature_cols}")
print(f"Target: {target_col}")

# Create X and y
X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Class distribution:\n{y.value_counts()}")

# ============================================================================
# 2. PREPROCESSING & FEATURE SCALING
# ============================================================================
print("\n[2/6] Preprocessing and scaling features...")

# Handle missing values and scale
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

X_processed = preprocessor.fit_transform(X)
X_processed = pd.DataFrame(X_processed, columns=feature_cols)

print(f"Processed shape: {X_processed.shape}")
print(f"\nFeature statistics AFTER scaling:")
print(X_processed.describe())

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n[3/6] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nTrain class distribution:\n{y_train.value_counts()}")
print(f"\nTest class distribution:\n{y_test.value_counts()}")

# ============================================================================
# 4. TRAIN MULTIPLE MODELS
# ============================================================================
print("\n[4/6] Training models...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
}

predictions = {}
probabilities = {}
metrics_dict = {}

for model_name, model in models.items():
    print(f"  - Training {model_name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Store results
    predictions[model_name] = y_pred
    probabilities[model_name] = y_pred_proba
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics_dict[model_name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'AUC-ROC': auc
    }
    
    print(f"    ✓ Accuracy: {acc:.4f}, AUC-ROC: {auc:.4f}")

# ============================================================================
# 5. CREATE COMPARISON TABLES
# ============================================================================
print("\n[5/6] Creating comparison tables...")

# Metrics DataFrame
metrics_df = pd.DataFrame(metrics_dict).T
print("\n" + "=" * 80)
print("METRICS COMPARISON TABLE")
print("=" * 80)
print(metrics_df.to_string())

# Save metrics table to CSV
metrics_df.to_csv('outputs/analysis/metrics_comparison_table.csv')
print("\n✓ Saved: outputs/analysis/metrics_comparison_table.csv")

# ============================================================================
# 6. GENERATE VISUALIZATIONS
# ============================================================================
print("\n[6/6] Generating visualizations...")

# ========== FIGURE 1: ROC CURVES COMPARISON ==========
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for (model_name, y_pred_proba), color in zip(probabilities.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2.5, color=color)

# Add diagonal line (random classifier)
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5000)', linewidth=2)

ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves Comparison - All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('outputs/analysis/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/analysis/roc_curves_comparison.png")
plt.close()

# ========== FIGURE 2: AUC-ROC BAR CHART ==========
fig, ax = plt.subplots(figsize=(10, 6))

auc_scores = {name: metrics_dict[name]['AUC-ROC'] for name in metrics_dict.keys()}
names = list(auc_scores.keys())
scores = list(auc_scores.values())

bars = ax.bar(names, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('AUC-ROC Score', fontsize=12, fontweight='bold')
ax.set_title('AUC-ROC Comparison - All Models', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
ax.axhline(y=0.5, color='red', linestyle='--', label='Random (0.5)', linewidth=2)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/analysis/auc_roc_comparison_bars.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/analysis/auc_roc_comparison_bars.png")
plt.close()

# ========== FIGURE 3: CONFUSION MATRICES ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (model_name, ax) in enumerate(zip(predictions.keys(), axes)):
    cm = confusion_matrix(y_test, predictions[model_name])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                cbar=True, annot_kws={'fontsize': 12, 'fontweight': 'bold'},
                xticklabels=['Safe', 'Threat'],
                yticklabels=['Safe', 'Threat'])
    
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name}\nAccuracy: {metrics_dict[model_name]["Accuracy"]:.4f}',
                fontsize=11, fontweight='bold')

plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/analysis/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/analysis/confusion_matrices_comparison.png")
plt.close()

# ========== FIGURE 4: METRICS COMPARISON HEATMAP ==========
fig, ax = plt.subplots(figsize=(10, 6))

# Create heatmap
sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax, 
            cbar_kws={'label': 'Score'},
            annot_kws={'fontsize': 10, 'fontweight': 'bold'},
            vmin=0, vmax=1, linewidths=0.5)

ax.set_title('Metrics Comparison Heatmap - All Models', fontsize=14, fontweight='bold')
ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Model', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/analysis/metrics_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/analysis/metrics_heatmap.png")
plt.close()

# ========== FIGURE 5: ALL METRICS COMPARISON ==========
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
colors_grad = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, metric in enumerate(metric_names):
    ax = axes[idx]
    values = [metrics_dict[name][metric] for name in metrics_dict.keys()]
    
    bars = ax.bar(list(metrics_dict.keys()), values, color=colors_grad, 
                   alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(metric, fontsize=12, fontweight='bold')

# Remove the extra subplot
axes[-1].remove()

plt.suptitle('All Metrics Comparison - Models vs Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/analysis/all_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/analysis/all_metrics_comparison.png")
plt.close()

# ========== FIGURE 6: CORRELATION HEATMAP (FEATURES) ==========
fig, ax = plt.subplots(figsize=(8, 6))

# Calculate correlation matrix of features + target
correlation_data = X_processed.copy()
correlation_data[target_col] = y

corr_matrix = correlation_data.corr()

sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='coolwarm', ax=ax,
            center=0, square=True, linewidths=1,
            annot_kws={'fontsize': 10, 'fontweight': 'bold'},
            cbar_kws={'label': 'Correlation'})

ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/analysis/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/analysis/correlation_heatmap.png")
plt.close()

# ========== FIGURE 7: DETAILED CLASSIFICATION REPORTS ==========
fig = plt.figure(figsize=(14, 10))

for idx, model_name in enumerate(predictions.keys()):
    ax = plt.subplot(2, 2, idx + 1)
    
    # Get classification report
    report = classification_report(y_test, predictions[model_name], 
                                   target_names=['Safe', 'Threat'],
                                   output_dict=True)
    
    # Create table data
    report_df = pd.DataFrame(report).iloc[:-1, :].T
    
    # Create table
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=report_df.values.round(4),
                    colLabels=report_df.columns,
                    rowLabels=report_df.index,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15]*len(report_df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(report_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title(f'{model_name}\nClassification Report', 
                fontweight='bold', fontsize=11, pad=10)

plt.suptitle('Detailed Classification Reports - All Models', 
            fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('outputs/analysis/classification_reports.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/analysis/classification_reports.png")
plt.close()

# ============================================================================
# PRINT SUMMARY REPORTS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY REPORT - ALL MODELS")
print("=" * 80)

for model_name in metrics_dict.keys():
    print(f"\n{model_name.upper()}")
    print("-" * 80)
    
    metrics = metrics_dict[model_name]
    cm = confusion_matrix(y_test, predictions[model_name])
    
    print(f"Accuracy:  {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['Precision']:.4f} ({metrics['Precision']*100:.2f}%)")
    print(f"Recall:    {metrics['Recall']:.4f} ({metrics['Recall']*100:.2f}%)")
    print(f"F1 Score:  {metrics['F1 Score']:.4f}")
    print(f"AUC-ROC:   {metrics['AUC-ROC']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (Safe):      {cm[0, 0]}")
    print(f"  False Positives (Safe→Thr): {cm[0, 1]}")
    print(f"  False Negatives (Thr→Safe): {cm[1, 0]}")
    print(f"  True Positives (Threat):    {cm[1, 1]}")

# ============================================================================
# SAVE DETAILED REPORT TO JSON
# ============================================================================
import json

report_json = {}
for model_name in metrics_dict.keys():
    cm = confusion_matrix(y_test, predictions[model_name])
    report_json[model_name] = {
        'metrics': metrics_dict[model_name],
        'confusion_matrix': {
            'TN': int(cm[0, 0]),
            'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0]),
            'TP': int(cm[1, 1])
        }
    }

with open('outputs/analysis/sklearn_comprehensive_report.json', 'w') as f:
    json.dump(report_json, f, indent=4)

print("\n✓ Saved: outputs/analysis/sklearn_comprehensive_report.json")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  📊 Visualizations:")
print("     - roc_curves_comparison.png")
print("     - auc_roc_comparison_bars.png")
print("     - confusion_matrices_comparison.png")
print("     - metrics_heatmap.png")
print("     - all_metrics_comparison.png")
print("     - correlation_heatmap.png")
print("     - classification_reports.png")
print("  📋 Data Tables:")
print("     - metrics_comparison_table.csv")
print("     - sklearn_comprehensive_report.json")
print("\nAll files saved to: outputs/analysis/")
print("=" * 80)
