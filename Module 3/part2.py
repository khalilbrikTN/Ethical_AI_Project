# --- Part 2: Robustness Stress Testing ---
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from copy import deepcopy

# Helper function to evaluate and print results
def evaluate_perturbation(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f} (Δ = {acc - baseline_accuracy:.4f})")
    print(f"Precision: {prec:.4f} (Δ = {prec - baseline_precision:.4f})")
    print(f"Recall:    {rec:.4f} (Δ = {rec - baseline_recall:.4f})")
    print(f"F1-Score:  {f1:.4f} (Δ = {f1 - baseline_f1:.4f})")
    return acc, prec, rec, f1

# Create copies of the dense test data to perturb
X_test_perturbed_1 = X_test_dense.copy()
X_test_perturbed_2 = X_test_dense.copy()
X_test_perturbed_3 = X_test_dense.copy()

# Perturbation 1: Add Gaussian noise (mean=0, std=0.05) to all numeric features
np.random.seed(42)
num_cols_indices = [all_feature_names.index(col) for col in numeric_features]
for idx in num_cols_indices:
    noise = np.random.normal(0, 0.05, size=X_test_perturbed_1.shape[0])
    X_test_perturbed_1[:, idx] += noise
# Rename for clarity
X_test_perturbed_1_name = "Add Gaussian Noise (std=0.05)"

# Perturbation 2: Introduce missing values (10% random) and re-impute
# We'll re-run the original preprocessor on the corrupted data.
X_test_corrupted = X_test.copy() # Use the original DataFrame
np.random.seed(42)
for col in X_test_corrupted.columns:
    if X_test_corrupted[col].dtype in ['int64', 'float64']:
        missing_idx = np.random.choice(X_test_corrupted.index, size=int(0.1 * len(X_test_corrupted)), replace=False)
        X_test_corrupted.loc[missing_idx, col] = np.nan
    else:
        # For categorical, we can also randomly set to NaN
        missing_idx = np.random.choice(X_test_corrupted.index, size=int(0.1 * len(X_test_corrupted)), replace=False)
        X_test_corrupted.loc[missing_idx, col] = np.nan

# Re-impute and transform
X_test_perturbed_2 = preprocessor.transform(X_test_corrupted).toarray()
X_test_perturbed_2_name = "10% Missing Values with Re-imputation"

# Perturbation 3: Round numeric features to 1 decimal place (reduce precision)
for idx in num_cols_indices:
    X_test_perturbed_3[:, idx] = np.round(X_test_perturbed_3[:, idx], 1)
X_test_perturbed_3_name = "Round Numeric Features to 1 Decimal"

# Evaluate on perturbed datasets
print("\n--- Model Evaluation Under Perturbations ---")
results = {}
results['Baseline'] = (baseline_accuracy, baseline_precision, baseline_recall, baseline_f1)

# Perturbation 1
y_pred_pert1 = dt_model.predict(X_test_perturbed_1)
results['Noise (std=0.05)'] = evaluate_perturbation(y_test, y_pred_pert1, X_test_perturbed_1_name)

# Perturbation 2
y_pred_pert2 = dt_model.predict(X_test_perturbed_2)
results['Missing Values (10%)'] = evaluate_perturbation(y_test, y_pred_pert2, X_test_perturbed_2_name)

# Perturbation 3
y_pred_pert3 = dt_model.predict(X_test_perturbed_3)
results['Rounding (1 decimal)'] = evaluate_perturbation(y_test, y_pred_pert3, X_test_perturbed_3_name)

# --- Visualization of Robustness Test Results ---
import matplotlib.pyplot as plt
import seaborn as sns

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scenarios = list(results.keys())
metric_values = {metric: [results[scenario][i] for scenario in scenarios] for i, metric in enumerate(metrics)}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(scenarios))
width = 0.2
multiplier = 0

for metric, values in metric_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, values, width, label=metric)
    multiplier += 1

ax.set_ylabel('Score')
ax.set_title('Model Performance Under Different Perturbations')
ax.set_xticks(x + width)
ax.set_xticklabels(scenarios, rotation=15, ha='right')
ax.legend(loc='lower right')
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()