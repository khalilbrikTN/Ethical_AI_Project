# --- Part 4: Defense Implementation ---
# Defense: Clip numeric feature values to the 1st and 99th percentiles from the training set.

# Calculate percentiles from the training set
X_train_df = pd.DataFrame(X_train_dense, columns=all_feature_names)
clipping_limits = {}
for col in numeric_features:
    col_idx = all_feature_names.index(col)
    lower = np.percentile(X_train_dense[:, col_idx], 1)
    upper = np.percentile(X_train_dense[:, col_idx], 99)
    clipping_limits[col_idx] = (lower, upper)

def apply_clipping(X_data, limits):
    X_clipped = X_data.copy()
    for idx, (lower, upper) in limits.items():
        X_clipped[:, idx] = np.clip(X_clipped[:, idx], lower, upper)
    return X_clipped

# Apply clipping to the test set and all perturbed sets
X_test_clipped = apply_clipping(X_test_dense, clipping_limits)
X_test_pert1_clipped = apply_clipping(X_test_perturbed_1, clipping_limits)
X_test_pert2_clipped = apply_clipping(X_test_perturbed_2, clipping_limits)
X_test_pert3_clipped = apply_clipping(X_test_perturbed_3, clipping_limits)

# Re-evaluate the model on the clipped datasets
def evaluate_clipped(X_data, y_true, name):
    y_pred = dt_model.predict(X_data)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n--- {name} (with Clipping) ---")
    print(f"Accuracy:  {acc:.4f} (Δ = {acc - results[name][0]:.4f})")
    print(f"Precision: {prec:.4f} (Δ = {prec - results[name][1]:.4f})")
    print(f"Recall:    {rec:.4f} (Δ = {rec - results[name][2]:.4f})")
    print(f"F1-Score:  {f1:.4f} (Δ = {f1 - results[name][3]:.4f})")
    return acc, prec, rec, f1

print("\n--- Evaluation After Defense (Clipping) ---")
# Baseline (clean data) after clipping
results_clipped = {}
results_clipped['Baseline'] = evaluate_clipped(X_test_clipped, y_test, 'Baseline')
# Perturbation 1
results_clipped['Noise (std=0.05)'] = evaluate_clipped(X_test_pert1_clipped, y_test, 'Noise (std=0.05)')
# Perturbation 2
results_clipped['Missing Values (10%)'] = evaluate_clipped(X_test_pert2_clipped, y_test, 'Missing Values (10%)')
# Perturbation 3
results_clipped['Rounding (1 decimal)'] = evaluate_clipped(X_test_pert3_clipped, y_test, 'Rounding (1 decimal)')

print("\n--- Summary of Defense Effectiveness (Δ from original, unclipped performance) ---")
print("Scenario                  | Acc Δ (Orig) | Acc Δ (Clipped) | F1 Δ (Orig) | F1 Δ (Clipped)")
print("-" * 80)
for scenario in ['Noise (std=0.05)', 'Missing Values (10%)', 'Rounding (1 decimal)']:
    acc_orig_delta = results[scenario][0] - baseline_accuracy
    acc_clipped_delta = results_clipped[scenario][0] - baseline_accuracy
    f1_orig_delta = results[scenario][3] - baseline_f1
    f1_clipped_delta = results_clipped[scenario][3] - baseline_f1
    print(f"{scenario:25} | {acc_orig_delta:+.4f}       | {acc_clipped_delta:+.4f}      | {f1_orig_delta:+.4f}     | {f1_clipped_delta:+.4f}")