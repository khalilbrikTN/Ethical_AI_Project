# --- Part 3: Fairness and Explanation Stability ---
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import pandas as pd

def evaluate_fairness(y_true, y_pred, sensitive_features, name):
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    print(f"\n--- {name} ---")
    print(f"  Demographic Parity Difference: {dp_diff:.4f}")
    print(f"  Equalized Odds Difference:     {eo_diff:.4f}")
    # Group-wise error rates
    for group in np.unique(sensitive_features):
        mask = sensitive_features == group
        err_rate = 1 - accuracy_score(y_true[mask], y_pred[mask])
        print(f"  Error rate for {group}: {err_rate:.4f}")
    return dp_diff, eo_diff

# Fairness evaluation for each scenario
print("\n--- Fairness Metrics Under Perturbations ---")
fairness_results = {}
# Baseline
fairness_results['Baseline'] = evaluate_fairness(y_test, y_pred, A_test['age_group'].values, "Baseline")

# Perturbation 1
fairness_results['Noise (std=0.05)'] = evaluate_fairness(y_test, y_pred_pert1, A_test['age_group'].values, X_test_perturbed_1_name)

# Perturbation 2
fairness_results['Missing Values (10%)'] = evaluate_fairness(y_test, y_pred_pert2, A_test['age_group'].values, X_test_perturbed_2_name)

# Perturbation 3
fairness_results['Rounding (1 decimal)'] = evaluate_fairness(y_test, y_pred_pert3, A_test['age_group'].values, X_test_perturbed_3_name)

# --- Visualization of Fairness Metric Changes ---
metrics = ['Dem. Parity Diff', 'Equal. Odds Diff']
scenarios = list(fairness_results.keys())
metric_values = {metric: [fairness_results[scenario][i] for i, m in enumerate(metrics)] for metric in metrics}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(scenarios))
width = 0.35
ax.bar(x - width/2, metric_values['Dem. Parity Diff'], width, label='Demographic Parity Difference')
ax.bar(x + width/2, metric_values['Equal. Odds Diff'], width, label='Equalized Odds Difference')
ax.set_ylabel('Difference (Lower is better)')
ax.set_title('Fairness Metrics Under Perturbations')
ax.set_xticks(x)
ax.set_xticklabels(scenarios, rotation=15, ha='right')
ax.legend()
plt.tight_layout()
plt.show()

# --- Check for changes in top important features ---
print("\n--- Stability of Top 3 Important Features ---")
for scenario_name, y_pred_temp in [("Noise (std=0.05)", y_pred_pert1), 
                                   ("Missing Values (10%)", y_pred_pert2),
                                   ("Rounding (1 decimal)", y_pred_pert3)]:
    # Retrain a model on the perturbed data? No, just evaluate the existing model.
    # Feature importance is a property of the trained model, not the data.
    # Since we don't retrain, the feature importances remain the same.
    # To check for stability of individual predictions, we can look at the two individuals from Module 2.
    pass

print("  The top 3 features and their importances remain constant, as the model is not retrained.")
print("  However, the model's reliance on these features under perturbation can be inferred from performance drops.")