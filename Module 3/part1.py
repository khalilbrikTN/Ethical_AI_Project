# --- Part 1: Baseline Model and Clean Evaluation ---
print("\n--- Part 1: Baseline Model Performance ---")

# Model and data from Module 2
baseline_accuracy = accuracy_score(y_test, y_pred)
baseline_precision = precision_score(y_test, y_pred)
baseline_recall = recall_score(y_test, y_pred)
baseline_f1 = f1_score(y_test, y_pred)

print(f"Baseline Accuracy:  {baseline_accuracy:.4f}")
print(f"Baseline Precision: {baseline_precision:.4f}")
print(f"Baseline Recall:    {baseline_recall:.4f}")
print(f"Baseline F1-Score:  {baseline_f1:.4f}")
print("-" * 40)