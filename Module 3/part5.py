# --- Part 5: Comprehensive Analysis After Defense ---
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# Store all results in a structured format
summary_data = []

# Define scenarios
scenarios = [
    ("Baseline (Clean Data)", y_test, y_pred, 
     dt_model.predict(X_test_clipped)),
    ("Noise (std=0.05)", y_test, y_pred_pert1, 
     dt_model.predict(X_test_pert1_clipped)),
    ("Missing Values (10%)", y_test, y_pred_pert2, 
     dt_model.predict(X_test_pert2_clipped)),
    ("Rounding (1 decimal)", y_test, y_pred_pert3, 
     dt_model.predict(X_test_pert3_clipped))
]

for name, y_true, y_pred_original, y_pred_defense in scenarios:
    # Original performance
    acc_orig = accuracy_score(y_true, y_pred_original)
    prec_orig = precision_score(y_true, y_pred_original)
    rec_orig = recall_score(y_true, y_pred_original)
    f1_orig = f1_score(y_true, y_pred_original)
    
    # Performance after defense
    acc_def = accuracy_score(y_true, y_pred_defense)
    prec_def = precision_score(y_true, y_pred_defense)
    rec_def = recall_score(y_true, y_pred_defense)
    f1_def = f1_score(y_true, y_pred_defense)
    
    # Fairness metrics (using age_group)
    dp_orig = demographic_parity_difference(y_true, y_pred_original, sensitive_features=A_test['age_group'].values)
    eo_orig = equalized_odds_difference(y_true, y_pred_original, sensitive_features=A_test['age_group'].values)
    
    dp_def = demographic_parity_difference(y_true, y_pred_defense, sensitive_features=A_test['age_group'].values)
    eo_def = equalized_odds_difference(y_true, y_pred_defense, sensitive_features=A_test['age_group'].values)
    
    # Error rates by group
    err_rates_orig = {}
    err_rates_def = {}
    for group in np.unique(A_test['age_group'].values):
        mask = A_test['age_group'].values == group
        err_rates_orig[group] = 1 - accuracy_score(y_true[mask], y_pred_original[mask])
        err_rates_def[group] = 1 - accuracy_score(y_true[mask], y_pred_defense[mask])
    
    summary_data.append({
        'Scenario': name,
        'Acc_Orig': acc_orig, 'Acc_Def': acc_def, 'Acc_Change': acc_def - acc_orig,
        'Prec_Orig': prec_orig, 'Prec_Def': prec_def, 'Prec_Change': prec_def - prec_orig,
        'Rec_Orig': rec_orig, 'Rec_Def': rec_def, 'Rec_Change': rec_def - rec_orig,
        'F1_Orig': f1_orig, 'F1_Def': f1_def, 'F1_Change': f1_def - f1_orig,
        'DP_Orig': dp_orig, 'DP_Def': dp_def, 'DP_Change': dp_def - dp_orig,
        'EO_Orig': eo_orig, 'EO_Def': eo_def, 'EO_Change': eo_def - eo_orig,
        'Err_Under25_Orig': err_rates_orig.get('under_25', 0),
        'Err_Under25_Def': err_rates_def.get('under_25', 0),
        'Err_Over25_Orig': err_rates_orig.get('age_25_and_over', 0),
        'Err_Over25_Def': err_rates_def.get('age_25_and_over', 0),
    })

# Create summary DataFrame
summary_df = pd.DataFrame(summary_data)
print("\n" + "="*100)
print("COMPREHENSIVE PERFORMANCE & FAIRNESS SUMMARY")
print("="*100)
print(summary_df.to_string(index=False))