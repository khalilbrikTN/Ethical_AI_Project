# Ethical AI Project

This repository contains the submissions for the **CSCE 4930: Responsible and Ethical Artificial Intelligence** project.

## Team Members
- Mohamed Khalil Brik (ID: 900225905)
- Adam Aberbach (ID: 900225980)
- Mark Kyrollos (ID: 900211436)
- Abdelrahman Ihab Elazab (ID: 900213468)

## Module 1: Fairness Audit & Bias Mitigation

- **`CSCE4930_Ethical_AI_Project_Module_1_German_Credit_Dataset_Completed.ipynb`**: The fully executed Jupyter Notebook containing baseline Model Training, Preprocessing Bias Mitigation (Resampling), and In-processing Bias Mitigation (Exponentiated Gradient) on the German Credit Dataset.
- **`report.tex` / `report.pdf`**: The synthesized technical report reflecting deep interpretations of our parity differences, model accuracy, and real-world implications of the dataset.

### Highlights
Our analysis prioritizes **Equalized Odds** given the allocative harms inherent in credit decisions. Our model mitigations showcase significant fairness improvements, effectively neutralizing the Demographic Parity gap while managing the accuracy/fairness trade-off constraints.

## Module 2: Explainability System Design

- **`Module 2/CSCE4930_Ethical_AI_Project_Module_2_German_Credit_Dataset.ipynb`**: Fully executed notebook building an interpretable Decision Tree (max_depth=5) on the German Credit Dataset. Covers feature importance analysis, split-level directional effects, local explanations for two individuals, and suspicious pattern detection across protected attributes.
- **`Module 2/EAI_Project_Module_2_report.pdf`**: Technical report answering all seven module questions with direct references to notebook figures and tables.
- **Figures**: `feature_importance.png`, `decision_tree.png`, `feature_effects.png`, `individual_comparison.png`, `disparity_analysis.png`, `confusion_by_group.png`

### Highlights
The Decision Tree achieves 68.5% accuracy while providing full transparency into its decisions. Our analysis uncovered that the tree directly splits on protected attributes (age, sex) accounting for 6% of total feature importance. Prediction rate disparities show males approved at 90% vs. females at 75%, and younger applicants face a 50% error rate. We recommend against deployment without fairness mitigation.
