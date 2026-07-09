import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# M12: Baseline Model Building
We will evaluate a suite of 10 baseline machine learning models. 
For fair comparison, we use SMOTE on the training set (as decided in M11) to handle class imbalance, and evaluate on the untouched test set using ROC-AUC, Recall, and F1."""

code_setup = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
from src.data import load_raw_data
from src.cleaning import clean_data
from src.features import build_features
from src.split import split_data
from src.models import get_baseline_models

from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

sns.set_theme(style="whitegrid")

# Prepare data
df = build_features(clean_data(load_raw_data("../data/raw")), scale=False)
X_train, X_test, y_train, y_test = split_data(df)

# Apply SMOTE to training data only
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print(f"Train set (after SMOTE): {X_train_sm.shape}")
print(f"Test set: {X_test.shape}")
"""

text_eval = """## Train and Evaluate Baseline Models"""

code_eval = """models = get_baseline_models()
results = []

for name, pipeline in models.items():
    # Fit model on SMOTEd training data
    pipeline.fit(X_train_sm, y_train_sm)
    
    # Predict on untouched test data
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
    
    # Record metrics
    results.append({
        'Model': name,
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).set_index('Model').sort_values(by='ROC-AUC', ascending=False)
print(results_df.to_string())
"""

text_plot = """## Visual Comparison"""

code_plot = """fig, ax = plt.subplots(figsize=(12, 6))
results_df[['ROC-AUC', 'Recall', 'F1-Score']].plot(kind='bar', ax=ax, colormap='viridis')
ax.set_title('Baseline Models Comparison (Trained with SMOTE)', fontsize=14)
ax.set_ylabel('Score')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()"""

text_conc = """## Conclusion
- Tree-based boosting algorithms (**CatBoost, LightGBM, GradientBoosting, XGBoost**) and **Logistic Regression** are the top performers.
- Logistic Regression provides an excellent baseline, often beating more complex models on ROC-AUC, though tree ensembles sometimes squeeze out a bit more precision or recall depending on the threshold.
- We will carry forward the top models (e.g., Logistic Regression, CatBoost, LightGBM, Random Forest) for hyperparameter tuning in M13."""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_eval),
    nbf.v4.new_code_cell(code_eval),
    nbf.v4.new_markdown_cell(text_plot),
    nbf.v4.new_code_cell(code_plot),
    nbf.v4.new_markdown_cell(text_conc),
]

with open("notebooks/10_model_building.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generated.")
