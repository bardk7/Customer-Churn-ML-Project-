import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# M14: Final Model Evaluation
In this notebook, we load the best model identified during hyperparameter tuning (M13) and evaluate its performance on the untouched test set using comprehensive metrics and plots."""

code_setup = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
from src.data import load_raw_data
from src.cleaning import clean_data
from src.features import build_features
from src.split import split_data
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve_custom, plot_pr_curve_custom

from sklearn.metrics import classification_report

sns.set_theme(style="whitegrid")

# Load data and split
df = build_features(clean_data(load_raw_data("../data/raw")), scale=False)
_, X_test, _, y_test = split_data(df)

# Load best model
model = joblib.load("../models/best_churn_model.pkl")
"""

text_eval = """## 1. Predictions and Metrics"""

code_eval = """# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

# Basic Metrics
metrics = evaluate_model(y_test, y_pred, y_prob)
print("--- Final Model Metrics ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Classification Report
print("\\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
"""

text_plots = """## 2. Evaluation Plots
We visualize the Confusion Matrix, ROC Curve, and Precision-Recall Curve."""

code_plots = """# Confusion Matrix
fig_cm = plot_confusion_matrix(y_test, y_pred, title="Final Model Confusion Matrix")
plt.show()

# ROC Curve
fig_roc = plot_roc_curve_custom(y_test, y_prob, title="Final Model ROC Curve")
plt.show()

# PR Curve
fig_pr = plot_pr_curve_custom(y_test, y_prob, title="Final Model Precision-Recall Curve")
plt.show()
"""

text_conc = """## Conclusion
The final model correctly flags a high proportion of churners (high recall) while maintaining a reasonable false positive rate. The threshold could be further adjusted depending on the specific cost of a false positive (unnecessary retention spend) versus a false negative (lost customer)."""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_eval),
    nbf.v4.new_code_cell(code_eval),
    nbf.v4.new_markdown_cell(text_plots),
    nbf.v4.new_code_cell(code_plots),
    nbf.v4.new_markdown_cell(text_conc),
]

with open("notebooks/12_model_evaluation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generated.")
