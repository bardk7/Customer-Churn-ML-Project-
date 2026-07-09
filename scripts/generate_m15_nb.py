import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# M15: Model Interpretation with SHAP
Machine learning models are often seen as "black boxes". To build trust and extract actionable business insights, we need to understand *why* the model makes its predictions. 

We use **SHAP (SHapley Additive exPlanations)** to interpret our best model. SHAP values calculate the exact contribution of each feature to the final prediction."""

code_setup = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
from src.data import load_raw_data
from src.cleaning import clean_data
from src.features import build_features
from src.split import split_data

# Load data
df = build_features(clean_data(load_raw_data("../data/raw")), scale=False)
_, X_test, _, _ = split_data(df)

# Load pipeline and extract the actual model
pipeline = joblib.load("../models/best_churn_model.pkl")
model = pipeline.named_steps['model']

# Note: Since the pipeline only contained SMOTE (which doesn't transform test data),
# X_test can be passed directly to the model.
"""

text_shap = """## SHAP Summary Plot
The summary plot shows the most important features and how their values impact the prediction (higher SHAP value = higher probability of churn)."""

code_shap = """# Initialize JS visualization code
shap.initjs()

# Create explainer and compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot summary
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot - Feature Impacts on Churn")
plt.tight_layout()
plt.show()
"""

text_dependence = """## SHAP Dependence Plots
Dependence plots show the relationship between a single feature and its impact on the prediction, including interactions with other features."""

code_dependence = """# Top 3 features by mean absolute SHAP value
mean_shap = np.abs(shap_values).mean(axis=0)
top_features = X_test.columns[np.argsort(mean_shap)[::-1][:3]]

for feature in top_features:
    shap.dependence_plot(feature, shap_values, X_test, show=False)
    plt.title(f"SHAP Dependence Plot: {feature}")
    plt.tight_layout()
    plt.show()
"""

text_conc = """## Key Takeaways
From the SHAP plots, we can see exactly which customer profiles are most likely to churn. For example, customers with short tenure, high monthly charges, and month-to-month contracts typically have strong positive SHAP values (pushing prediction towards Churn). We will compile these into concrete business recommendations in M16."""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_shap),
    nbf.v4.new_code_cell(code_shap),
    nbf.v4.new_markdown_cell(text_dependence),
    nbf.v4.new_code_cell(code_dependence),
    nbf.v4.new_markdown_cell(text_conc),
]

with open("notebooks/13_model_interpretation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generated.")
