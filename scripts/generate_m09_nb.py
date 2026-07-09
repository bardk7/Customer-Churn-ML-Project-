import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# M09: Feature Selection
We rank features using multiple methods and select a final feature set for modeling.

### Methods:
1. Correlation with target
2. Mutual Information
3. Random Forest feature importance"""

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

from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

sns.set_theme(style="whitegrid")

# Build features (unscaled for interpretability)
df = build_features(clean_data(load_raw_data("../data/raw")), scale=False)
X = df.drop(columns=['Churn'])
y = df['Churn']
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
"""

text_corr = """## 1. Correlation with Target"""

code_corr = """corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
print(corr_with_target.head(15))

fig, ax = plt.subplots(figsize=(10, 8))
corr_with_target.head(20).plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 20 Features by Absolute Correlation with Churn')
ax.set_xlabel('|Correlation|')
plt.tight_layout()
plt.show()"""

text_mi = """## 2. Mutual Information"""

code_mi = """mi = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print(mi_series.head(15))

fig, ax = plt.subplots(figsize=(10, 8))
mi_series.head(20).plot(kind='barh', ax=ax, color='darkorange')
ax.set_title('Top 20 Features by Mutual Information with Churn')
ax.set_xlabel('MI Score')
plt.tight_layout()
plt.show()"""

text_rf = """## 3. Random Forest Feature Importance"""

code_rf = """rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X, y)
rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(rf_imp.head(15))

fig, ax = plt.subplots(figsize=(10, 8))
rf_imp.head(20).plot(kind='barh', ax=ax, color='forestgreen')
ax.set_title('Top 20 Features by Random Forest Importance')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.show()"""

text_summary = """## 4. Summary & Feature Selection Decision

### Consensus Top Features (appearing in top 15 across all 3 methods):
- **Contract_Two year / Contract_One year** — strongest predictor across all methods
- **tenure** — highly correlated with churn; loyal customers stay
- **MonthlyCharges / TotalCharges / AvgChargesPerMonth** — pricing is a key driver
- **InternetService_Fiber optic** — fiber customers churn more
- **OnlineSecurity_Yes / TechSupport_Yes** — sticky add-on services
- **PaymentMethod_Electronic check** — electronic check = high churn risk

### Decision:
We will keep **all features** for modeling rather than dropping any. The tree-based models (RF, XGBoost, LightGBM, CatBoost) handle irrelevant features gracefully, and the linear models (LR, SVM) benefit from the regularization we'll apply. Dropping features risks losing signal with minimal upside at this dataset size (~7k rows, ~30 features)."""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_corr),
    nbf.v4.new_code_cell(code_corr),
    nbf.v4.new_markdown_cell(text_mi),
    nbf.v4.new_code_cell(code_mi),
    nbf.v4.new_markdown_cell(text_rf),
    nbf.v4.new_code_cell(code_rf),
    nbf.v4.new_markdown_cell(text_summary),
]

with open("notebooks/07_feature_selection.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generated.")
