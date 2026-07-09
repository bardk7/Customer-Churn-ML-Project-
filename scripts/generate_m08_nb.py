import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# M08: Feature Engineering
This notebook applies the feature engineering pipeline defined in `src/features.py`.

### Steps:
1. **Encode target**: Convert `Churn` from Yes/No to 1/0.
2. **Drop customerID**: Not a predictive feature.
3. **Engineer new features**:
   - `AvgChargesPerMonth` = TotalCharges / max(tenure, 1) — normalizes spending by tenure.
   - `tenure_group` — buckets tenure into interpretable segments.
4. **One-hot encode** all remaining categorical columns (drop_first to avoid multicollinearity).
5. **Standard scale** numerical columns for models sensitive to feature magnitude (e.g., SVM, KNN, Logistic Regression)."""

code_setup = """import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
from src.data import load_raw_data
from src.cleaning import clean_data
from src.features import build_features, add_engineered_features, encode_target, drop_customer_id

df = clean_data(load_raw_data("../data/raw"))
print(f"Clean data shape: {df.shape}")
"""

text_eng = """## Engineered Features Preview"""

code_eng = """df_eng = add_engineered_features(df)
df_eng[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargesPerMonth', 'tenure_group']].head(10)"""

text_full = """## Full Pipeline Output (unscaled, for inspection)"""

code_full = """df_feat = build_features(df, scale=False)
print(f"Feature-engineered shape: {df_feat.shape}")
print(f"\\nColumns ({len(df_feat.columns)}):")
print(df_feat.columns.tolist())
print(f"\\nNull count: {df_feat.isnull().sum().sum()}")
print(f"\\nObject columns remaining: {df_feat.select_dtypes(include=['object']).columns.tolist()}")
df_feat.head()"""

text_save = """## Save processed data for modeling"""

code_save = """import os
os.makedirs("../data/processed", exist_ok=True)
df_feat.to_csv("../data/processed/telco_features.csv", index=False)
print("Saved to data/processed/telco_features.csv")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_eng),
    nbf.v4.new_code_cell(code_eng),
    nbf.v4.new_markdown_cell(text_full),
    nbf.v4.new_code_cell(code_full),
    nbf.v4.new_markdown_cell(text_save),
    nbf.v4.new_code_cell(code_save),
]

with open("notebooks/06_feature_engineering.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generated.")
