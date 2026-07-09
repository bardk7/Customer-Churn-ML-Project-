import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# M10: Train/Test Split
We split the feature-engineered data into training (80%) and testing (20%) sets using stratified sampling to preserve the class distribution of the target variable."""

code_setup = """import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
from src.data import load_raw_data
from src.cleaning import clean_data
from src.features import build_features
from src.split import split_data

df = build_features(clean_data(load_raw_data("../data/raw")), scale=False)
X_train, X_test, y_train, y_test = split_data(df)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Test set:     {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"\\nTrain churn rate: {y_train.mean():.4f}")
print(f"Test churn rate:  {y_test.mean():.4f}")
print(f"Overall churn rate: {df['Churn'].mean():.4f}")
"""

text_verify = """## Verification
- Stratification preserved: train and test churn rates match the overall rate.
- No shared indices between train and test (no data leakage).
- Split ratio is 80/20 as specified."""

code_verify = """# Verify no shared indices
overlap = set(X_train.index) & set(X_test.index)
print(f"Index overlap: {len(overlap)} (should be 0)")
assert len(overlap) == 0, "Data leakage detected!"
print("No data leakage. ✓")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_verify),
    nbf.v4.new_code_cell(code_verify),
]

with open("notebooks/08_train_test_split.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generated.")
