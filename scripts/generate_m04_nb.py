import nbformat as nbf

nb = nbf.v4.new_notebook()

text_1 = """# M04: Data Cleaning
This notebook documents the data cleaning steps applied to the Telco Customer Churn dataset.

### Decisions Made:
1. **TotalCharges**: Contains some empty spaces (`" "`) which represent missing values (typically for customers with 0 tenure). These are converted to `NaN` and then imputed with `0`. The column dtype is converted to `float64`.
2. **Duplicates**: Any fully duplicated rows are dropped to prevent bias.
"""

code_1 = """import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd().parent))
from src.data import load_raw_data
from src.cleaning import clean_data

# Load data
df = load_raw_data("../data/raw")
print(f"Raw shape: {df.shape}")

# Clean data
df_clean = clean_data(df)
print(f"Clean shape: {df_clean.shape}")

# Verify no missing values
print(f"Total nulls: {df_clean.isnull().sum().sum()}")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1)
]

with open("notebooks/02_data_cleaning.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generated.")
