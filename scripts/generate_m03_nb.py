import nbformat as nbf

nb = nbf.v4.new_notebook()

text_1 = """# M03: Dataset Overview
This notebook loads the Telco Customer Churn dataset and explores its shape, data types, features, and summary statistics."""

code_1 = """import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd().parent))
from src.data import load_raw_data

# Load data
df = load_raw_data("../data/raw")
df.head()"""

text_2 = """## Shape and Data Types"""

code_2 = """print(f"Dataset Shape: {df.shape}")
print("\\nData Types:")
print(df.dtypes)"""

text_3 = """## Target Variable Summary"""

code_3 = """print(df['Churn'].value_counts(normalize=True))"""

text_4 = """## Summary Statistics"""

code_4 = """df.describe(include='all').T"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(text_4),
    nbf.v4.new_code_cell(code_4)
]

with open("notebooks/01_data_understanding.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generated.")
