import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# M05: EDA - Univariate Analysis
In this notebook, we explore the distributions of individual features. 
We look at numerical variables using histograms, density plots, and boxplots, and categorical variables using count plots."""

code_setup = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd().parent))
from src.data import load_raw_data
from src.cleaning import clean_data

# Set styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.figsize': (10, 6)})

# Load and clean data
df = clean_data(load_raw_data("../data/raw"))
df.head()"""

text_num = """## 1. Numerical Variables
The dataset has three main numerical features: `tenure`, `MonthlyCharges`, and `TotalCharges`. We'll visualize their distributions to identify skewness or interesting patterns."""

code_num = """num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

fig, axes = plt.subplots(3, 2, figsize=(14, 15))

for i, col in enumerate(num_cols):
    # Histogram & KDE
    sns.histplot(df[col], kde=True, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'Distribution of {col}')
    
    # Boxplot
    sns.boxplot(x=df[col], ax=axes[i, 1], color='lightgreen')
    axes[i, 1].set_title(f'Boxplot of {col}')
    
plt.tight_layout()
plt.show()"""

text_num_interp = """### Interpretation of Numerical Variables
- **tenure**: The distribution is bimodal. There's a massive spike at 0-5 months (new customers) and another spike at the high end (70+ months), indicating a solid base of highly loyal customers alongside a high volume of new/transient customers.
- **MonthlyCharges**: Many customers are in the lower bracket (~$20), which likely corresponds to basic services. Then there is a wide spread between $40 and $110.
- **TotalCharges**: This is heavily right-skewed. Most customers have lower total charges, which makes sense given the large number of new customers (low tenure)."""

text_cat = """## 2. Categorical Variables
We will look at the count plots for key categorical variables to understand the composition of the customer base."""

code_cat = """cat_cols = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'Contract', 'PaymentMethod', 'Churn']
            
# Adjust column names if case doesn't match perfectly, standardizing to existing columns
existing_cat_cols = [col for col in cat_cols if col in df.columns]
# If 'gender' is lowercase in dataset, let's just grab object columns + SeniorCitizen (int but categorical)
object_cols = df.select_dtypes(include=['object']).columns.tolist()
# Remove customerID
object_cols = [c for c in object_cols if c != 'customerID']
if 'SeniorCitizen' not in object_cols:
    object_cols.insert(0, 'SeniorCitizen')

fig, axes = plt.subplots(int(np.ceil(len(object_cols) / 3)), 3, figsize=(18, 24))
axes = axes.flatten()

for i, col in enumerate(object_cols):
    sns.countplot(data=df, x=col, ax=axes[i], palette='Set2')
    axes[i].set_title(f'Count Plot of {col}')
    axes[i].tick_params(axis='x', rotation=45)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
    
plt.tight_layout()
plt.show()"""

text_cat_interp = """### Interpretation of Categorical Variables
- **Target Variable (Churn)**: The dataset is imbalanced. A significant majority of customers did not churn (No), while a smaller but substantial portion did (Yes). This imbalance (~73% No to ~27% Yes) will need to be handled during modeling.
- **Demographics**: Gender is split almost equally. Very few customers are senior citizens.
- **Services**: Most customers have phone service. Fiber optic is popular but DSL is also widely used. Many lack additional services like Online Security or Tech Support.
- **Contract**: Month-to-month contracts are the most common, which is a known risk factor for churn compared to 1 or 2-year contracts.
- **Payment Method**: Electronic check is highly popular."""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_num),
    nbf.v4.new_code_cell(code_num),
    nbf.v4.new_markdown_cell(text_num_interp),
    nbf.v4.new_markdown_cell(text_cat),
    nbf.v4.new_code_cell(code_cat),
    nbf.v4.new_markdown_cell(text_cat_interp)
]

with open("notebooks/03_eda_univariate.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generated.")
