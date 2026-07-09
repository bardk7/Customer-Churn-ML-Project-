# Assessing Customer Churn Using Machine Learning

An end-to-end, production-quality machine learning project for predicting customer churn in a telecommunications context.

## Project Overview
This project takes the public Telco Customer Churn dataset and applies a full machine learning lifecycle, from exploratory data analysis and feature engineering to model training, evaluation, interpretation, and business recommendations.

## Tech Stack
- **Python 3.11+**
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, xgboost, lightgbm, catboost, imbalanced-learn
- **Interpretability**: shap

## Repository Structure
- `/data/raw` - The original, immutable dataset.
- `/data/processed` - Cleaned and transformed datasets.
- `/notebooks` - Jupyter notebooks containing EDA and modeling steps.
- `/src` - Reusable Python modules for data cleaning, feature engineering, modeling, and evaluation.
- `/tests` - Unit tests for the `/src` modules.
- `/reports` - Generated analysis, figures, and the final business report.

## Setup Instructions

1. **Clone the repository**
2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: .\venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run Notebooks**
   ```bash
   jupyter lab
   ```

## Workflow
The project follows a rigorous 19-milestone process, strictly adhering to conventional commits and rigorous testing.