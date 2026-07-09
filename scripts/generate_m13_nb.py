import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# M13: Hyperparameter Tuning
We will tune the hyperparameters of our top baseline models: Logistic Regression, LightGBM, and CatBoost.
Since we decided to use SMOTE, we must include SMOTE inside a pipeline so it is only applied to the training folds during cross-validation, preventing data leakage."""

code_setup = """import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
from src.data import load_raw_data
from src.cleaning import clean_data
from src.features import build_features
from src.split import split_data

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Load data
df = build_features(clean_data(load_raw_data("../data/raw")), scale=False)
X_train, X_test, y_train, y_test = split_data(df)
"""

text_tune = """## RandomizedSearchCV on Top Models"""

code_tune = """tuned_models = {}

# 1. Logistic Regression
lr_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42, max_iter=1000))
])
lr_param_grid = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l2']
}
lr_search = RandomizedSearchCV(lr_pipeline, lr_param_grid, n_iter=5, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
lr_search.fit(X_train, y_train)
tuned_models['LogisticRegression'] = lr_search.best_estimator_
print(f"Logistic Regression Best AUC: {lr_search.best_score_:.4f}")
print(f"Logistic Regression Best Params: {lr_search.best_params_}\\n")

# 2. LightGBM
lgb_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', lgb.LGBMClassifier(random_state=42))
])
lgb_param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 5, 7, -1],
    'model__num_leaves': [15, 31, 63]
}
lgb_search = RandomizedSearchCV(lgb_pipeline, lgb_param_grid, n_iter=10, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
lgb_search.fit(X_train, y_train)
tuned_models['LightGBM'] = lgb_search.best_estimator_
print(f"LightGBM Best AUC: {lgb_search.best_score_:.4f}")
print(f"LightGBM Best Params: {lgb_search.best_params_}\\n")

# 3. CatBoost
cat_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', CatBoostClassifier(random_state=42, verbose=0))
])
cat_param_grid = {
    'model__iterations': [100, 200, 300],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__depth': [4, 6, 8]
}
cat_search = RandomizedSearchCV(cat_pipeline, cat_param_grid, n_iter=10, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
cat_search.fit(X_train, y_train)
tuned_models['CatBoost'] = cat_search.best_estimator_
print(f"CatBoost Best AUC: {cat_search.best_score_:.4f}")
print(f"CatBoost Best Params: {cat_search.best_params_}\\n")
"""

text_save = """## Save the Best Model
CatBoost typically performs very well, let's pick the best model from our tuning (based on CV AUC) and save it for future evaluation."""

code_save = """import joblib
import os

best_model_name = max(
    [
        ('LogisticRegression', lr_search.best_score_), 
        ('LightGBM', lgb_search.best_score_), 
        ('CatBoost', cat_search.best_score_)
    ], 
    key=lambda x: x[1]
)[0]

print(f"Overall Best Model: {best_model_name}")

os.makedirs("../models", exist_ok=True)
joblib.dump(tuned_models[best_model_name], "../models/best_churn_model.pkl")
print(f"Saved {best_model_name} pipeline to ../models/best_churn_model.pkl")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_tune),
    nbf.v4.new_code_cell(code_tune),
    nbf.v4.new_markdown_cell(text_save),
    nbf.v4.new_code_cell(code_save),
]

with open("notebooks/11_hyperparameter_tuning.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generated.")
