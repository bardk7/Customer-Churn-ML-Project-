# Telecom Customer Churn Prediction & Retention

Predict telecom customer churn using a tuned LightGBM pipeline, deployed as a REST API and a Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)

## Overview
This project tackles the critical business problem of telecommunications customer churn. By analyzing demographic, account, and service usage data, we built an end-to-end Machine Learning pipeline to identify customers at high risk of leaving. The final solution leverages a tuned LightGBM model (handling class imbalance via SMOTE), exposed via a FastAPI backend, and consumed by an interactive Streamlit frontend for retention agents.

## Project Architecture
```text
.
├── api/             # FastAPI backend application and Dockerfile
├── app/             # Streamlit frontend dashboard and Dockerfile
├── data/
│   ├── processed/   # Cleaned and transformed datasets
│   └── raw/         # Original immutable dataset
├── models/          # Serialized ML artifacts (e.g., best_churn_model.pkl)
├── notebooks/       # Jupyter notebooks for EDA, feature engineering, and modeling
├── reports/         # Generated business reports and deployment designs
├── scripts/         # Utility scripts (e.g., automated smoke testing)
├── src/             # Reusable Python modules (data cleaning, prediction wrapper)
└── tests/           # Pytest integration and unit tests
```

## Dataset
- **Source**: Telco Customer Churn dataset (IBM / Kaggle).
- **Size**: 7,043 rows, 21 initial columns.
- **Target Variable**: `Churn` (Yes/No).
- **Availability**: Available in `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Methodology Summary
1. **Exploratory Data Analysis (EDA)**: Analyzed univariate, bivariate, and multivariate trends to identify early churn signals.
2. **Data Cleaning & Feature Engineering**: Handled missing numeric fields (`TotalCharges`), one-hot encoded categorical variables, and scaled numerical features.
3. **Class Imbalance**: Applied SMOTE (Synthetic Minority Over-sampling Technique) exclusively within the training pipeline to prevent data leakage.
4. **Model Selection**: Evaluated 10 baselines. Tree-based ensembles (LightGBM, CatBoost) and Logistic Regression performed best.
5. **Hyperparameter Tuning**: Used `RandomizedSearchCV` to optimize the top candidates. **LightGBM** emerged as the winning model balancing recall and precision.

## Results
The final tuned `LGBMClassifier` was evaluated on an untouched 20% test set, achieving the following metrics:
- **ROC-AUC**: 0.835
- **Recall**: 0.68
- **Precision**: 0.55
- **F1-Score**: 0.61

## Key Business Insights
- **Contract Type**: Customers on Month-to-Month contracts are at a severely higher risk of churning compared to 1-year or 2-year commitments.
- **Tenure**: The first 12 months are the most critical period. Churn probability drops significantly after the first year.
- **Monthly Charges**: High monthly charges, particularly for Fiber Optic internet services, correlate strongly with higher churn risk. Customers lacking tech support or online security are more vulnerable.
- **Recommendation**: Incentivize 1-year contracts and implement a proactive "First-Year Nurturing Program" to stabilize new users.

## Installation & Setup

### 1. Running via Docker Compose (Recommended)
You must have Docker Desktop or a running Docker daemon.
```bash
docker-compose up --build -d
```
- **Streamlit Frontend**: http://localhost:8501
- **FastAPI Backend Docs**: http://localhost:8000/docs

### 2. Running Locally (Virtual Environment)
In Terminal 1 (API):
```bash
python -m venv venv
# Windows: .\venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In Terminal 2 (Frontend):
```bash
# Activate the same venv
streamlit run app/app.py
```

## API Usage
**GET /health**
```bash
curl -X GET "http://127.0.0.1:8000/health"
# Response: {"status": "ok"}
```

**POST /predict**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "gender": "Female",
           "SeniorCitizen": 0,
           "Partner": "Yes",
           "Dependents": "No",
           "tenure": 1,
           "PhoneService": "No",
           "MultipleLines": "No phone service",
           "InternetService": "DSL",
           "OnlineSecurity": "No",
           "OnlineBackup": "Yes",
           "DeviceProtection": "No",
           "TechSupport": "No",
           "StreamingTV": "No",
           "StreamingMovies": "No",
           "Contract": "Month-to-month",
           "PaperlessBilling": "Yes",
           "PaymentMethod": "Electronic check",
           "MonthlyCharges": 29.85,
           "TotalCharges": "29.85"
         }'
```
**Response**:
```json
{
  "probability": 0.825,
  "prediction": 1,
  "risk_tier": "High"
}
```

## Running Tests
To run the automated test suite (including model pipeline and API endpoint tests):
```bash
python -m pytest -q
```

## Development Process
This repository was built using a structured, rigorously tested 27-milestone process (19 ML milestones + 8 deployment milestones). Every step was sequentially committed using Conventional Commits to ensure full reproducibility and a clean history.

## Limitations & Future Work
- **Precision Tradeoff**: The model prioritizes Recall (0.68) to catch as many churners as possible, resulting in lower Precision (0.55). This means 45% of flagged "high risk" customers are false alarms. Retention interventions should focus on low-cost tactics first.
- **Threshold Tuning**: Future work should implement a cost-matrix to dynamically adjust the probability threshold (currently 0.5) to maximize net profit based on actual business retention costs.
- **Time-Series Telemetry**: Adding granular usage data (e.g., dropped calls, sudden bandwidth drops) would likely drastically improve predictive power compared to static demographic snapshots.
