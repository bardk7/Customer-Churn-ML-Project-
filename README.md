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
The project follows a rigorous 19-milestone ML phase process and an 8-milestone deployment phase, strictly adhering to conventional commits and rigorous testing.

## Deployment

The final trained LightGBM model is deployed alongside a FastAPI backend and a Streamlit frontend.

### 1. Running via Docker Compose (Recommended)
This requires Docker Desktop or a running Docker daemon.
```bash
docker-compose up --build -d
```
- **Streamlit Frontend**: `http://localhost:8501`
- **FastAPI Backend**: `http://localhost:8000/docs`

### 2. Running Locally (Virtual Environment)
In two separate terminals:
**Terminal 1 (API)**:
```bash
.\venv\Scripts\activate
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
**Terminal 2 (Frontend)**:
```bash
.\venv\Scripts\activate
streamlit run app/app.py
```

### API Specification
**POST /predict**
Payload (JSON):
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
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
  "TotalCharges": "358.20"
}
```
Response (JSON):
```json
{
  "probability": 0.825,
  "prediction": 1,
  "risk_tier": "High"
}
```

### Known Limitations
- The model expects the specific 19 features outlined above.
- TotalCharges must be passable as a float/numeric string, otherwise it defaults to 0.0.
- Categorical variables strictly enforce exact spelling (e.g. `Fiber optic`, not `Fiber Optic`).