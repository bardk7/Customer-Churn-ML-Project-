import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Extract expected features exactly from the trained model
EXPECTED_FEATURES = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargesPerMonth', 
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 
    'MultipleLines_No_phone_service', 'MultipleLines_Yes', 'InternetService_Fiber_optic', 
    'InternetService_No', 'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes', 
    'OnlineBackup_No_internet_service', 'OnlineBackup_Yes', 'DeviceProtection_No_internet_service', 
    'DeviceProtection_Yes', 'TechSupport_No_internet_service', 'TechSupport_Yes', 
    'StreamingTV_No_internet_service', 'StreamingTV_Yes', 'StreamingMovies_No_internet_service', 
    'StreamingMovies_Yes', 'Contract_One_year', 'Contract_Two_year', 'PaperlessBilling_Yes', 
    'PaymentMethod_Credit_card_(automatic)', 'PaymentMethod_Electronic_check', 
    'PaymentMethod_Mailed_check', 'tenure_group_13-24', 'tenure_group_25-48', 
    'tenure_group_49-60', 'tenure_group_61-72'
]

_MODEL = None

def load_model(model_path="models/best_churn_model.pkl"):
    """Loads the model pipeline."""
    global _MODEL
    if _MODEL is None:
        p = Path(model_path)
        # Handle case where predict is called from a nested dir like api/
        if not p.exists():
            p = Path("..") / model_path
        if not p.exists():
            p = Path(__file__).parent.parent / model_path
            
        _MODEL = joblib.load(p)
    return _MODEL

def preprocess_input(input_dict: dict) -> pd.DataFrame:
    """
    Transforms a raw input dictionary into the exact pandas DataFrame 
    expected by the model.
    """
    df = pd.DataFrame([input_dict])
    
    # Drop target or ID if accidentally passed
    df = df.drop(columns=['customerID', 'Churn'], errors='ignore')
    
    # Numeric conversions
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    
    # 1. Engineered features
    df['AvgChargesPerMonth'] = df['TotalCharges'] / df['tenure'].clip(lower=1)
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 72],
        labels=['0-12', '13-24', '25-48', '49-60', '61-72'],
        right=True, include_lowest=True
    )
    
    # 2. Get dummies (do not drop_first because single row drops the only value)
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
    
    # 3. Align to expected features (fill missing with 0)
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0
            
    # Keep only expected features in exact order
    return df[EXPECTED_FEATURES]

def predict(input_dict: dict) -> dict:
    """
    Takes a raw customer dictionary, preprocesses it, and returns the prediction.
    """
    model = load_model()
    X = preprocess_input(input_dict)
    
    # Pipeline handles prediction. Since pipeline is ImbPipeline(SMOTE -> LightGBM), 
    # predict_proba works directly on X.
    prob = float(model.predict_proba(X)[0][1])
    pred = int(prob > 0.5)
    
    # Determine risk tier
    if prob >= 0.8:
        tier = "High"
    elif prob >= 0.5:
        tier = "Medium"
    else:
        tier = "Low"
        
    return {
        "probability": round(prob, 4),
        "prediction": pred,
        "risk_tier": tier
    }
