import pytest
from src.predict import preprocess_input, predict, EXPECTED_FEATURES

def test_preprocess_input():
    raw_input = {
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
    }
    
    df = preprocess_input(raw_input)
    assert df.shape == (1, len(EXPECTED_FEATURES))
    assert list(df.columns) == EXPECTED_FEATURES
    assert df['tenure'].iloc[0] == 1
    assert df['AvgChargesPerMonth'].iloc[0] == 29.85
    # Check dummy that should be 1
    assert df['Partner_Yes'].iloc[0] == 1
    # Check dummy that should be 0 (Female doesn't create gender_Male=1)
    assert df['gender_Male'].iloc[0] == 0

def test_predict_returns_correct_format():
    raw_input = {
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
    }
    
    result = predict(raw_input)
    assert "probability" in result
    assert "prediction" in result
    assert "risk_tier" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in [0, 1]
    assert result["risk_tier"] in ["Low", "Medium", "High"]
