from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

valid_payload = {
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
    "TotalCharges": "358.2"
}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid_payload():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert "prediction" in data
    assert "risk_tier" in data

def test_predict_invalid_payload_missing_field():
    bad_payload = valid_payload.copy()
    del bad_payload["MonthlyCharges"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422 # Unprocessable Entity

def test_predict_invalid_payload_wrong_type():
    bad_payload = valid_payload.copy()
    bad_payload["tenure"] = -5 # Invalid according to ge=0
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    
def test_predict_invalid_payload_bad_category():
    bad_payload = valid_payload.copy()
    bad_payload["Contract"] = "Three year" # Not allowed
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
