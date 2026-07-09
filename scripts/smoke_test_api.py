import httpx, json

client = httpx.Client(timeout=60.0)

# 1. Health check
r = client.get("http://127.0.0.1:8000/health")
print("=== GET /health ===")
print(f"Status: {r.status_code}")
print(f"Body:   {r.json()}")
print()

# 2. Valid prediction
payload = {
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
r = client.post("http://127.0.0.1:8000/predict", json=payload)
print("=== POST /predict (valid) ===")
print(f"Status: {r.status_code}")
print(f"Body:   {json.dumps(r.json(), indent=2)}")
print()

# 3. Invalid payload (bad category)
bad = payload.copy()
bad["Contract"] = "Three year"
r = client.post("http://127.0.0.1:8000/predict", json=bad)
print("=== POST /predict (invalid category) ===")
print(f"Status: {r.status_code}")
print()

# 4. Check /docs exists
r = client.get("http://127.0.0.1:8000/docs")
print("=== GET /docs (Swagger UI) ===")
print(f"Status: {r.status_code}")
ct = r.headers.get("content-type", "")
print(f"Content-Type: {ct}")
has_swagger = "swagger" in r.text.lower()
print(f"Contains Swagger: {has_swagger}")
