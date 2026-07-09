from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
import traceback
import sys
from pathlib import Path

# Add src to path so we can import predict
sys.path.append(str(Path(__file__).parent.parent))
from src.predict import predict

app = FastAPI(title="Customer Churn Prediction API")

class CustomerData(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0.0)
    TotalCharges: str | float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        # Convert pydantic model to dict
        input_dict = customer.model_dump()
        result = predict(input_dict)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error during prediction.")
