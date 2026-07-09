import streamlit as st
import httpx
import os
import json

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")
st.title("📊 Customer Churn Predictor")
st.markdown("Enter customer details below to predict churn probability.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multi = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.5)
        total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0, step=1.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": str(total)
    }

    try:
        with httpx.Client(timeout=90.0) as client:
            r = client.post(f"{API_URL}/predict", json=payload)

        if r.status_code == 200:
            data = r.json()
            prob = data["probability"]
            tier = data["risk_tier"]

            st.divider()
            st.subheader("Prediction Result")

            color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[tier]
            label = "Will Churn" if data["prediction"] == 1 else "Will Stay"

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", label)
            c2.metric("Probability", f"{prob:.1%}")
            c3.metric("Risk Tier", f"{color} {tier}")

            st.progress(prob)
        elif r.status_code == 422:
            st.error(f"Invalid input: {r.json()}")
        else:
            st.error(f"API error (status {r.status_code})")
    except httpx.ConnectError:
        st.error(f"Could not connect to API at {API_URL}. Is the server running?")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
