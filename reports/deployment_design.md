# Deployment Design Document (M17)

## 1. Objective
To deploy the tuned LightGBM Customer Churn prediction model (`best_churn_model.pkl`) into a production environment where it can score active customers and surface high-risk individuals to the retention team.

## 2. Deployment Architecture Options

### Option A: Real-Time API (FastAPI)
- **Concept**: Wrap the model in a FastAPI application. The CRM or billing system sends a REST API request with customer features, and the API returns a churn probability score.
- **Pros**: Instant predictions, easy integration with real-time web applications.
- **Cons**: Overkill if the retention team only works from a daily/weekly list. Requires managing server uptime and scaling.

### Option B: Batch Scoring Job (Airflow + Python Script)
- **Concept**: A scheduled script (e.g., weekly) that pulls active customer data from the data warehouse, runs the model pipeline over all records, and writes the scores back to a database table or CRM dashboard.
- **Pros**: Cost-effective, handles large volumes easily, perfectly matches the workflow of a retention team that makes outbound calls based on a list.
- **Cons**: Predictions are only as fresh as the last batch run (e.g., 24 hours old).

### Option C: Interactive Dashboard (Streamlit)
- **Concept**: A Streamlit web application where retention managers can upload a CSV of customers or search for a specific customer ID to see their churn probability and the top SHAP features driving that score.
- **Pros**: Highly interpretable, empowers non-technical teams with self-serve analytics.
- **Cons**: Manual process, does not automatically flag customers in the background.

## 3. Chosen Approach: Hybrid Batch Scoring + Streamlit Dashboard
Given the business context, the most effective approach is a **Hybrid Strategy**:
1. **Weekly Batch Scoring**: Run a Python script via a scheduler (cron/Airflow) every Monday morning to score all active customers. The top 5% highest-risk customers (Prob > 0.8) are automatically routed to the retention team's queue.
2. **Interpretability Dashboard**: Deploy an internal Streamlit app where retention agents can view the specific SHAP drivers for a flagged customer *before* making the retention call. Knowing *why* a customer is at risk (e.g., high charges on fiber optic) allows the agent to tailor the offer (e.g., offering a bundled discount).

## 4. Technical Requirements
- **Environment**: Docker container running Python 3.12.
- **Dependencies**: `lightgbm`, `scikit-learn`, `pandas`, `shap`, `streamlit`.
- **Model Storage**: Load `best_churn_model.pkl` from an S3 bucket or local volume.
- **Data Source**: Secure read access to the customer data warehouse; write access to the `Customer_Risk_Scores` table.
