# Executive Summary: Customer Churn Prediction

## 1. Objective
The goal of this project was to analyze telecom customer data to predict customer churn (when a customer leaves the service) and extract actionable insights to improve retention.

## 2. Approach
- **Data Preparation**: Cleaned and transformed demographic, service, and account data (7,043 customers).
- **Modeling**: Evaluated 10 baseline machine learning models. We handled class imbalance using SMOTE. The top performing models were tree-based ensembles (CatBoost, LightGBM) and Logistic Regression.
- **Tuning**: Hyperparameter tuning via cross-validation identified CatBoost as the best overall model, balancing high recall (catching churners) with strong precision (minimizing false alarms).
- **Interpretation**: Used SHAP (SHapley Additive exPlanations) to understand what drives the model's decisions at a granular level.

## 3. Key Drivers of Churn
Based on feature importance and SHAP analysis, the most significant drivers of customer churn are:
1. **Contract Type**: Customers on Month-to-Month contracts are at a severely higher risk of churning compared to those on 1-year or 2-year contracts.
2. **Tenure**: The first year is the most critical period. Churn probability drops significantly after the first 12 months.
3. **Monthly Charges & Service Types**: High monthly charges, particularly for Fiber Optic internet services, correlate strongly with higher churn risk. Customers without tech support or online security add-ons are also more vulnerable.

## 4. Business Recommendations
- **Incentivize Long-Term Commitments**: Focus marketing spend on converting month-to-month customers to 1-year contracts via targeted discounts or loyalty perks.
- **First-Year Nurturing Program**: Implement a proactive check-in sequence during a customer's first 12 months (e.g., at months 3, 6, and 9) to ensure satisfaction and resolve any onboarding issues.
- **Value-Add Bundling**: Offer bundled services (like Tech Support and Online Security) at a reduced rate for Fiber Optic customers to increase switching costs and perceived value.

## 5. Next Steps
- Deploy the predictive model to score the active customer base weekly.
- Route high-risk customers (Prob > 0.8) to a specialized retention team for immediate outreach.
- Track the conversion rate of targeted retention offers against a control group to measure the ROI of the predictive model.
