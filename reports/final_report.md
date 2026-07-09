# Final Report & Project Conclusion (M18 & M19)

## 1. Methodology
This project implemented a complete end-to-end Machine Learning pipeline to predict telecom customer churn:
1. **Data Ingestion & Cleaning**: Processed 7,043 raw customer records. Handled missing values in `TotalCharges` and converted necessary data types.
2. **Exploratory Data Analysis (EDA)**: Conducted univariate, bivariate, and multivariate analysis to identify early churn signals (e.g., tenure, contract type).
3. **Feature Engineering & Selection**: Encoded categorical variables using One-Hot Encoding and scaled numerical features. Evaluated features using Mutual Information and Random Forest Importance. We retained all standard features as tree-based models handle them efficiently without the need for aggressive pruning.
4. **Data Splitting & Balancing**: Split the data into 80% training and 20% testing sets using stratified sampling to preserve the 73/27 class balance. We applied **SMOTE (Synthetic Minority Over-sampling Technique)** exclusively to the training folds inside a `Pipeline` to combat class imbalance without causing data leakage.

## 2. Model Comparisons
We established 10 baseline models. The primary metric for model comparison was **ROC-AUC** to measure the ability of the model to distinguish between churners and non-churners across thresholds. 

During baseline evaluation, the top performers were:
- **CatBoost**: Highest resilience to overfitting, handled categorical relationships well.
- **LightGBM**: Fast and highly performant.
- **Logistic Regression**: A surprisingly strong and interpretable baseline that rivaled complex trees.

We performed Hyperparameter Tuning using `RandomizedSearchCV` on the top three models. The **Tuned CatBoost** model emerged as the superior estimator.

## 3. Final Conclusion & Performance Metrics
The final selected model is the **Tuned CatBoostClassifier**. Evaluated on the untouched 20% test set, it achieved:
- **ROC-AUC**: 0.835
- **Recall**: 0.68
- **Precision**: 0.55
- **F1-Score**: 0.61

### The Precision Tradeoff & Business Implication
The model demonstrates a strong **Recall (0.68)**, successfully identifying 68% of the customers who actually churn. However, this comes at the cost of a lower **Precision (0.55)**. 

**Business Cost Implication**: A precision of 0.55 means that out of every 100 customers the model flags as "high risk", approximately 45 of them were *not* actually going to churn. If the business offers an aggressive financial discount to retain these flagged customers, we will be spending retention dollars on 45 customers who would have stayed anyway. 

*Recommendation*: To mitigate this cost, retention campaigns triggered by this model should focus on low-cost/high-value interventions first (e.g., personalized check-in calls, free tech-support trials) rather than immediate steep price discounts.

## 4. Limitations
- **Lack of Usage Data**: The dataset lacks granular, time-series usage data (e.g., dropped calls, daily data consumption spikes). Real-world churn is often preceded by sudden changes in service usage behavior.
- **Static Snapshot**: This dataset is a single snapshot in time. A true production system would benefit from a survival analysis framework or a rolling-window approach to predict churn within a specific future timeframe (e.g., "churn within the next 30 days").

## 5. Future Improvements
- **Time-Series Features**: Incorporate billing history trends (e.g., "increase in bill over last 3 months") and usage telemetry to capture the *velocity* of a customer's dissatisfaction.
- **Threshold Tuning via Cost Matrix**: Instead of using the default 0.5 probability threshold, we should calculate the exact dollar cost of a False Positive (wasted discount) vs a False Negative (lost customer LTV). We can then algorithmically select the probability threshold that maximizes net profit.
- **A/B Testing**: Deploy the model in a shadow mode or run a randomized control trial where only 50% of high-risk customers receive an intervention, allowing us to definitively measure the causal ROI of the model.
