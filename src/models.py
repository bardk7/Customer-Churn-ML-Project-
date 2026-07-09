from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def get_baseline_models(random_state=42):
    """
    Returns a dictionary of baseline models, wrapped in a pipeline
    with standard scaling if the model is sensitive to scale.
    """
    models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=random_state, max_iter=1000))
        ]),
        'DecisionTree': Pipeline([
            ('model', DecisionTreeClassifier(random_state=random_state))
        ]),
        'RandomForest': Pipeline([
            ('model', RandomForestClassifier(random_state=random_state))
        ]),
        'GradientBoosting': Pipeline([
            ('model', GradientBoostingClassifier(random_state=random_state))
        ]),
        'XGBoost': Pipeline([
            ('model', xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss'))
        ]),
        'LightGBM': Pipeline([
            ('model', lgb.LGBMClassifier(random_state=random_state))
        ]),
        'CatBoost': Pipeline([
            ('model', CatBoostClassifier(random_state=random_state, verbose=0))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(random_state=random_state, probability=True))
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier())
        ]),
        'NaiveBayes': Pipeline([
            ('scaler', StandardScaler()),
            ('model', GaussianNB())
        ])
    }
    return models
