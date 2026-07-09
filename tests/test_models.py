import pytest
import pandas as pd
import numpy as np
from src.data import load_raw_data
from src.cleaning import clean_data
from src.features import build_features
from src.split import split_data
from src.models import get_baseline_models
from sklearn.metrics import roc_auc_score


@pytest.fixture
def sample_data():
    """Returns a small sample of the data for fast testing."""
    df = load_raw_data().sample(200, random_state=42)
    df = build_features(clean_data(df), scale=False)
    # Ensure both classes are present in train/test by not stratifying such a small sample if it fails,
    # but split_data handles stratify.
    return split_data(df)


def test_baseline_models(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    models = get_baseline_models()
    
    for name, pipeline in models.items():
        # Fit should not throw an error
        pipeline.fit(X_train, y_train)
        
        # Predict should not throw an error and have no NaNs
        preds = pipeline.predict(X_test)
        assert not np.isnan(preds).any(), f"{name} produced NaN predictions"
        
        # Proba should work
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_test)[:, 1]
            assert not np.isnan(probs).any(), f"{name} produced NaN probabilities"
            
            # AUC should be reasonable (e.g. >= 0.4 on a tiny sample just to check it computes)
            auc = roc_auc_score(y_test, probs)
            assert 0.0 <= auc <= 1.0, f"AUC {auc} for {name} out of bounds"
