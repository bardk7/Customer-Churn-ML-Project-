import pytest
import numpy as np
from src.data import load_raw_data
from src.cleaning import clean_data
from src.features import build_features
from src.split import split_data


@pytest.fixture
def feature_df():
    return build_features(clean_data(load_raw_data()), scale=False)


def test_split_sizes(feature_df):
    X_train, X_test, y_train, y_test = split_data(feature_df)
    total = len(feature_df)
    assert len(X_train) + len(X_test) == total
    assert abs(len(X_test) / total - 0.2) < 0.02  # ~20% test


def test_stratification(feature_df):
    X_train, X_test, y_train, y_test = split_data(feature_df)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    # Stratified split should preserve class ratio within ~2%
    assert abs(train_ratio - test_ratio) < 0.02


def test_no_data_leakage(feature_df):
    X_train, X_test, y_train, y_test = split_data(feature_df)
    # No shared indices
    assert len(set(X_train.index) & set(X_test.index)) == 0
