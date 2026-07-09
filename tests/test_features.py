import pytest
import pandas as pd
import numpy as np
from src.data import load_raw_data
from src.cleaning import clean_data
from src.features import (
    encode_target, drop_customer_id, encode_categoricals,
    add_engineered_features, build_features
)


@pytest.fixture
def clean_df():
    return clean_data(load_raw_data())


def test_encode_target(clean_df):
    df = encode_target(clean_df)
    assert df['Churn'].dtype in [np.int32, np.int64, int]
    assert set(df['Churn'].unique()) == {0, 1}


def test_drop_customer_id(clean_df):
    df = drop_customer_id(clean_df)
    assert 'customerID' not in df.columns


def test_encode_categoricals(clean_df):
    df = encode_target(clean_df)
    df = drop_customer_id(df)
    df = encode_categoricals(df)
    assert df.select_dtypes(include=['object']).shape[1] == 0, "No object columns should remain"


def test_engineered_features(clean_df):
    df = add_engineered_features(clean_df)
    assert 'AvgChargesPerMonth' in df.columns
    assert 'tenure_group' in df.columns
    # No infinities from division
    assert not np.isinf(df['AvgChargesPerMonth']).any()


def test_build_features_no_leakage(clean_df):
    df = build_features(clean_df, scale=False)
    assert 'customerID' not in df.columns
    assert df.select_dtypes(include=['object']).shape[1] == 0
    assert df.isnull().sum().sum() == 0, "No nulls after full pipeline"
