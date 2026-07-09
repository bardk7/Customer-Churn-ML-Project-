import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the Churn target column to binary 0/1."""
    df = df.copy()
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    return df


def drop_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Drop customerID — not a predictive feature."""
    return df.drop(columns=['customerID'], errors='ignore')


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all remaining object-type columns.
    Uses drop_first=True to avoid multicollinearity.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.
    
    Rationale:
    - AvgChargesPerMonth: normalizes TotalCharges by tenure, captures
      per-month spending independent of how long a customer has been around.
    - tenure_group: buckets tenure into interpretable segments for downstream
      analysis and potential interaction features.
    - NumServices: count of add-on services a customer has, a proxy for
      how "embedded" they are in the platform.
    """
    df = df.copy()

    # Average monthly spend (handle tenure=0 by capping at 1)
    df['AvgChargesPerMonth'] = df['TotalCharges'] / df['tenure'].clip(lower=1)

    # Tenure buckets
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 72],
        labels=['0-12', '13-24', '25-48', '49-60', '61-72'],
        right=True, include_lowest=True
    )

    return df


def scale_numerical(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """
    StandardScaler on numerical columns.
    If cols is None, scales tenure, MonthlyCharges, TotalCharges, AvgChargesPerMonth.
    """
    df = df.copy()
    if cols is None:
        cols = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargesPerMonth']
                if c in df.columns]
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def build_features(df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline: encode target, drop ID, engineer
    features, encode categoricals, optionally scale numericals.
    """
    df = encode_target(df)
    df = drop_customer_id(df)
    df = add_engineered_features(df)
    df = encode_categoricals(df)
    if scale:
        df = scale_numerical(df)
    return df
