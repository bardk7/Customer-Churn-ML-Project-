import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw Telco Customer Churn dataset.
    - Handles missing/empty values in TotalCharges
    - Converts dtypes
    - Removes duplicates
    """
    df = df.copy()
    
    # 1. Handle empty spaces in 'TotalCharges'
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Impute with 0 since these are mostly new customers with tenure = 0
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
    # 2. Remove duplicates (if any)
    df = df.drop_duplicates()
    
    return df
