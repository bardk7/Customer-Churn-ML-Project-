import pytest
import pandas as pd
import numpy as np
from src.data import load_raw_data
from src.cleaning import clean_data

def test_cleaning_total_charges():
    df = load_raw_data()
    
    # Assert that there are some empty spaces initially
    assert (df['TotalCharges'] == ' ').sum() > 0, "Expected empty strings in TotalCharges"
    
    cleaned_df = clean_data(df)
    
    # Assert type changed to float
    assert np.issubdtype(cleaned_df['TotalCharges'].dtype, np.number), "TotalCharges should be numeric"
    
    # Assert no nulls exist
    assert cleaned_df.isnull().sum().sum() == 0, "No null values should be present post-cleaning"

def test_no_duplicates():
    # Create dummy data with duplicates
    df = pd.DataFrame({
        'customerID': ['A', 'A', 'B'],
        'tenure': [1, 1, 2],
        'TotalCharges': ['10', '10', '20']
    })
    
    cleaned_df = clean_data(df)
    assert len(cleaned_df) == 2, "Duplicates should be removed"
