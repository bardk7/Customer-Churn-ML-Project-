import pytest
from src.data import load_raw_data

def test_data_shape_and_dtypes():
    df = load_raw_data()
    # Check expected shape: ~7043 rows, 21 columns
    assert df.shape == (7043, 21), f"Unexpected shape {df.shape}"
    
    # Check some essential columns exist
    expected_cols = ['customerID', 'tenure', 'TotalCharges', 'Churn']
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"
        
    # Check target is present
    assert df['Churn'].nunique() == 2, "Churn column should have 2 unique values"
