import pandas as pd
from pathlib import Path

def load_raw_data(data_dir="data/raw"):
    """
    Loads the raw Telco Customer Churn dataset.
    """
    file_path = Path(data_dir) / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_csv(file_path)
