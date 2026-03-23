import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw churn CSV and drop identifier columns."""
    df = pd.read_csv(filepath)
    df.drop(["customer_id", "branch_code", "city"], inplace=True, axis=1)
    return df
