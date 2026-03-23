import pandas as pd


def data_info(data: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame with dtype, unique counts, and null counts."""
    records = []
    for col in data.columns:
        records.append({
            "Col": col,
            "dtype": data[col].dtype,
            "n_uniques": data[col].nunique(),
            "Unique Values": data[col].unique(),
            "Nulls": data[col].isna().sum(),
        })
    return pd.DataFrame(records)
