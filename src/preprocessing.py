import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric nulls with column mean, categorical nulls with mode."""
    df = df.copy()
    df.fillna(df.mean(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode gender as binary int and label-encode occupation."""
    df = df.copy()
    df["gender"] = df["gender"].replace({"Female": 0, "Male": 1}).astype("int64")
    le = LabelEncoder()
    df["occupation"] = le.fit_transform(df["occupation"])
    return df


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 11,
) -> list[str]:
    """Return top-k feature names ranked by Random Forest importance."""
    fs_model = RandomForestClassifier(n_estimators=300, random_state=42)
    fs_model.fit(X_train, y_train)
    importance = pd.Series(
        fs_model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    return list(importance.head(top_k).index)


def apply_smote(X_train, y_train, random_state: int = 42):
    """Balance training set with SMOTE oversampling."""
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)


def scale_features(X_train, X_test):
    """Fit StandardScaler on training data and transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
