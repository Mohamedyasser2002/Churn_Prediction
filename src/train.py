from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

SCALED_MODELS = {"LogisticRegression", "KNN", "SVM"}


def build_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=300),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(eval_metric="logloss"),
    }


def train_and_evaluate(
    models: dict,
    X_train_bal,
    y_train_bal,
    X_test_sel,
    y_test,
    X_train_scaled,
    X_test_scaled,
) -> pd.DataFrame:
    """Train all models and return a summary DataFrame."""
    results = []
    for name, model in models.items():
        if name in SCALED_MODELS:
            model.fit(X_train_scaled, y_train_bal)
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train_bal, y_train_bal)
            train_pred = model.predict(X_train_bal)
            test_pred = model.predict(X_test_sel)

        train_acc = accuracy_score(y_train_bal, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        results.append([name, train_acc, test_acc])

    results_df = pd.DataFrame(results, columns=["Model", "Train Accuracy", "Test Accuracy"])
    return results_df.sort_values(by="Test Accuracy", ascending=False)
