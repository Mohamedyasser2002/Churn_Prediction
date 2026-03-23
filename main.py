import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.data_loader import load_data
from src.preprocessing import (
    fill_missing, encode_features, select_top_features,
    apply_smote, scale_features,
)
from src.train import build_models, train_and_evaluate
from src.evaluate import (
    print_classification_report,
    plot_model_comparison,
    plot_feature_importance,
    plot_gender_churn,
    plot_age_distribution,
    plot_correlation_heatmap,
)
from src.utils import data_info

DATA_PATH   = "data/churn_prediction.csv"
PLOTS_DIR   = "outputs/plots"
MODELS_DIR  = "outputs/models"
REPORT_PATH = "outputs/pipeline_report.txt"
TOP_K       = 11


# ── Helper: prints to screen AND writes to file ───────────────────────────────
class Logger:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal = sys.stdout
        self.log      = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Start logging ─────────────────────────────────────────────────────────
    logger = Logger(REPORT_PATH)
    sys.stdout = logger

    # ── 1. Load ───────────────────────────────────────────────────────────────
    section("1. LOAD DATA")
    df = load_data(DATA_PATH)
    print(f"Shape          : {df.shape}")
    print(f"Columns        : {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head().to_string()}")
    print(f"\nData Types:\n{df.dtypes.to_string()}")
    print(f"\nMissing Values:\n{df.isnull().sum().to_string()}")

    # ── 2. EDA plots ──────────────────────────────────────────────────────────
    section("2. EDA PLOTS")
    plot_gender_churn(df,     save_path=f"{PLOTS_DIR}/gender_vs_churn.png")
    plot_age_distribution(df, save_path=f"{PLOTS_DIR}/age_distribution.png")
    print("Saved: gender_vs_churn.png")
    print("Saved: age_distribution.png")

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    section("3. PREPROCESSING — Fill Missing Values")
    df = fill_missing(df)
    print(f"Missing after fill:\n{df.isnull().sum().to_string()}")

    section("3. PREPROCESSING — Encode Features")
    df = encode_features(df)
    print(f"Shape after encoding : {df.shape}")
    print(f"\nEncoded Data Info:\n{data_info(df).to_string()}")
    print(f"\nStatistical Summary:\n{df.describe().to_string()}")

    plot_correlation_heatmap(df, save_path=f"{PLOTS_DIR}/correlation_heatmap.png")
    print("\nSaved: correlation_heatmap.png")

    # ── 4. Train / Test Split ─────────────────────────────────────────────────
    section("4. TRAIN / TEST SPLIT")
    X = df.drop("churn", axis=1)
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Total samples    : {len(df)}")
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Testing  samples : {X_test.shape[0]}")
    print(f"Features         : {X_train.shape[1]}")
    print(f"\nClass distribution in Train:\n{y_train.value_counts().to_string()}")
    print(f"\nClass distribution in Test:\n{y_test.value_counts().to_string()}")

    # ── 5. Feature Selection ──────────────────────────────────────────────────
    section("5. FEATURE SELECTION (Top K Features)")
    top_features = select_top_features(X_train, y_train, top_k=TOP_K)
    print(f"Selected {TOP_K} features:")
    for i, f in enumerate(top_features, 1):
        print(f"  {i:2}. {f}")

    X_train_sel = X_train[top_features]
    X_test_sel  = X_test[top_features]

    # ── 6. SMOTE + Scaling ────────────────────────────────────────────────────
    section("6. SMOTE BALANCING + FEATURE SCALING")
    X_train_bal, y_train_bal = apply_smote(X_train_sel, y_train)
    print(f"Before SMOTE — Class counts:\n{pd.Series(y_train).value_counts().to_string()}")
    print(f"\nAfter  SMOTE — Class counts:\n{pd.Series(y_train_bal).value_counts().to_string()}")
    print(f"\nTraining shape after SMOTE : {X_train_bal.shape}")

    X_train_scaled, X_test_scaled, _ = scale_features(X_train_bal, X_test_sel)
    print(f"Scaling applied — Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

    # ── 7. Train & Evaluate All Models ───────────────────────────────────────
    section("7. MODEL TRAINING & EVALUATION")
    models = build_models()
    print(f"Models to train: {list(models.keys())}\n")

    results_df = train_and_evaluate(
        models, X_train_bal, y_train_bal,
        X_test_sel, y_test,
        X_train_scaled, X_test_scaled,
    )

    print("\n===== Model Performance Summary =====")
    print(results_df.to_string())

    plot_model_comparison(results_df, save_path=f"{PLOTS_DIR}/model_comparison.png")
    print("\nSaved: model_comparison.png")

    # ── 8. Classification Reports for ALL Models ──────────────────────────────
    section("8. CLASSIFICATION REPORTS")
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        # decide which X_test to use (scaled models vs tree-based)
        if model_name in ["LogisticRegression", "SVM", "KNN"]:
            X_eval = X_test_scaled
        else:
            X_eval = X_test_sel
        y_pred = model.predict(X_eval)
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # ── 9. Save All Trained Models ────────────────────────────────────────────
    section("9. SAVING TRAINED MODELS")
    for model_name, model in models.items():
        save_path = f"{MODELS_DIR}/{model_name}.pkl"
        joblib.dump(model, save_path)
        print(f"Saved: {save_path}")

    # ── Done ──────────────────────────────────────────────────────────────────
    section("PIPELINE COMPLETE")
    print(f"All plots   → {PLOTS_DIR}/")
    print(f"All models  → {MODELS_DIR}/")
    print(f"Full report → {REPORT_PATH}")

    logger.close()
    sys.stdout = logger.terminal


if __name__ == "__main__":
    main()