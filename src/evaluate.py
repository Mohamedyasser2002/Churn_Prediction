from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def print_classification_report(model, X_test, y_test, model_name: str = "Model"):
    """Print sklearn classification report for a fitted model."""
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Classification Report (Test Data)")
    print(classification_report(y_test, y_pred))


def plot_model_comparison(results_df: pd.DataFrame, save_path: str | None = None):
    """Bar chart comparing test accuracy across models."""
    colors = ["#59A14F", "#E15759", "#4E79A7", "#F28E2B", "#76B7B2", "#EDC948"]
    ax = results_df.plot(
        x="Model", y="Test Accuracy", kind="bar", legend=False, color=colors,
        figsize=(10, 6)
    )
    ax.set_title("Model Evaluation Results (Test Set)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Test Accuracy")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_feature_importance(importance: pd.Series, save_path: str | None = None):
    """Horizontal bar chart of top feature importances."""
    importance.sort_values(ascending=True).plot(
        kind="barh", title="Top Feature Importance", figsize=(8, 6)
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_gender_churn(df: pd.DataFrame, save_path: str | None = None):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ["#59A14F", "#E15759"]
    sns.countplot(data=df, x="gender", hue="churn", palette=palette, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=10)
    ax.set_title("Gender vs Churn", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_age_distribution(df: pd.DataFrame, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df, x="age", hue="churn", bins=30, kde=True,
        palette=["#59A14F", "#E15759"], alpha=0.6, ax=ax
    )
    ax.set_title("Age Distribution by Churn", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str | None = None):
    plt.figure(figsize=(12, 12))
    sns.heatmap(
        df.corr(numeric_only=True), annot=True,
        cmap="YlGnBu", fmt=".1f", cbar=True
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
