"""Evaluate a trained model's performance on held-out data."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute standard classification metrics on a test set.

    Returns a dict of accuracy, precision, recall, and F1, plus the
    full sklearn classification report as text.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "report": classification_report(y_test, y_pred, zero_division=0),
    }
    return metrics


def plot_confusion_matrix(model, X_test, y_test, labels=None, save_path: str | None = None) -> None:
    """Plot (and optionally save) a confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # Example: assumes you already have a trained model, X_test, y_test
    # from src/models/train.py in scope (e.g. loaded via pickle).
    print("Import evaluate_model() / plot_confusion_matrix() into your training script.")
