"""Train a baseline classifier on extracted audio features."""

import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_baseline(
    features_df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train a baseline RandomForest classifier on a feature table.

    Args:
        features_df: DataFrame where each row is one audio sample, one
            column is `label_col`, and the rest are numeric features.
        label_col: Name of the target column.
        test_size: Fraction of data held out for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (trained_model, scaler, X_test, y_test).
    """
    X = features_df.drop(columns=[label_col])
    y = features_df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test_scaled, y_test


def save_model(model, scaler, path: str = "models/saved_models/baseline.pkl") -> None:
    """Persist a trained model and its scaler to disk."""
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)


if __name__ == "__main__":
    # Example usage — replace with your real feature table (e.g. from
    # src/audio/feature_extraction.py output, saved as CSV).
    df = pd.read_csv("data/processed/features.csv")
    model, scaler, X_test, y_test = train_baseline(df)
    save_model(model, scaler)
    print("Model trained and saved to models/saved_models/baseline.pkl")
