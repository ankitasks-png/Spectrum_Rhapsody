"""Run inference on new audio samples using a trained model."""

import pickle

import numpy as np

from src.audio.feature_extraction import extract_all_features
from src.audio.preprocessing import preprocess


def load_model(path: str = "models/saved_models/baseline.pkl"):
    """Load a trained model + scaler bundle from disk."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"]


def predict_from_file(file_path: str, model_path: str = "models/saved_models/baseline.pkl") -> dict:
    """Run the full pipeline on a raw audio file and return a prediction.

    Returns:
        Dict with the extracted features, predicted class, and class
        probabilities.
    """
    model, scaler = load_model(model_path)

    y, sr = preprocess(file_path)
    features = extract_all_features(y, sr)

    feature_vector = np.array([list(features.values())])
    feature_vector_scaled = scaler.transform(feature_vector)

    prediction = model.predict(feature_vector_scaled)[0]
    probabilities = model.predict_proba(feature_vector_scaled)[0]

    return {
        "features": features,
        "prediction": prediction,
        "probabilities": dict(zip(model.classes_, probabilities.tolist())),
    }


if __name__ == "__main__":
    result = predict_from_file("data/raw/sample_042.wav")
    print(result)
