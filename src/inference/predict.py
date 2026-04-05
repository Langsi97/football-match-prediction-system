"""
Prediction utilities for the final tuned Random Forest model.

This module:
- loads the trained model, preprocessor, and label encoder
- aligns incoming features to the exact training schema
- returns class prediction and probabilities
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd


MODEL_PATH = Path("artifacts/models/random_forest_best.joblib")
PREPROCESSOR_PATH = Path("artifacts/preprocessor.joblib")
ENCODER_PATH = Path("artifacts/label_encoder.joblib")


def load_artifacts() -> Tuple[object, object, object]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found: {ENCODER_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, preprocessor, encoder


def align_feature_columns(X: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """
    Align input dataframe to the exact feature order used in training.
    """
    if not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError("Preprocessor does not expose feature_names_in_.")

    expected_columns = list(preprocessor.feature_names_in_)

    missing_columns = [col for col in expected_columns if col not in X.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X_aligned = X[expected_columns].copy()
    return X_aligned


def predict_from_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict class and probabilities from a feature-ready dataframe.
    """
    model, preprocessor, encoder = load_artifacts()

    X_aligned = align_feature_columns(feature_df, preprocessor)
    X_processed = preprocessor.transform(X_aligned)

    predicted_encoded = model.predict(X_processed)
    predicted_labels = encoder.inverse_transform(predicted_encoded)

    probabilities = model.predict_proba(X_processed)
    class_labels = encoder.inverse_transform(model.classes_)

    results = pd.DataFrame({"prediction": predicted_labels})

    for i, label in enumerate(class_labels):
        results[f"prob_{label}"] = probabilities[:, i]

    return results