"""
Model evaluation utilities.

Evaluates trained classification models using:
- accuracy
- precision (weighted)
- recall (weighted + macro)
- F1 (weighted)
- Brier score

This version supports encoded targets produced for XGBoost training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def load_test_data(
    X_test_path: str,
    y_test_path: str,
    label_encoder_path: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load test data and encode target labels using the saved encoder.
    """
    X_test = pd.read_csv(X_test_path)
    y_test_raw = pd.read_csv(y_test_path)["FTR"]

    label_encoder = joblib.load(label_encoder_path)
    y_test_encoded = pd.Series(
        label_encoder.transform(y_test_raw),
        name="FTR",
    )

    return X_test, y_test_encoded


def load_models(model_dir: str) -> Dict[str, object]:
    """
    Load all saved models from directory.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    models: Dict[str, object] = {}
    for file_path in model_path.glob("*.joblib"):
        models[file_path.stem] = joblib.load(file_path)

    if not models:
        raise ValueError(f"No saved models found in: {model_path}")

    return models


def compute_brier_score(y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> float:
    """
    Compute multiclass Brier score.
    Lower is better.
    """
    y_true_encoded = np.zeros_like(y_proba)

    for i, cls in enumerate(classes):
        y_true_encoded[:, i] = (y_true == cls).astype(int)

    return float(np.mean(np.sum((y_proba - y_true_encoded) ** 2, axis=1)))


def evaluate_models(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate all models and return a comparison table.
    """
    rows: List[dict] = []

    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            brier = compute_brier_score(
                y_true=y_test.values,
                y_proba=y_proba,
                classes=model.classes_,
            )
        else:
            brier = np.nan

        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_weighted": precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "recall_weighted": recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "recall_macro": recall_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "f1_weighted": f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "brier_score": brier,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        by=["recall_macro", "recall_weighted", "f1_weighted", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    return results