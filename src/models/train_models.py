"""
Model training module.

Trains baseline models:
- Logistic Regression
- Random Forest
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def load_training_data(
    X_train_path: str,
    y_train_path: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load processed training features and target.
    """
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)["FTR"]
    return X_train, y_train


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    """
    Train selected baseline models.
    """
    models: Dict[str, object] = {}

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    return models


def save_models(models: Dict[str, object], output_dir: str) -> None:
    """
    Save trained models to disk.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        model_path = output_path / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model: {model_path}")