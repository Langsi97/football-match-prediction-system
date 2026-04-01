"""
Model training module.

Trains:
- Logistic Regression
- Random Forest
- XGBoost

Notes
-----
- Logistic Regression and XGBoost require numeric targets.
- Random Forest can handle string labels, but we standardize all models
  to use the same encoded target space for consistency.
- The label encoder is saved separately for inverse mapping during inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def load_training_data(
    X_train_path: str,
    y_train_path: str,
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Load processed training data and encode target labels.

    Parameters
    ----------
    X_train_path : str
        Path to processed/resampled training features.
    y_train_path : str
        Path to processed/resampled training target.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, LabelEncoder]
        Training features, encoded target, fitted label encoder.
    """
    X_train = pd.read_csv(X_train_path)
    y_train_raw = pd.read_csv(y_train_path)["FTR"]

    label_encoder = LabelEncoder()
    y_train_encoded = pd.Series(
        label_encoder.fit_transform(y_train_raw),
        name="FTR",
    )

    return X_train, y_train_encoded, label_encoder


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, object]:
    """
    Train all baseline and boosted models.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Encoded training target.

    Returns
    -------
    dict[str, object]
        Dictionary of trained models.
    """
    models: Dict[str, object] = {}

    # Logistic Regression
    logistic_regression = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    logistic_regression.fit(X_train, y_train)
    models["logistic_regression"] = logistic_regression

    # Random Forest
    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    random_forest.fit(X_train, y_train)
    models["random_forest"] = random_forest

    # XGBoost
    xgboost_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    xgboost_model.fit(X_train, y_train)
    models["xgboost"] = xgboost_model

    return models


def save_models(models: Dict[str, object], output_dir: str) -> None:
    """
    Save trained models to disk.

    Parameters
    ----------
    models : dict[str, object]
        Trained model dictionary.
    output_dir : str
        Output directory path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        model_path = output_path / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model: {model_path}")


def save_label_encoder(label_encoder: LabelEncoder, output_path: str) -> None:
    """
    Save fitted label encoder to disk.

    Parameters
    ----------
    label_encoder : LabelEncoder
        Fitted label encoder.
    output_path : str
        Save path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, path)
    print(f"Saved label encoder: {path}")