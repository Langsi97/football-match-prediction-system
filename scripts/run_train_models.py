"""
Execution script for model training.

Inputs:
    data/processed/X_train_resampled.csv
    data/processed/y_train_resampled.csv

Outputs:
    artifacts/models/
    artifacts/label_encoder.joblib
"""

from __future__ import annotations

from pathlib import Path

from src.models.train_models import (
    load_training_data,
    save_label_encoder,
    save_models,
    train_models,
)


X_TRAIN_PATH = Path("data/processed/X_train_resampled.csv")
Y_TRAIN_PATH = Path("data/processed/y_train_resampled.csv")

MODEL_OUTPUT_DIR = Path("artifacts/models")
LABEL_ENCODER_PATH = Path("artifacts/label_encoder.joblib")


def main() -> None:
    if not X_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {X_TRAIN_PATH}")
    if not Y_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {Y_TRAIN_PATH}")

    X_train, y_train, label_encoder = load_training_data(
        X_train_path=str(X_TRAIN_PATH),
        y_train_path=str(Y_TRAIN_PATH),
    )

    models = train_models(X_train=X_train, y_train=y_train)

    save_models(models=models, output_dir=str(MODEL_OUTPUT_DIR))
    save_label_encoder(label_encoder=label_encoder, output_path=str(LABEL_ENCODER_PATH))

    print("Model training completed.")


if __name__ == "__main__":
    main()