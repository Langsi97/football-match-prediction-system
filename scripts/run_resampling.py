"""
Execution script for training-only class resampling.

Inputs:
    data/processed/X_train_processed.csv
    data/processed/y_train.csv

Outputs:
    data/processed/X_train_resampled.csv
    data/processed/y_train_resampled.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.resampling import oversample_target_classes


X_TRAIN_PATH = Path("data/processed/X_train_processed.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")

X_RESAMPLED_OUT = Path("data/processed/X_train_resampled.csv")
Y_RESAMPLED_OUT = Path("data/processed/y_train_resampled.csv")


def main() -> None:
    if not X_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {X_TRAIN_PATH}")
    if not Y_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {Y_TRAIN_PATH}")

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["FTR"]

    print("Original class distribution:")
    print(y_train.value_counts())

    X_resampled, y_resampled = oversample_target_classes(
        X_train=X_train,
        y_train=y_train,
        target_classes=["D"],  # notebook parity: oversample draw class
        random_state=42,
    )

    X_RESAMPLED_OUT.parent.mkdir(parents=True, exist_ok=True)
    X_resampled.to_csv(X_RESAMPLED_OUT, index=False)
    y_resampled.to_frame(name="FTR").to_csv(Y_RESAMPLED_OUT, index=False)

    print("\nResampled class distribution:")
    print(y_resampled.value_counts())

    print("\nResampling completed.")
    print(f"X_train original shape  : {X_train.shape}")
    print(f"X_train resampled shape : {X_resampled.shape}")
    print(f"Saved features          : {X_RESAMPLED_OUT}")
    print(f"Saved target            : {Y_RESAMPLED_OUT}")


if __name__ == "__main__":
    main()