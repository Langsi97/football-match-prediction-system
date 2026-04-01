"""
Execution script for preprocessing.

Inputs:
    data/processed/train.csv
    data/processed/test.csv

Outputs:
    data/processed/X_train_processed.csv
    data/processed/X_test_processed.csv
    data/processed/y_train.csv
    data/processed/y_test.csv
    artifacts/preprocessor.joblib
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features.preprocessing import (
    fit_transform_preprocessor,
    select_model_columns,
)


TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

X_TRAIN_OUT = Path("data/processed/X_train_processed.csv")
X_TEST_OUT = Path("data/processed/X_test_processed.csv")
Y_TRAIN_OUT = Path("data/processed/y_train.csv")
Y_TEST_OUT = Path("data/processed/y_test.csv")

PREPROCESSOR_OUT = Path("artifacts/preprocessor.joblib")


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    selected = select_model_columns(
        train_df=train_df,
        test_df=test_df,
        target_column="FTR",
        metadata_columns=["Date", "HomeTeam", "AwayTeam"],
    )

    preprocessor, X_train_processed, X_test_processed = fit_transform_preprocessor(
        X_train=selected["X_train"],
        X_test=selected["X_test"],
    )

    X_TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_OUT.parent.mkdir(parents=True, exist_ok=True)

    X_train_processed.to_csv(X_TRAIN_OUT, index=False)
    X_test_processed.to_csv(X_TEST_OUT, index=False)
    selected["y_train"].to_frame(name="FTR").to_csv(Y_TRAIN_OUT, index=False)
    selected["y_test"].to_frame(name="FTR").to_csv(Y_TEST_OUT, index=False)

    joblib.dump(preprocessor, PREPROCESSOR_OUT)

    print("Preprocessing completed.")
    print(f"X_train processed shape: {X_train_processed.shape}")
    print(f"X_test processed shape : {X_test_processed.shape}")
    print(f"Saved preprocessor     : {PREPROCESSOR_OUT}")


if __name__ == "__main__":
    main()