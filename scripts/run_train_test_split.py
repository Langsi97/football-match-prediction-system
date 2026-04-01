"""
Execution script for time-based train/test split.

Input:
    data/processed/matches_final_features.csv

Outputs:
    data/processed/train.csv
    data/processed/test.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.train_test_split import load_yaml_config, split_dataset_by_date


CONFIG_PATH = Path("config/data.yaml")
INPUT_PATH = Path("data/processed/matches_final_features.csv")
TRAIN_OUTPUT_PATH = Path("data/processed/train.csv")
TEST_OUTPUT_PATH = Path("data/processed/test.csv")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    config = load_yaml_config(CONFIG_PATH)
    split_date = config["split"]["test_start_date"]

    df = pd.read_csv(INPUT_PATH)

    train_df, test_df = split_dataset_by_date(
        df=df,
        split_date=split_date,
        date_column="Date",
    )

    TRAIN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print("Train/test split completed.")
    print(f"Split date   : {split_date}")
    print(f"Train shape  : {train_df.shape}")
    print(f"Test shape   : {test_df.shape}")
    print(f"Saved train  : {TRAIN_OUTPUT_PATH}")
    print(f"Saved test   : {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()