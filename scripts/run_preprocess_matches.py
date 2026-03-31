"""
Preprocess the merged match dataset.

Run:
    python -m scripts.run_preprocess_matches
"""

from src.utils.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.data.preprocess import (
    trim_columns_up_to_avg_a,
    drop_irrelevant_columns,
    preprocess_date_and_time,
    add_season_column,
)

import pandas as pd


def main() -> None:
    """
    Load merged season data, apply base preprocessing,
    and save the processed dataset.
    """
    input_path = INTERIM_DATA_DIR / "merged_seasons.csv"
    output_path = PROCESSED_DATA_DIR / "matches_base_preprocessed.csv"

    df = pd.read_csv(input_path)

    print(f"Loaded merged dataset: shape={df.shape}")

    df = trim_columns_up_to_avg_a(df)
    print(f"After trimming columns to AvgA: shape={df.shape}")

    df = drop_irrelevant_columns(df)
    print(f"After dropping irrelevant columns: shape={df.shape}")

    df = preprocess_date_and_time(df)
    df = add_season_column(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nPreprocessing completed successfully.")
    print(f"Final processed dataset shape: {df.shape}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()