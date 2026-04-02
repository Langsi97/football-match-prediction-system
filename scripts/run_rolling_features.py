"""
Execution script for rolling feature generation.

Input:
    data/processed/matches_enriched_with_positions.csv

Output:
    data/processed/matches_with_rolling_features.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.rolling_features import build_rolling_feature_dataset


INPUT_PATH = Path("data/processed/matches_enriched_with_positions.csv")
OUTPUT_PATH = Path("data/processed/matches_with_rolling_features.csv")
ROLLING_WINDOW = 5


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    original_rows = len(df)

    output_df = build_rolling_feature_dataset(df, window=ROLLING_WINDOW)

    if len(output_df) != original_rows:
        raise ValueError(
            f"Row count mismatch after rolling feature generation: "
            f"{original_rows} -> {len(output_df)}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    new_columns = [col for col in output_df.columns if col.endswith(f"_roll{ROLLING_WINDOW}")]

    print("Rolling features generated successfully.")
    print(f"Window      : {ROLLING_WINDOW}")
    print(f"Input shape : {df.shape}")
    print(f"Output shape: {output_df.shape}")
    print(f"Added cols  : {len(new_columns)}")
    print(f"Saved to    : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()