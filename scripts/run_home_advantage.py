"""
Execution script for home advantage feature generation.

Input:
    data/processed/matches_with_rolling_and_form.csv

Output:
    data/processed/matches_with_all_features.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.home_advantage import build_home_advantage_dataset


INPUT_PATH = Path("data/processed/matches_with_rolling_and_form.csv")
OUTPUT_PATH = Path("data/processed/matches_with_all_features.csv")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    original_rows = len(df)

    output_df = build_home_advantage_dataset(df)

    if len(output_df) != original_rows:
        raise ValueError(
            f"Row count mismatch after home advantage generation: "
            f"{original_rows} -> {len(output_df)}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print("Home advantage generated successfully.")
    print(f"Input shape : {df.shape}")
    print(f"Output shape: {output_df.shape}")
    print("Added column: HomeTeam_HomeAdvantage")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()