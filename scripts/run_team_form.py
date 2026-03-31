"""
Execution script for building team form features.

Input:
    data/processed/matches_with_rolling_features.csv

Output:
    data/processed/matches_with_rolling_and_form.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.team_form import build_team_form_dataset


INPUT_PATH = Path("data/processed/matches_with_rolling_features.csv")
OUTPUT_PATH = Path("data/processed/matches_with_rolling_and_form.csv")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    original_rows = len(df)

    output_df = build_team_form_dataset(df)

    if len(output_df) != original_rows:
        raise ValueError(
            f"Row count mismatch after team form generation: "
            f"{original_rows} -> {len(output_df)}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    form_columns = ["HomeTeam_Form", "AwayTeam_Form"]

    print("Team form features generated successfully.")
    print(f"Input shape : {df.shape}")
    print(f"Output shape: {output_df.shape}")
    print(f"Added columns: {form_columns}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()