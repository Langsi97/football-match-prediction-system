"""
Execution script for feature audit.

Input:
    data/processed/matches_with_all_features.csv

Output:
    data/processed/matches_final_features.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.feature_audit import run_feature_audit


INPUT_PATH = Path("data/processed/matches_with_all_features.csv")
OUTPUT_PATH = Path("data/processed/matches_final_features.csv")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    output_df = run_feature_audit(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature audit completed.")
    print(f"Final shape: {output_df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()