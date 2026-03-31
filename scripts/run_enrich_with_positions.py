"""
Merge the preprocessed match dataset with league-position data.

Run:
    python -m scripts.run_enrich_with_positions
"""

import pandas as pd

from src.utils.config import load_yaml, CONFIG_DIR
from src.utils.paths import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR
from src.data.loaders import load_excel
from src.data.enrich import merge_with_league_positions, validate_row_count


def main() -> None:
    """
    Load datasets, merge them, validate row count,
    and save the enriched output.
    """
    config = load_yaml(CONFIG_DIR / "data.yaml")
    league_position_file = config["external_data"]["league_position_file"]

    matches_path = PROCESSED_DATA_DIR / "matches_base_preprocessed.csv"
    league_positions_path = EXTERNAL_DATA_DIR / league_position_file
    output_path = PROCESSED_DATA_DIR / "matches_enriched_with_positions.csv"

    matches_df = pd.read_csv(matches_path)
    league_positions_df = load_excel(league_positions_path)

    print(f"Matches dataset shape: {matches_df.shape}")
    print(f"League position dataset shape: {league_positions_df.shape}")

    merged_df = merge_with_league_positions(
        matches_df=matches_df,
        league_positions_df=league_positions_df,
    )

    validate_row_count(merged_df, expected_rows=league_positions_df.shape[0])

    merged_df.to_csv(output_path, index=False)

    print("\nMerge completed successfully.")
    print(f"Final enriched dataset shape: {merged_df.shape}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()