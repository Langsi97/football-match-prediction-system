"""
Standardize and merge all season CSV files.

Run:
    python -m scripts.run_merge_data
"""

from src.utils.config import load_yaml, CONFIG_DIR
from src.utils.paths import RAW_DATA_DIR, INTERIM_DATA_DIR
from src.data.loaders import load_csv
from src.data.merge import standardize_season_dataframe, merge_seasons


def main() -> None:
    """
    Load, standardize, merge, and save all season datasets.
    """
    config = load_yaml(CONFIG_DIR / "data.yaml")
    season_files = config["raw_data"]["season_files"]

    standardized_dfs = []

    print("Loading and standardizing season files...")
    for filename in season_files:
        file_path = RAW_DATA_DIR / filename
        df = load_csv(file_path)
        df = standardize_season_dataframe(df)
        standardized_dfs.append(df)
        print(f"{filename}: standardized shape={df.shape}")

    merged_df = merge_seasons(standardized_dfs)

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERIM_DATA_DIR / "merged_seasons.csv"
    merged_df.to_csv(output_path, index=False)

    print("\nMerge completed successfully.")
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()