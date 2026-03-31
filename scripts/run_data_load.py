"""
Load all configured raw season CSV files and the external league position file.

Run:
    python scripts/run_data_load.py
"""

from src.utils.config import load_yaml, CONFIG_DIR
from src.utils.paths import RAW_DATA_DIR, EXTERNAL_DATA_DIR
from src.data.loaders import load_csv, load_excel


def main() -> None:
    """
    Load all season CSV files and the league position Excel file
    defined in config/data.yaml.
    """
    config = load_yaml(CONFIG_DIR / "data.yaml")

    season_files = config["raw_data"]["season_files"]
    league_position_file = config["external_data"]["league_position_file"]

    print("Loading season files...")
    season_dataframes = []

    for filename in season_files:
        file_path = RAW_DATA_DIR / filename
        df = load_csv(file_path)
        season_dataframes.append(df)
        print(f"Loaded {filename}: shape={df.shape}")

    print("\nLoading league position file...")
    league_position_path = EXTERNAL_DATA_DIR / league_position_file
    league_position_df = load_excel(league_position_path)
    print(f"Loaded {league_position_file}: shape={league_position_df.shape}")

    print("\nSummary")
    print(f"Total season files loaded: {len(season_dataframes)}")
    print(f"League position rows: {league_position_df.shape[0]}")
    print("Data loading completed successfully.")


if __name__ == "__main__":
    main()