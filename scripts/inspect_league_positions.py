"""
Inspect the league position dataset structure.

Run:
    python -m scripts.inspect_league_positions
"""

from src.utils.config import load_yaml, CONFIG_DIR
from src.utils.paths import EXTERNAL_DATA_DIR
from src.data.loaders import load_excel


def main() -> None:
    config = load_yaml(CONFIG_DIR / "data.yaml")
    filename = config["external_data"]["league_position_file"]

    file_path = EXTERNAL_DATA_DIR / filename
    df = load_excel(file_path)

    print("=== League Position Dataset Info ===")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")

    print("\nDtypes:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()