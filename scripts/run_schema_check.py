"""
Check schema consistency across all season CSV files.

Run:
    python -m scripts.run_schema_check
"""

from src.utils.config import load_yaml, CONFIG_DIR
from src.utils.paths import RAW_DATA_DIR
from src.data.loaders import load_csv
from src.data.validate import get_column_report, find_schema_differences


def main() -> None:
    """
    Load season files and print schema consistency information.
    """
    config = load_yaml(CONFIG_DIR / "data.yaml")
    season_files = config["raw_data"]["season_files"]

    dataframes = {}
    for filename in season_files:
        file_path = RAW_DATA_DIR / filename
        dataframes[filename] = load_csv(file_path)

    report = get_column_report(dataframes)
    differences = find_schema_differences(dataframes)

    print("=== Schema Summary ===")
    print(f"Files checked: {len(dataframes)}")
    print(f"Common columns: {len(report['common_columns'])}")
    print(f"Union columns: {len(report['union_columns'])}")

    print("\n=== Per-file shape ===")
    for name, df in dataframes.items():
        print(f"{name}: shape={df.shape}")

    print("\n=== Schema differences ===")
    for name, diff in differences.items():
        print(f"\n{name}")
        print(f"Missing columns: {diff['missing_columns']}")
        print(f"Extra columns: {diff['extra_columns']}")


if __name__ == "__main__":
    main()