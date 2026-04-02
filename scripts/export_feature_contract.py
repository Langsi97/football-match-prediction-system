"""
Export the final feature contract used by the trained preprocessor.

Outputs:
    artifacts/reports/final_feature_contract.csv
    artifacts/reports/final_feature_contract.txt
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


PREPROCESSOR_PATH = Path("artifacts/preprocessor.joblib")
CSV_OUTPUT_PATH = Path("artifacts/reports/final_feature_contract.csv")
TXT_OUTPUT_PATH = Path("artifacts/reports/final_feature_contract.txt")


def main() -> None:
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    if not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError("Loaded preprocessor does not expose feature_names_in_.")

    feature_names = list(preprocessor.feature_names_in_)

    df = pd.DataFrame(
        {
            "feature_order": range(1, len(feature_names) + 1),
            "feature_name": feature_names,
        }
    )

    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUTPUT_PATH, index=False)

    with TXT_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for idx, feature in enumerate(feature_names, start=1):
            f.write(f"{idx}. {feature}\n")

    print("Feature contract exported successfully.")
    print(f"Feature count: {len(feature_names)}")
    print(f"CSV saved to : {CSV_OUTPUT_PATH}")
    print(f"TXT saved to : {TXT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()