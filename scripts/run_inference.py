"""
Run inference using trained model.
"""

from __future__ import annotations

import pandas as pd

from src.inference.predict import predict


def main():
    df = pd.read_csv("data/processed/X_test_processed.csv")

    results = predict(df)

    print("\nPredictions:")
    print(results.head())


if __name__ == "__main__":
    main()