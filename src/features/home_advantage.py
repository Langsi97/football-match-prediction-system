"""
Home advantage feature engineering.

This module computes a leakage-free home advantage feature for the home team
using up to the previous 5 HOME matches only.

Definition
----------
HomeTeam_HomeAdvantage =
    points from last <= 5 previous home matches / (3 * number_of_matches_used)

Points mapping from FTR:
- H -> 3
- D -> 1
- A -> 0

Output column
-------------
- HomeTeam_HomeAdvantage
"""

from __future__ import annotations

import pandas as pd


FORM_WINDOW = 5
REQUIRED_COLUMNS = {"Date", "HomeTeam", "FTR"}


def validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_home_points_from_ftr(ftr: str) -> int:
    if ftr == "H":
        return 3
    if ftr == "D":
        return 1
    return 0


def build_home_advantage_dataset(
    df: pd.DataFrame,
    window: int = FORM_WINDOW,
) -> pd.DataFrame:
    """
    Build leakage-free home advantage feature.

    Parameters
    ----------
    df : pd.DataFrame
        Match-level dataframe.
    window : int, default=5
        Number of previous home matches to use.

    Returns
    -------
    pd.DataFrame
        Original dataframe with HomeTeam_HomeAdvantage added.
    """
    validate_required_columns(df, REQUIRED_COLUMNS)

    result = df.copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")

    if result["Date"].isna().any():
        raise ValueError("Date column contains invalid values after datetime conversion.")

    if "MatchID" not in result.columns:
        result["MatchID"] = range(len(result))

    original_row_count = len(result)

    home_history = result[["MatchID", "Date", "HomeTeam", "FTR"]].copy()
    home_history["HomePoints"] = home_history["FTR"].map(compute_home_points_from_ftr)

    home_history = home_history.sort_values(
        ["HomeTeam", "Date", "MatchID"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    shifted_points = home_history.groupby("HomeTeam")["HomePoints"].transform(lambda s: s.shift(1))

    rolling_sum = shifted_points.groupby(home_history["HomeTeam"]).transform(
        lambda s: s.rolling(window=window, min_periods=1).sum()
    )
    rolling_count = shifted_points.groupby(home_history["HomeTeam"]).transform(
        lambda s: s.rolling(window=window, min_periods=1).count()
    )

    home_history["HomeTeam_HomeAdvantage"] = rolling_sum / (3 * rolling_count)

    feature_df = home_history[["MatchID", "HomeTeam_HomeAdvantage"]].copy()

    result = result.merge(feature_df, on="MatchID", how="left")

    if len(result) != original_row_count:
        raise ValueError(
            f"Row count changed after home advantage merge: "
            f"{original_row_count} -> {len(result)}"
        )

    return result