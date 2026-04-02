"""
Rolling feature engineering for football match data.

This module builds leakage-free rolling averages for team performance
statistics. The rolling window is configurable so training and inference
can share the same historical definition.

Output columns follow:
- Home_<Metric>_roll{window}
- Away_<Metric>_roll{window}
"""

from __future__ import annotations

from typing import Tuple, List

import pandas as pd


REQUIRED_COLUMNS = {
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
}


TEAM_METRIC_COLUMNS = [
    "GoalsFor",
    "GoalsAgainst",
    "ShotsFor",
    "ShotsAgainst",
    "ShotsOnTargetFor",
    "ShotsOnTargetAgainst",
    "FoulsFor",
    "FoulsAgainst",
    "CornersFor",
    "CornersAgainst",
    "YellowCardsFor",
    "YellowCardsAgainst",
    "RedCardsFor",
    "RedCardsAgainst",
]


def validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _build_home_team_view(df: pd.DataFrame) -> pd.DataFrame:
    home_df = df[
        [
            "MatchID",
            "Date",
            "HomeTeam",
            "FTHG",
            "FTAG",
            "HS",
            "AS",
            "HST",
            "AST",
            "HF",
            "AF",
            "HC",
            "AC",
            "HY",
            "AY",
            "HR",
            "AR",
        ]
    ].copy()

    home_df = home_df.rename(
        columns={
            "HomeTeam": "Team",
            "FTHG": "GoalsFor",
            "FTAG": "GoalsAgainst",
            "HS": "ShotsFor",
            "AS": "ShotsAgainst",
            "HST": "ShotsOnTargetFor",
            "AST": "ShotsOnTargetAgainst",
            "HF": "FoulsFor",
            "AF": "FoulsAgainst",
            "HC": "CornersFor",
            "AC": "CornersAgainst",
            "HY": "YellowCardsFor",
            "AY": "YellowCardsAgainst",
            "HR": "RedCardsFor",
            "AR": "RedCardsAgainst",
        }
    )
    home_df["Venue"] = "Home"
    return home_df


def _build_away_team_view(df: pd.DataFrame) -> pd.DataFrame:
    away_df = df[
        [
            "MatchID",
            "Date",
            "AwayTeam",
            "FTAG",
            "FTHG",
            "AS",
            "HS",
            "AST",
            "HST",
            "AF",
            "HF",
            "AC",
            "HC",
            "AY",
            "HY",
            "AR",
            "HR",
        ]
    ].copy()

    away_df = away_df.rename(
        columns={
            "AwayTeam": "Team",
            "FTAG": "GoalsFor",
            "FTHG": "GoalsAgainst",
            "AS": "ShotsFor",
            "HS": "ShotsAgainst",
            "AST": "ShotsOnTargetFor",
            "HST": "ShotsOnTargetAgainst",
            "AF": "FoulsFor",
            "HF": "FoulsAgainst",
            "AC": "CornersFor",
            "HC": "CornersAgainst",
            "AY": "YellowCardsFor",
            "HY": "YellowCardsAgainst",
            "AR": "RedCardsFor",
            "HR": "RedCardsAgainst",
        }
    )
    away_df["Venue"] = "Away"
    return away_df


def build_team_centric_match_table(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df, REQUIRED_COLUMNS)

    working_df = df.copy()
    working_df["Date"] = pd.to_datetime(working_df["Date"], errors="coerce")

    if working_df["Date"].isna().any():
        raise ValueError("Date column contains invalid values after datetime conversion.")

    if "MatchID" not in working_df.columns:
        working_df["MatchID"] = range(len(working_df))

    home_df = _build_home_team_view(working_df)
    away_df = _build_away_team_view(working_df)

    team_df = pd.concat([home_df, away_df], axis=0, ignore_index=True)
    team_df = team_df.sort_values(["Team", "Date", "MatchID"]).reset_index(drop=True)

    return team_df


def add_rolling_features(team_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add leakage-free rolling means using only previous matches.
    """
    result = team_df.copy()

    result[TEAM_METRIC_COLUMNS] = (
        result.groupby("Team", group_keys=False)[TEAM_METRIC_COLUMNS]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
    )

    return result


def split_home_away_rolling_features(
    team_df: pd.DataFrame,
    window: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    suffix = f"_roll{window}"

    home_df = team_df.loc[team_df["Venue"] == "Home", ["MatchID"] + TEAM_METRIC_COLUMNS].copy()
    away_df = team_df.loc[team_df["Venue"] == "Away", ["MatchID"] + TEAM_METRIC_COLUMNS].copy()

    home_df = home_df.rename(columns={col: f"Home_{col}{suffix}" for col in TEAM_METRIC_COLUMNS})
    away_df = away_df.rename(columns={col: f"Away_{col}{suffix}" for col in TEAM_METRIC_COLUMNS})

    return home_df, away_df


def merge_rolling_features_back(
    matches_df: pd.DataFrame,
    home_roll_df: pd.DataFrame,
    away_roll_df: pd.DataFrame,
) -> pd.DataFrame:
    result = matches_df.copy()

    if "MatchID" not in result.columns:
        result["MatchID"] = range(len(result))

    original_row_count = len(result)

    result = result.merge(home_roll_df, on="MatchID", how="left")
    result = result.merge(away_roll_df, on="MatchID", how="left")

    if len(result) != original_row_count:
        raise ValueError(
            f"Row count changed after rolling feature merge: {original_row_count} -> {len(result)}"
        )

    return result


def build_rolling_feature_dataset(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    working_df = df.copy()

    if "MatchID" not in working_df.columns:
        working_df["MatchID"] = range(len(working_df))

    team_df = build_team_centric_match_table(working_df)
    team_df = add_rolling_features(team_df, window=window)
    home_roll_df, away_roll_df = split_home_away_rolling_features(team_df, window=window)

    final_df = merge_rolling_features_back(working_df, home_roll_df, away_roll_df)

    if len(final_df) != len(df):
        raise ValueError("Final rolling feature dataset does not preserve row count.")

    return final_df