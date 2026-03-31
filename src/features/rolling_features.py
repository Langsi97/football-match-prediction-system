"""
Rolling feature engineering for Belgian Pro League match data.

This module builds leakage-free 3-match rolling averages for team performance
statistics. The implementation is team-centric:

1. Each match is expanded into two rows:
   - one row from the home team's perspective
   - one row from the away team's perspective

2. Features are shifted by 1 match before rolling, so the current match is
   never included in its own feature calculation.

3. The engineered rolling features are merged back into the original
   match-level dataset as:
   - Home_<feature>_roll3
   - Away_<feature>_roll3

This module is designed for production ML pipelines:
- explicit validation
- deterministic ordering
- no target leakage
- reusable functions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


ROLLING_WINDOW = 5


@dataclass(frozen=True)
class RollingFeatureSpec:
    """
    Defines how home/away raw match columns map into generic team-centric metrics.
    """

    generic_name: str
    home_for: str
    home_against: str
    away_for: str
    away_against: str


ROLLING_FEATURE_SPECS: List[RollingFeatureSpec] = [
    RollingFeatureSpec("Goals", "FTHG", "FTAG", "FTAG", "FTHG"),
    RollingFeatureSpec("Shots", "HS", "AS", "AS", "HS"),
    RollingFeatureSpec("ShotsOnTarget", "HST", "AST", "AST", "HST"),
    RollingFeatureSpec("Fouls", "HF", "AF", "AF", "HF"),
    RollingFeatureSpec("Corners", "HC", "AC", "AC", "HC"),
    RollingFeatureSpec("YellowCards", "HY", "AY", "AY", "HY"),
    RollingFeatureSpec("RedCards", "HR", "AR", "AR", "HR"),
]


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


def validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """
    Validate that the dataframe contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    required_columns : set[str]
        Required column names.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _build_home_team_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a team-centric dataframe from the home team's perspective.
    """
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
    """
    Build a team-centric dataframe from the away team's perspective.
    """
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
    """
    Convert match-level data into a team-centric table with one row per team per match.

    Parameters
    ----------
    df : pd.DataFrame
        Match-level dataframe.

    Returns
    -------
    pd.DataFrame
        Team-centric dataframe with both home and away team perspectives.
    """
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

    # Deterministic ordering is important for reproducibility.
    team_df = team_df.sort_values(["Team", "Date", "MatchID"], ascending=[True, True, True]).reset_index(drop=True)
    return team_df


def add_rolling_features(
    team_df: pd.DataFrame,
    rolling_window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Add shifted rolling mean features to the team-centric dataframe.

    Parameters
    ----------
    team_df : pd.DataFrame
        Team-centric dataframe.
    rolling_window : int, default=3
        Number of previous matches to include in the rolling average.

    Returns
    -------
    pd.DataFrame
        Team-centric dataframe with rolling features added.
    """
    rolling_input_columns = [
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

    for col in rolling_input_columns:
        if col not in team_df.columns:
            raise ValueError(f"Expected rolling input column '{col}' not found.")

    result = team_df.copy()

    # Shift by 1 before rolling to ensure causal, leakage-free features.
    result[rolling_input_columns] = (
        result.groupby("Team", group_keys=False)[rolling_input_columns]
        .transform(lambda s: s.shift(1).rolling(window=rolling_window, min_periods=1).mean())
    )

    return result


def split_home_away_rolling_features(team_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split team-centric rolling features back into home and away match-level features.

    Parameters
    ----------
    team_df : pd.DataFrame
        Team-centric dataframe containing rolling features.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (home_features_df, away_features_df)
    """
    rolling_columns = [
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

    home_df = team_df.loc[team_df["Venue"] == "Home", ["MatchID"] + rolling_columns].copy()
    away_df = team_df.loc[team_df["Venue"] == "Away", ["MatchID"] + rolling_columns].copy()

    home_df = home_df.rename(columns={col: f"Home_{col}_roll3" for col in rolling_columns})
    away_df = away_df.rename(columns={col: f"Away_{col}_roll3" for col in rolling_columns})

    return home_df, away_df


def merge_rolling_features_back(
    matches_df: pd.DataFrame,
    home_roll_df: pd.DataFrame,
    away_roll_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge home and away rolling features back into the original match dataframe.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Original match dataframe.
    home_roll_df : pd.DataFrame
        Home rolling features by MatchID.
    away_roll_df : pd.DataFrame
        Away rolling features by MatchID.

    Returns
    -------
    pd.DataFrame
        Final dataframe with rolling features added.
    """
    merged = matches_df.copy()

    if "MatchID" not in merged.columns:
        merged["MatchID"] = range(len(merged))

    original_row_count = len(merged)

    merged = merged.merge(home_roll_df, on="MatchID", how="left")
    merged = merged.merge(away_roll_df, on="MatchID", how="left")

    if len(merged) != original_row_count:
        raise ValueError(
            f"Row count changed after rolling feature merge: "
            f"{original_row_count} -> {len(merged)}"
        )

    return merged


def build_rolling_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end builder for rolling features.

    Parameters
    ----------
    df : pd.DataFrame
        Input match-level dataframe.

    Returns
    -------
    pd.DataFrame
        Match-level dataframe enriched with rolling features.
    """
    working_df = df.copy()
    if "MatchID" not in working_df.columns:
        working_df["MatchID"] = range(len(working_df))

    team_df = build_team_centric_match_table(working_df)
    team_df = add_rolling_features(team_df, rolling_window=ROLLING_WINDOW)
    home_roll_df, away_roll_df = split_home_away_rolling_features(team_df)

    final_df = merge_rolling_features_back(working_df, home_roll_df, away_roll_df)

    if len(final_df) != len(df):
        raise ValueError("Final rolling feature dataset does not preserve row count.")

    return final_df