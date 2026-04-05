"""
Streamlit input schema for the final football match prediction app.

This module maps user-entered UI fields to the exact model feature names.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


UI_TO_MODEL_FEATURE_MAP: Dict[str, str] = {
    # Context features
    "matchday": "matchday",
    "home_pre_position": "Home_pre_po",
    "away_pre_position": "Away_pre_po",
    "hour": "Hour",
    "home_form": "HomeTeam_Form",
    "away_form": "AwayTeam_Form",
    "home_advantage": "HomeTeam_HomeAdvantage",

    # Home team last-5 metrics
    "home_goals_scored_last5": "Home_GoalsFor_roll5",
    "home_goals_conceded_last5": "Home_GoalsAgainst_roll5",
    "home_shots_for_last5": "Home_ShotsFor_roll5",
    "home_shots_against_last5": "Home_ShotsAgainst_roll5",
    "home_shots_on_target_for_last5": "Home_ShotsOnTargetFor_roll5",
    "home_shots_on_target_against_last5": "Home_ShotsOnTargetAgainst_roll5",
    "home_fouls_for_last5": "Home_FoulsFor_roll5",
    "home_fouls_against_last5": "Home_FoulsAgainst_roll5",
    "home_corners_for_last5": "Home_CornersFor_roll5",
    "home_corners_against_last5": "Home_CornersAgainst_roll5",
    "home_yellow_cards_for_last5": "Home_YellowCardsFor_roll5",
    "home_yellow_cards_against_last5": "Home_YellowCardsAgainst_roll5",
    "home_red_cards_for_last5": "Home_RedCardsFor_roll5",
    "home_red_cards_against_last5": "Home_RedCardsAgainst_roll5",

    # Away team last-5 metrics
    "away_goals_scored_last5": "Away_GoalsFor_roll5",
    "away_goals_conceded_last5": "Away_GoalsAgainst_roll5",
    "away_shots_for_last5": "Away_ShotsFor_roll5",
    "away_shots_against_last5": "Away_ShotsAgainst_roll5",
    "away_shots_on_target_for_last5": "Away_ShotsOnTargetFor_roll5",
    "away_shots_on_target_against_last5": "Away_ShotsOnTargetAgainst_roll5",
    "away_fouls_for_last5": "Away_FoulsFor_roll5",
    "away_fouls_against_last5": "Away_FoulsAgainst_roll5",
    "away_corners_for_last5": "Away_CornersFor_roll5",
    "away_corners_against_last5": "Away_CornersAgainst_roll5",
    "away_yellow_cards_for_last5": "Away_YellowCardsFor_roll5",
    "away_yellow_cards_against_last5": "Away_YellowCardsAgainst_roll5",
    "away_red_cards_for_last5": "Away_RedCardsFor_roll5",
    "away_red_cards_against_last5": "Away_RedCardsAgainst_roll5",
}


def required_ui_fields() -> List[str]:
    return list(UI_TO_MODEL_FEATURE_MAP.keys())


def build_feature_ready_row(user_inputs: Dict[str, float]) -> pd.DataFrame:
    """
    Convert UI inputs to one-row dataframe with exact model feature names.
    """
    row = {}

    for ui_field, model_feature in UI_TO_MODEL_FEATURE_MAP.items():
        if ui_field not in user_inputs:
            raise ValueError(f"Missing required UI field: {ui_field}")
        row[model_feature] = user_inputs[ui_field]

    return pd.DataFrame([row])