from pydantic import BaseModel


class PredictionRequest(BaseModel):
    matchday: float
    hour: float
    home_pre_position: float
    away_pre_position: float
    home_form: float
    away_form: float
    home_advantage: float

    home_goals_scored_last5: float
    home_goals_conceded_last5: float
    home_shots_for_last5: float
    home_shots_against_last5: float
    home_shots_on_target_for_last5: float
    home_shots_on_target_against_last5: float
    home_fouls_for_last5: float
    home_fouls_against_last5: float
    home_corners_for_last5: float
    home_corners_against_last5: float
    home_yellow_cards_for_last5: float
    home_yellow_cards_against_last5: float
    home_red_cards_for_last5: float
    home_red_cards_against_last5: float

    away_goals_scored_last5: float
    away_goals_conceded_last5: float
    away_shots_for_last5: float
    away_shots_against_last5: float
    away_shots_on_target_for_last5: float
    away_shots_on_target_against_last5: float
    away_fouls_for_last5: float
    away_fouls_against_last5: float
    away_corners_for_last5: float
    away_corners_against_last5: float
    away_yellow_cards_for_last5: float
    away_yellow_cards_against_last5: float
    away_red_cards_for_last5: float
    away_red_cards_against_last5: float


class PredictionResponse(BaseModel):
    prediction: str
    prob_H: float
    prob_D: float
    prob_A: float