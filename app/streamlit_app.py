"""
Streamlit application for football match prediction.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.inference.input_schema import build_feature_ready_row
from src.inference.predict import predict_from_features


st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Football Match Prediction System")
st.caption("Enter last-5-match team statistics and core pre-match features to get prediction probabilities.")


def stat_input_block(prefix: str, label: str):
    st.subheader(label)

    c1, c2 = st.columns(2)

    inputs = {}
    inputs[f"{prefix}_goals_scored_last5"] = c1.number_input("Goals scored", min_value=0.0, value=1.0, step=0.1, key=f"{prefix}_goals_scored")
    inputs[f"{prefix}_goals_conceded_last5"] = c2.number_input("Goals conceded", min_value=0.0, value=1.0, step=0.1, key=f"{prefix}_goals_conceded")

    inputs[f"{prefix}_shots_for_last5"] = c1.number_input("Shots for", min_value=0.0, value=4.0, step=0.1, key=f"{prefix}_shots_for")
    inputs[f"{prefix}_shots_against_last5"] = c2.number_input("Shots against", min_value=0.0, value=4.0, step=0.1, key=f"{prefix}_shots_against")

    inputs[f"{prefix}_shots_on_target_for_last5"] = c1.number_input("Shots on target for", min_value=0.0, value=2.0, step=0.1, key=f"{prefix}_shots_ot_for")
    inputs[f"{prefix}_shots_on_target_against_last5"] = c2.number_input("Shots on target against", min_value=0.0, value=2.0, step=0.1, key=f"{prefix}_shots_ot_against")

    inputs[f"{prefix}_fouls_for_last5"] = c1.number_input("Fouls for", min_value=0.0, value=10.0, step=0.1, key=f"{prefix}_fouls_for")
    inputs[f"{prefix}_fouls_against_last5"] = c2.number_input("Fouls against", min_value=0.0, value=10.0, step=0.1, key=f"{prefix}_fouls_against")

    inputs[f"{prefix}_corners_for_last5"] = c1.number_input("Corners for", min_value=0.0, value=5.0, step=0.1, key=f"{prefix}_corners_for")
    inputs[f"{prefix}_corners_against_last5"] = c2.number_input("Corners against", min_value=0.0, value=5.0, step=0.1, key=f"{prefix}_corners_against")

    inputs[f"{prefix}_yellow_cards_for_last5"] = c1.number_input("Yellow cards for", min_value=0.0, value=1.5, step=0.1, key=f"{prefix}_yellow_for")
    inputs[f"{prefix}_yellow_cards_against_last5"] = c2.number_input("Yellow cards against", min_value=0.0, value=1.5, step=0.1, key=f"{prefix}_yellow_against")

    inputs[f"{prefix}_red_cards_for_last5"] = c1.number_input("Red cards for", min_value=0.0, value=0.1, step=0.1, key=f"{prefix}_red_for")
    inputs[f"{prefix}_red_cards_against_last5"] = c2.number_input("Red cards against", min_value=0.0, value=0.1, step=0.1, key=f"{prefix}_red_against")

    return inputs


with st.container():
    st.subheader("Match Context")
    c1, c2, c3, c4 = st.columns(4)

    matchday = c1.number_input("Matchday", min_value=1, max_value=40, value=10, step=1)
    hour = c2.number_input("Kickoff hour", min_value=0, max_value=23, value=20, step=1)
    home_pre_position = c3.number_input("Home pre-match position", min_value=1, max_value=20, value=6, step=1)
    away_pre_position = c4.number_input("Away pre-match position", min_value=1, max_value=20, value=10, step=1)

    c5, c6, c7 = st.columns(3)
    home_form = c5.number_input("Home form (0-1)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    away_form = c6.number_input("Away form (0-1)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    home_advantage = c7.number_input("Home advantage (0-1)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

st.markdown("---")

left, right = st.columns(2)

with left:
    home_inputs = stat_input_block("home", "Home Team — Last 5 Matches")

with right:
    away_inputs = stat_input_block("away", "Away Team — Last 5 Matches")

if st.button("Predict Match Outcome", use_container_width=True):
    try:
        user_inputs = {
            "matchday": float(matchday),
            "hour": float(hour),
            "home_pre_position": float(home_pre_position),
            "away_pre_position": float(away_pre_position),
            "home_form": float(home_form),
            "away_form": float(away_form),
            "home_advantage": float(home_advantage),
            **home_inputs,
            **away_inputs,
        }

        feature_df = build_feature_ready_row(user_inputs)
        results = predict_from_features(feature_df)

        row = results.iloc[0]

        st.success(f"Predicted outcome: {row['prediction']}")

        p1, p2, p3 = st.columns(3)
        p1.metric("Home Win Probability", f"{row.get('prob_H', 0):.2%}")
        p2.metric("Draw Probability", f"{row.get('prob_D', 0):.2%}")
        p3.metric("Away Win Probability", f"{row.get('prob_A', 0):.2%}")

        st.subheader("Prediction Details")
        st.dataframe(results, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")