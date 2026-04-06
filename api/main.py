from __future__ import annotations

from fastapi import FastAPI, HTTPException

from api.schemas import PredictionRequest, PredictionResponse
from api.service import predict_match

app = FastAPI(
    title="Football Match Prediction API",
    description="FastAPI backend for Belgian Jupiler Pro League match prediction.",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Simple health endpoint for checking whether the API is running.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """
    Predict match outcome probabilities from pre-match features.
    """
    try:
        result = predict_match(payload.model_dump())
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc