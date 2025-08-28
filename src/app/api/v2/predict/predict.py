"""
Prediction Endpoint

This module provides the endpoint for making predictions using trained GPR models.
"""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
import numpy as np
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from src.app.db import get_session
from src.app.models.image_classification import ImageClassification
from src.app.models.image_evaluation import ImageEvaluation
from src.app.services.predict.predict_service import PredictionService
from src.modules.metrics.eeg_sam_heuristic import EmotionData

router = APIRouter()


class PredictionResult(BaseModel):
    """Model for prediction results."""

    valence: float = Field(..., description="Predicted valence value")
    valence_std: float = Field(
        ..., description="Standard deviation of valence prediction"
    )
    arousal: float = Field(..., description="Predicted arousal value")
    arousal_std: float = Field(
        ..., description="Standard deviation of arousal prediction"
    )


@router.get("/")
async def predict(
    image_id: Annotated[int, Query()],
    user_id: Annotated[int, Query()],
    session: Annotated[Session, Depends(get_session)],
) -> PredictionResult:
    """
    Predict valence and arousal values for a given image using a user's trained models.
    Predictions include bias correction based on all available historical data.

    Args:
        image_id: The ID of the image to predict for
        user_id: The ID of the user whose models to use
        session: Database session

    Returns:
        PredictionResult containing the predicted values and their standard deviations

    Raises:
        HTTPException: If the image is not found or if the user's models are not found
    """
    # Get image feature vector
    statement = select(ImageClassification.feature_vector).where(
        ImageClassification.image_id == image_id
    )
    result = session.exec(statement).first()

    if not result:
        raise HTTPException(
            status_code=404, detail=f"Image with ID {image_id} not found"
        )

    feature_vector = np.array(json.loads(result))

    # Get all historical data for bias correction
    statement = (
        select(ImageEvaluation)
        .where(ImageEvaluation.user_id == user_id)
        .order_by(ImageEvaluation.created_at.desc())
    )  # Get all evaluations, most recent first
    historical_results = session.exec(statement).all()

    historical_data = (
        [
            EmotionData(
                sam_valence=eval.sam_valence,
                eeg_valence=eval.eeg_valence,
                sam_arousal=eval.sam_arousal,
                eeg_arousal=eval.eeg_arousal,
            )
            for eval in historical_results
        ]
        if historical_results
        else None
    )

    try:
        prediction_service = PredictionService.from_user_models(
            user_id=user_id, historical_data=historical_data
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Models for user {user_id} not found. Please train the models first.",
        ) from None

    # Make predictions
    valence, valence_std, arousal, arousal_std = prediction_service.predict(
        feature_vector
    )

    return PredictionResult(
        valence=valence,
        valence_std=valence_std,
        arousal=arousal,
        arousal_std=arousal_std,
    )
