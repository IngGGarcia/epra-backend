"""
This module contains the stats API routes.
"""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Path
import numpy as np
from pydantic import BaseModel
from sqlmodel import Session, distinct, func, select

from src.app.db import get_session
from src.app.models.image_classification import ImageClassification
from src.app.models.image_evaluation import ImageEvaluation
from src.app.services.predict.predict_service import PredictionService
from src.app.services.train import train_by_user_service
from src.modules.metrics.eeg_sam_heuristic import EmotionData, analyze_emotion_data

router = APIRouter(prefix="/stats")


class Stats(BaseModel):
    user_id: int
    evaluation_quantity: int


class EmotionStats(BaseModel):
    mean_valence_deviation: float
    mean_arousal_deviation: float
    systematic_bias: bool
    valence_deviations: list[float]
    arousal_deviations: list[float]


class EvaluationDetail(BaseModel):
    image_id: int
    sam_valence: float
    eeg_valence: float
    sam_arousal: float
    eeg_arousal: float
    valence_deviation: float
    arousal_deviation: float
    predicted_valence: float | None = None
    predicted_arousal: float | None = None


class UserStats(BaseModel):
    user_id: int
    evaluation_quantity: int
    emotion_stats: EmotionStats
    evaluations: list[EvaluationDetail]


@router.get("/")
async def get_stats(
    session: Annotated[Session, Depends(get_session)],
) -> list[Stats]:
    """
    Get the stats for the user.
    """

    # Get all image evaluations
    statement = select(
        ImageEvaluation.user_id,
        func.count(distinct(ImageEvaluation.evaluation_id)),
    ).group_by(ImageEvaluation.user_id)
    results = session.exec(statement).all()

    return [
        Stats(user_id=user_id, evaluation_quantity=count) for user_id, count in results
    ]


@router.get("/{user_id}")
async def get_stats_for_user(
    session: Annotated[Session, Depends(get_session)],
    user_id: Annotated[int, Path(..., description="The ID of the user")],
) -> UserStats:
    """
    Get the stats for the user including SAM-EEG deviation analysis and detailed evaluation information.
    If models don't exist, they will be automatically trained.
    """
    # Get all evaluations for the user
    statement = select(ImageEvaluation).where(ImageEvaluation.user_id == user_id)
    results = session.exec(statement).all()

    # Convert evaluations to EmotionData objects
    emotion_data_list = [
        EmotionData(
            sam_valence=eval.sam_valence,
            eeg_valence=eval.eeg_valence,
            sam_arousal=eval.sam_arousal,
            eeg_arousal=eval.eeg_arousal,
        )
        for eval in results
    ]

    # Calculate emotion statistics
    emotion_stats = analyze_emotion_data(emotion_data_list)

    # Get prediction service with bias correction
    try:
        prediction_service = PredictionService.from_user_models(
            user_id=user_id, historical_data=emotion_data_list
        )
    except FileNotFoundError:
        # If models don't exist, train them
        train_by_user_service(user_id, session)
        # Try to get prediction service again
        prediction_service = PredictionService.from_user_models(
            user_id=user_id, historical_data=emotion_data_list
        )

    # Create detailed evaluation list
    evaluations = []
    for i, eval in enumerate(results):
        # Get feature vector for prediction
        feature_vector = None
        if prediction_service:
            statement = select(ImageClassification.feature_vector).where(
                ImageClassification.image_id == eval.image_id
            )
            result = session.exec(statement).first()
            if result:  # Check if result exists
                try:
                    # Convert string representation to numpy array using json.loads
                    feature_vector = np.array(json.loads(result))
                except (ValueError, json.JSONDecodeError):
                    feature_vector = None

        # Calculate deviations
        valence_deviation = abs(eval.sam_valence - eval.eeg_valence)
        arousal_deviation = abs(eval.sam_arousal - eval.eeg_arousal)

        # Get predictions if possible
        predicted_valence = None
        predicted_arousal = None
        if prediction_service and feature_vector is not None:
            try:
                valence, _, arousal, _ = prediction_service.predict(feature_vector)
                predicted_valence = float(valence)
                predicted_arousal = float(arousal)
            except Exception:
                # If prediction fails, keep predictions as None
                pass

        evaluations.append(
            EvaluationDetail(
                image_id=eval.image_id,
                sam_valence=eval.sam_valence,
                eeg_valence=eval.eeg_valence,
                sam_arousal=eval.sam_arousal,
                eeg_arousal=eval.eeg_arousal,
                valence_deviation=valence_deviation,
                arousal_deviation=arousal_deviation,
                predicted_valence=predicted_valence,
                predicted_arousal=predicted_arousal,
            )
        )

    return UserStats(
        user_id=user_id,
        evaluation_quantity=len(results),
        emotion_stats=EmotionStats(
            mean_valence_deviation=emotion_stats["mean_valence_deviation"],
            mean_arousal_deviation=emotion_stats["mean_arousal_deviation"],
            systematic_bias=emotion_stats["systematic_bias"],
            valence_deviations=emotion_stats["valence_deviations"],
            arousal_deviations=emotion_stats["arousal_deviations"],
        ),
        evaluations=evaluations,
    )
