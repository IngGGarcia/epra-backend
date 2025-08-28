"""
Get Best Images Service

This service provides functionality for getting the best images based on predicted valence and arousal values.
"""

import json
import random

import numpy as np
from sqlmodel import Session, select

from src.app.models.image_classification import ImageClassification
from src.app.models.image_evaluation import ImageEvaluation
from src.app.services.predict.predict_service import PredictionService
from src.app.services.train import train_by_user_service
from src.modules.metrics.eeg_sam_heuristic import EmotionData


def get_best_images(
    session: Session,
    user_id: int,
    limit: int = 10,
) -> list[int]:
    """
    Get the best images based on predicted valence and arousal values.
    Images are sorted first by valence (highest first) and then by arousal.
    Predictions include bias correction based on user's historical data.
    Excludes images that have already been evaluated by the user.

    Args:
        session: Database session
        user_id: ID of the user whose models to use for prediction
        limit: Maximum number of images to return

    Returns:
        List of image IDs sorted by valence (highest first) and then by arousal
    """

    # Train models if not already trained
    train_by_user_service(user_id, session)

    # Get historical data for bias correction
    # Use all available evaluations for more accurate bias correction
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

    # Get prediction service with bias correction
    prediction_service = PredictionService.from_user_models(
        user_id=user_id, historical_data=historical_data
    )

    # Get all image feature vectors, excluding already evaluated images
    statement = select(
        ImageClassification.image_id, ImageClassification.feature_vector
    ).where(
        ImageClassification.image_id.not_in(
            select(ImageEvaluation.image_id).where(ImageEvaluation.user_id == user_id)
        )
    )
    results = session.exec(statement).all()

    if not results:
        raise ValueError("No new images available for evaluation")

    # Make predictions for all images
    predictions = []
    for image_id, feature_vector in results:
        feature_vector = np.array(json.loads(feature_vector))
        valence, valence_std, arousal, arousal_std = prediction_service.predict(
            feature_vector
        )

        predictions.append((image_id, valence, arousal))

    # Sort predictions by valence (descending) and then by arousal (descending)
    sorted_predictions = sorted(predictions, key=lambda x: (x[1], x[2]), reverse=True)

    # Select n-1 best images and the worst image
    best_images = [image_id for image_id, _, _ in sorted_predictions[: limit - 1]]
    worst_image = sorted_predictions[-1][0]
    best_images.append(worst_image)

    # Shuffle the list to ensure the worst image is not always at the end
    random.shuffle(best_images)

    return best_images
