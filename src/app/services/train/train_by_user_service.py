"""
Train by User Service

This service provides functionality for training a model by user.

References:
    - User-Specific Emotion Recognition: Koelstra, S., et al. (2011). DEAP: A Database for Emotion
      Analysis using Physiological Signals. IEEE Transactions on Affective Computing,
      3(1), 18-31. https://doi.org/10.1109/T-AFFC.2011.15

    - Model Persistence Best Practices: Pedregosa, F., et al. (2011). Scikit-learn: Machine
      Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
      https://doi.org/10.5555/1953048.2078195

    - Database Integration: SQLModel Documentation. https://sqlmodel.tiangolo.com/
"""

import json
import os

from sqlmodel import Session, select

from src.app.models.image_classification import ImageClassification
from src.app.models.image_evaluation import ImageEvaluation
from src.app.models.training import ImageTrainingData
from src.modules.GPR.model import GaussianProcessRegressor

from .train_arousal import train_arousal_model
from .train_valence import train_valence_model


def save_model(model: GaussianProcessRegressor, user_id: int, model_type: str) -> str:
    """
    Save a trained model to disk.

    Args:
        model: The trained GPR model
        user_id: The ID of the user the model was trained for
        model_type: Type of model ('arousal' or 'valence')

    Returns:
        Path where the model was saved
    """
    # Create user directory
    user_dir = os.path.join("storage", "models", f"user_{user_id}")
    os.makedirs(user_dir, exist_ok=True)

    # Save model with simple name
    filename = f"gpr_{model_type}.joblib"
    save_path = os.path.join(user_dir, filename)

    # Save the model
    model.save(save_path)

    return save_path


def get_training_data(user_id: int, session: Session) -> list[ImageTrainingData]:
    """
    Get training data for a specific user.
    """
    # Get training data
    statement = (
        select(
            ImageEvaluation.image_id,
            ImageClassification.feature_vector,
            ImageEvaluation.eeg_valence,
            ImageEvaluation.eeg_arousal,
        )
        .join(
            ImageClassification,
            ImageEvaluation.image_id == ImageClassification.image_id,
        )
        .where(
            ImageEvaluation.user_id == user_id,
        )
    )
    results = session.exec(statement).all()

    # Validate results
    if not results:
        raise ValueError(f"No training data found for user {user_id}")

    training_data = [
        ImageTrainingData(
            image_id=id,
            feature_vector=json.loads(feature_vector),
            eeg_valence=eeg_valence,
            eeg_arousal=eeg_arousal,
        )
        for id, feature_vector, eeg_valence, eeg_arousal in results
    ]

    return training_data


def train_by_user_service(
    user_id: int,
    session: Session,
) -> dict[str, GaussianProcessRegressor]:
    """
    Train arousal and valence models for a specific user.

    Args:
        user_id: The ID of the user to train models for
        session: Database session

    Returns:
        ModelsTrained containing the trained models
    """
    training_data = get_training_data(user_id, session)

    # Train both models
    arousal_model = train_arousal_model(training_data)
    valence_model = train_valence_model(training_data)

    # Save models
    save_model(arousal_model, user_id, "arousal")
    save_model(valence_model, user_id, "valence")

    return {
        "arousal_model": arousal_model,
        "valence_model": valence_model,
    }
