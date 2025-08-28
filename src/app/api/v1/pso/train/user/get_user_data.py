"""
This module contains the FastAPI endpoints to interact with the image feature optimizer.
"""

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from src.app.models import EEGImageEvaluation, ImageClassification


def rescale_predictions(preds, new_min=2, new_max=8):
    """
    Rescale the predictions to a new range.
    """
    p_min, p_max = preds.min(), preds.max()
    return ((preds - p_min) / (p_max - p_min)) * (new_max - new_min) + new_min


def get_user_data(user_id: int, session: Session) -> pd.DataFrame:
    """
    Get the data for a specific user and return it as a pandas DataFrame.

    Args:
        user_id (int): The ID of the user to get data for
        session (Session): SQLModel database session

    Returns:
        pd.DataFrame: DataFrame containing the user's data with feature vectors as numpy arrays
    """
    statement = (
        select(
            EEGImageEvaluation.valence_eeg,
            EEGImageEvaluation.arousal_eeg,
            ImageClassification.feature_vector,
        )
        .where(EEGImageEvaluation.user_id == user_id)
        .join(
            ImageClassification,
            EEGImageEvaluation.image_id == ImageClassification.image_id,
        )
    )

    results = session.exec(statement).all()

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Convert feature_vector from string to numpy array
    df["feature_vector"] = df["feature_vector"].apply(lambda x: np.array(eval(x)))

    # Rescale the predictions
    df["valence_eeg"] = rescale_predictions(df["valence_eeg"])
    df["arousal_eeg"] = rescale_predictions(df["arousal_eeg"])

    return df
