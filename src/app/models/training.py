"""
Training Models

This module contains shared models used across training-related functionality.
"""

from pydantic import BaseModel


class ImageTrainingData(BaseModel):
    """Model for image training data."""

    image_id: int
    feature_vector: list[float]
    eeg_valence: float
    eeg_arousal: float
