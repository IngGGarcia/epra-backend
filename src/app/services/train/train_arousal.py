"""
Arousal Training Service

This module provides functionality for training a Gaussian Process Regression (GPR) model
to predict arousal values from image features. The arousal dimension represents the
level of activation or intensity of an emotional response.

The module uses normalized EEG data to train a GPR model that can predict arousal
values for new images based on their feature vectors.

References:
    - Gaussian Process Regression for Emotion: Rasmussen, C. E., & Williams, C. K. I. (2006).
      Gaussian Processes for Machine Learning. MIT Press.
      https://doi.org/10.7551/mitpress/3206.001.0001

    - Arousal in Emotion Recognition: Bradley, M. M., & Lang, P. J. (1994). Measuring emotion:
      The Self-Assessment Manikin and the Semantic Differential. Journal of Behavior
      Therapy and Experimental Psychiatry, 25(1), 49-59.
      https://doi.org/10.1016/0005-7916(94)90063-9

    - Feature Extraction for Emotion: Soleymani, M., et al. (2017). A Survey of
      Multimodal Sentiment Analysis. Image and Vision Computing, 65, 3-14.
      https://doi.org/10.1016/j.imavis.2017.08.003
"""

import numpy as np

from src.app.models.training import ImageTrainingData
from src.modules.GPR.model import GaussianProcessRegressor

from .normalize_values import normalize_range


def train_arousal_model(
    training_data: list[ImageTrainingData],
) -> GaussianProcessRegressor:
    """
    Train a GPR model for arousal prediction using the provided training data.

    This function processes EEG arousal data and image feature vectors to create
    a predictive model. The arousal values are normalized to a standard range
    before training to ensure consistent model performance.

    Args:
        training_data (list[ImageTrainingData]): List of training data objects containing:
            - image_id: Unique identifier for the image
            - feature_vector: Extracted features from the image
            - eeg_arousal: Raw EEG arousal value
            - eeg_valence: Raw EEG valence value (not used in this function)

    Returns:
        GaussianProcessRegressor: Trained GPR model for arousal prediction

    Raises:
        ValueError: If training_data is empty or if feature vectors have inconsistent dimensions
        RuntimeError: If model training fails

    Examples:
        >>> training_data = [
        ...     ImageTrainingData(
        ...         image_id=1,
        ...         feature_vector=[0.1, 0.2, 0.3],
        ...         eeg_arousal=5.0,
        ...         eeg_valence=6.0
        ...     )
        ... ]
        >>> model = train_arousal_model(training_data)
        >>> isinstance(model, GaussianProcessRegressor)
        True
    """
    if not training_data:
        raise ValueError("Training data cannot be empty")

    # Extract features and arousal values
    X = [data.feature_vector for data in training_data]
    y = [data.eeg_arousal for data in training_data]

    # Validate feature vector dimensions
    if not all(len(x) == len(X[0]) for x in X):
        raise ValueError("All feature vectors must have the same dimension")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Normalize arousal values to standard range
    y = normalize_range(y)

    # Initialize and train the GPR model
    model = GaussianProcessRegressor()
    try:
        model.fit(X, y)
    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")

    return model
