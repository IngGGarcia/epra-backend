"""
Prediction Service

This module provides core prediction functionality using trained GPR models.
"""

from typing import Tuple

import numpy as np

from src.modules.GPR.model import GaussianProcessRegressor
from src.modules.metrics.eeg_sam_heuristic import EmotionData, analyze_emotion_data


class PredictionService:
    """Service for making predictions using trained GPR models."""

    def __init__(
        self,
        arousal_model: GaussianProcessRegressor,
        valence_model: GaussianProcessRegressor,
        bias_correction: dict[str, float] | None = None,
    ):
        """
        Initialize the prediction service with trained models.

        Args:
            arousal_model: Trained GPR model for arousal prediction
            valence_model: Trained GPR model for valence prediction
            bias_correction: Dictionary containing bias correction factors
                           {'valence_bias': float, 'arousal_bias': float}
        """
        self.arousal_model = arousal_model
        self.valence_model = valence_model
        self.bias_correction = bias_correction or {
            "valence_bias": 0.0,
            "arousal_bias": 0.0,
        }

    def predict(self, feature_vector: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Predict valence and arousal values for a given feature vector.

        Args:
            feature_vector: Input feature vector of shape (1, n_features)

        Returns:
            Tuple containing (valence, valence_std, arousal, arousal_std)
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        arousal_pred, arousal_std = self.arousal_model.predict(feature_vector)
        valence_pred, valence_std = self.valence_model.predict(feature_vector)

        # Apply bias correction
        valence_pred = valence_pred + self.bias_correction["valence_bias"]
        arousal_pred = arousal_pred + self.bias_correction["arousal_bias"]

        # Ensure predictions stay within valid range [1, 9]
        valence_pred = np.clip(valence_pred, 1.0, 9.0)
        arousal_pred = np.clip(arousal_pred, 1.0, 9.0)

        return (
            float(valence_pred[0]),
            float(valence_std[0]),
            float(arousal_pred[0]),
            float(arousal_std[0]),
        )

    @classmethod
    def from_user_models(
        cls, user_id: int, historical_data: list[EmotionData] | None = None
    ) -> "PredictionService":
        """
        Create a PredictionService instance from a user's saved models.

        Args:
            user_id: ID of the user whose models to load
            historical_data: Optional list of historical EmotionData to calculate bias correction

        Returns:
            PredictionService instance with loaded models

        Raises:
            FileNotFoundError: If the user's models are not found
        """
        user_dir = f"storage/models/user_{user_id}"
        arousal_model = GaussianProcessRegressor.load(f"{user_dir}/gpr_arousal.joblib")
        valence_model = GaussianProcessRegressor.load(f"{user_dir}/gpr_valence.joblib")

        bias_correction = None
        if historical_data:
            # Calculate systematic bias from historical data
            stats = analyze_emotion_data(historical_data)

            # Calculate average bias direction
            valence_signs = [
                np.sign(data.sam_valence - data.eeg_valence) for data in historical_data
            ]
            arousal_signs = [
                np.sign(data.sam_arousal - data.eeg_arousal) for data in historical_data
            ]

            valence_bias = np.mean(valence_signs) * stats["mean_valence_deviation"]
            arousal_bias = np.mean(arousal_signs) * stats["mean_arousal_deviation"]

            bias_correction = {
                "valence_bias": float(valence_bias),
                "arousal_bias": float(arousal_bias),
            }

        return cls(arousal_model, valence_model, bias_correction)
