"""
Image feature optimizer using PSO.

This module implements image feature optimization to maximize
arousal and valence in user responses. It uses Particle Swarm Optimization
to find the optimal combination of image features that best predict
user emotional responses.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .pso_optimizer import PSOOptimizer


class ImageFeatureOptimizer:
    """
    Image feature optimizer using PSO.

    This optimizer seeks to find the best combination of features
    that maximize arousal and valence in user responses. It uses
    correlation between weighted features and user responses as
    the optimization objective.
    """

    def __init__(
        self,
        image_features: pd.DataFrame,
        user_responses: pd.DataFrame,
        n_particles: int = 30,
        max_iter: int = 100,
    ):
        """
        Initialize the image feature optimizer.

        Args:
            image_features: DataFrame containing image features
            user_responses: DataFrame containing user responses (excitation and valence)
            n_particles: Number of particles in the swarm
            max_iter: Maximum number of iterations for optimization
        """
        self.image_features = image_features
        self.user_responses = user_responses
        self.n_particles = n_particles
        self.max_iter = max_iter

        # Prepare data
        self._prepare_data()

        # Initialize PSO optimizer
        self.pso = PSOOptimizer(
            n_particles=n_particles,
            n_dimensions=len(self.feature_columns),
            objective_function=self._objective_function,
            bounds=[(-1, 1) for _ in self.feature_columns],
            max_iter=max_iter,
        )

    def _prepare_data(self):
        """
        Prepare data for optimization.

        This method:
        1. Normalizes the feature data using StandardScaler
        2. Extracts numerical features
        3. Prepares excitation and valence response data
        """
        # Normalize features
        self.scaler = StandardScaler()
        self.feature_columns = self.image_features.select_dtypes(
            include=[np.number]
        ).columns
        self.X = self.scaler.fit_transform(self.image_features[self.feature_columns])

        # Prepare responses
        self.y_excitation = self.user_responses["excitation"].values
        self.y_valence = self.user_responses["valence"].values

    def _objective_function(self, weights: np.ndarray) -> float:
        """
        Objective function for optimization.

        This function calculates a combined score based on the correlation
        between weighted features and both excitation and valence responses.

        Args:
            weights: Weights for the features

        Returns:
            float: Combined excitation and valence score (average of correlations)
        """
        # Calculate weighted prediction
        weighted_features = np.dot(self.X, weights)

        # Calculate correlation with excitation and valence
        excitation_corr = np.corrcoef(weighted_features, self.y_excitation)[0, 1]
        valence_corr = np.corrcoef(weighted_features, self.y_valence)[0, 1]

        # Combine metrics (you can adjust weights as needed)
        return 0.5 * excitation_corr + 0.5 * valence_corr

    def optimize(self) -> pd.Series:
        """
        Run feature optimization.

        This method runs the PSO algorithm to find the optimal weights
        for each feature that maximize the objective function.

        Returns:
            pd.Series: Optimal weights for each feature
        """
        best_position, best_score = self.pso.optimize()

        # Create Series with results
        results = pd.Series(
            best_position, index=self.feature_columns, name="feature_weights"
        )

        return results

    def get_important_features(self, threshold: float = 0.1) -> pd.Series:
        """
        Get the most important features based on weights.

        This method identifies features whose absolute weights exceed
        the specified threshold, indicating their importance in
        predicting user responses.

        Args:
            threshold: Threshold to consider a feature as important

        Returns:
            pd.Series: Important features and their weights, sorted by absolute value
        """
        weights = self.optimize()
        return weights[abs(weights) > threshold].sort_values(ascending=False)
