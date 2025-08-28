"""
Image feature optimizer specific to a user using PSO.

This module implements image feature optimization to maximize
arousal and valence in the responses of a specific user. It uses
Particle Swarm Optimization to find the optimal combination of
image features that best predict individual user responses.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .eeg_optimizer import EEGPSOOptimizer
from .pso_optimizer import PSOOptimizer


class UserImageOptimizer:
    """
    Image feature optimizer specific to a user.

    This optimizer seeks to find the best combination of features
    that maximize correlation with the user's subjective responses
    and EEG measurements. It tracks convergence during optimization
    and provides detailed results about feature importance.
    """

    def __init__(
        self,
        image_features: pd.DataFrame,
        user_responses: pd.DataFrame,
        user_id: int,
        n_particles: int = 50,
        max_iter: int = 200,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 2.0,
        c2: float = 1.0,
        velocity_clamp: float = 0.5,
        convergence_threshold: float = 1e-6,
        convergence_window: int = 10,
    ):
        """
        Initialize the image feature optimizer for a user.

        Args:
            image_features: DataFrame containing image features
            user_responses: DataFrame containing user responses (valence, arousal, heuristic, eeg_valence, eeg_arousal)
            user_id: ID of the user to optimize for
            n_particles: Number of particles in the swarm
            max_iter: Maximum number of iterations for optimization
            w_start: Initial inertia weight
            w_end: Final inertia weight
            c1: Cognitive learning factor
            c2: Social learning factor
            velocity_clamp: Maximum velocity magnitude
            convergence_threshold: Threshold for convergence detection
            convergence_window: Number of iterations to check for convergence
        """
        self.image_features = image_features
        self.user_responses = user_responses
        self.user_id = user_id
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.velocity_clamp = velocity_clamp
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window

        # Prepare data
        self._prepare_data()

        # Initialize PSO optimizers
        self.pso = PSOOptimizer(
            n_particles=n_particles,
            n_dimensions=len(self.feature_columns),
            objective_function=self._objective_function,
            bounds=[(-1, 1) for _ in self.feature_columns],
            w_start=w_start,
            w_end=w_end,
            c1=c1,
            c2=c2,
            max_iter=max_iter,
            velocity_clamp=velocity_clamp,
            convergence_threshold=convergence_threshold,
            convergence_window=convergence_window,
        )

        # Initialize EEG-specific optimizer
        self.eeg_pso = EEGPSOOptimizer(
            image_features=image_features,
            eeg_responses=user_responses,
            n_particles=n_particles,
            max_iter=max_iter,
            w_start=w_start,
            w_end=w_end,
            c1=c1,
            c2=c2,
            velocity_clamp=velocity_clamp,
            convergence_threshold=convergence_threshold,
            convergence_window=convergence_window,
        )

        # List to store convergence curve
        self.pso.convergence_curve = []

    def _prepare_data(self):
        """
        Prepare data for optimization.

        This method:
        1. Normalizes the feature data using StandardScaler
        2. Extracts numerical features
        3. Prepares user response data (valence, arousal, heuristic, eeg_valence, eeg_arousal)
        """
        # Normalize features
        self.scaler = StandardScaler()
        self.feature_columns = self.image_features.select_dtypes(
            include=[np.number]
        ).columns
        self.X = self.scaler.fit_transform(self.image_features[self.feature_columns])

        # Prepare user responses
        self.y_valence = self.user_responses["valence"].values
        self.y_arousal = self.user_responses["arousal"].values
        self.y_heurist = self.user_responses["metric_heurist"].values
        self.y_eeg_valence = self.user_responses["valence_eeg"].values
        self.y_eeg_arousal = self.user_responses["arousal_eeg"].values

    def _objective_function(self, weights: np.ndarray) -> float:
        """
        Objective function for optimization.

        This function calculates a combined score based on the correlation
        between weighted features and user responses (valence, arousal, heuristic).
        It also tracks the convergence of the optimization process.

        Args:
            weights: Weights for the features

        Returns:
            float: Combined correlation score weighted by importance
        """
        # Calculate weighted prediction
        weighted_features = np.dot(self.X, weights)

        # Calculate correlations
        valence_corr = np.corrcoef(weighted_features, self.y_valence)[0, 1]
        arousal_corr = np.corrcoef(weighted_features, self.y_arousal)[0, 1]
        heurist_corr = np.corrcoef(weighted_features, self.y_heurist)[0, 1]
        eeg_valence_corr = np.corrcoef(weighted_features, self.y_eeg_valence)[0, 1]
        eeg_arousal_corr = np.corrcoef(weighted_features, self.y_eeg_arousal)[0, 1]

        # Save score in convergence curve
        self.pso.convergence_curve.append(
            float(valence_corr + arousal_corr + eeg_valence_corr + eeg_arousal_corr)
        )

        # Combine metrics with adjusted weights
        return (
            0.25 * valence_corr
            + 0.25 * arousal_corr
            + 0.1 * heurist_corr
            + 0.2 * eeg_valence_corr
            + 0.2 * eeg_arousal_corr
        )

    def optimize(self) -> dict[str, float | dict[str, float] | int]:
        """
        Run feature optimization for the user.

        This method runs the PSO algorithm to find the optimal weights
        for each feature that maximize the objective function. It returns
        a comprehensive dictionary of results including feature weights,
        important features, and optimization metrics.

        Returns:
            dict[str, float | dict[str, float] | int]: Dictionary containing:
                - user_id: ID of the optimized user
                - best_score: Best objective function score achieved
                - feature_weights: Dictionary of all feature weights
                - important_features: Dictionary of features with weights > 0.1
                - n_responses: Number of user responses used
                - eeg_optimization: Results from EEG-specific optimization
        """
        # Run standard PSO optimization
        best_position, best_score = self.pso.optimize()

        # Run EEG-specific optimization
        eeg_results = self.eeg_pso.optimize()

        # Create Series with results
        feature_weights = pd.Series(
            best_position, index=self.feature_columns, name="feature_weights"
        )

        # Get important features
        important_features = feature_weights[abs(feature_weights) > 0.1].sort_values(
            ascending=False
        )

        return {
            "user_id": self.user_id,
            "best_score": float(best_score),
            "feature_weights": feature_weights.to_dict(),
            "important_features": important_features.to_dict(),
            "n_responses": len(self.user_responses),
            "eeg_optimization": eeg_results,
        }
