"""
EEG-specific PSO optimizer for predicting EEG responses.

This module implements a specialized PSO optimizer that uses EEG responses
(valence and arousal) as target vectors for optimization. It includes
separate optimization for valence and arousal predictions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .pso_optimizer import PSOOptimizer


class EEGPSOOptimizer:
    """
    EEG-specific PSO optimizer for predicting EEG responses.

    This optimizer uses Particle Swarm Optimization to find the optimal
    combination of image features that best predict EEG-derived valence
    and arousal responses. It maintains separate optimization processes
    for valence and arousal predictions.
    """

    def __init__(
        self,
        image_features: pd.DataFrame,
        eeg_responses: pd.DataFrame,
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
        Initialize the EEG-specific PSO optimizer.

        Args:
            image_features: DataFrame containing image features
            eeg_responses: DataFrame containing EEG responses (valence_eeg, arousal_eeg)
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
        self.eeg_responses = eeg_responses
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.velocity_clamp = velocity_clamp
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window

        # Initialize PSO optimizers for valence and arousal
        pso_params = {
            "n_particles": n_particles,
            "n_dimensions": len(image_features.columns),
            "bounds": [(-1, 1) for _ in image_features.columns],
            "w_start": w_start,
            "w_end": w_end,
            "c1": c1,
            "c2": c2,
            "max_iter": max_iter,
            "velocity_clamp": velocity_clamp,
            "convergence_threshold": convergence_threshold,
            "convergence_window": convergence_window,
        }

        # Create separate optimizers for valence and arousal
        self.valence_pso = PSOOptimizer(
            objective_function=lambda w: self._objective_function(w, "valence"),
            **pso_params,
        )
        self.arousal_pso = PSOOptimizer(
            objective_function=lambda w: self._objective_function(w, "arousal"),
            **pso_params,
        )

    def _objective_function(self, weights: np.ndarray, target: str) -> float:
        """
        Objective function for PSO optimization.

        This function calculates the error between predicted and actual
        EEG responses for all images in the training set.

        Args:
            weights: Vector of weights to evaluate
            target: Either 'valence' or 'arousal'

        Returns:
            float: Error for the specified target
        """
        # Calculate weighted predictions for all images at once
        weighted_features = np.dot(self.image_features, weights)

        # Calculate error for the specified target
        target_column = f"{target}_eeg"
        error = mean_squared_error(weighted_features, self.eeg_responses[target_column])

        return error

    def optimize(self) -> dict:
        """
        Run the PSO optimization process for both valence and arousal.

        Returns:
            dict: Optimization results including:
                - valence_weights: Optimal weights for valence prediction
                - arousal_weights: Optimal weights for arousal prediction
                - valence_score: Best error achieved for valence
                - arousal_score: Best error achieved for arousal
                - valence_importance: Dictionary of feature importance for valence
                - arousal_importance: Dictionary of feature importance for arousal
                - convergence_curves: Dictionary with convergence curves for both targets
        """
        # Run PSO optimization for both targets
        valence_position, valence_score = self.valence_pso.optimize()
        arousal_position, arousal_score = self.arousal_pso.optimize()

        # Calculate feature importance for both targets
        valence_importance = {
            feature: abs(weight)
            for feature, weight in zip(self.image_features.columns, valence_position)
        }
        arousal_importance = {
            feature: abs(weight)
            for feature, weight in zip(self.image_features.columns, arousal_position)
        }

        # Sort features by importance
        valence_importance = dict(
            sorted(valence_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        arousal_importance = dict(
            sorted(arousal_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return {
            "valence_weights": dict(zip(self.image_features.columns, valence_position)),
            "arousal_weights": dict(zip(self.image_features.columns, arousal_position)),
            "valence_score": float(valence_score),
            "arousal_score": float(arousal_score),
            "valence_importance": valence_importance,
            "arousal_importance": arousal_importance,
            "convergence_curves": {
                "valence": self.valence_pso.convergence_curve,
                "arousal": self.arousal_pso.convergence_curve,
            },
        }

    def predict(self, weights: dict[str, dict], features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict EEG responses using the optimized weights.

        Args:
            weights: Dictionary containing 'valence_weights' and 'arousal_weights'
            features: DataFrame of image features to predict for

        Returns:
            DataFrame: Predicted EEG responses (valence and arousal)
        """
        # Convert weights to arrays in the same order as features
        valence_weights = np.array(
            [weights["valence_weights"][feature] for feature in features.columns]
        )
        arousal_weights = np.array(
            [weights["arousal_weights"][feature] for feature in features.columns]
        )

        # Calculate predictions
        valence_predictions = np.dot(features, valence_weights)
        arousal_predictions = np.dot(features, arousal_weights)

        # Create DataFrame with predictions
        return pd.DataFrame(
            {
                "predicted_valence": valence_predictions,
                "predicted_arousal": arousal_predictions,
            }
        )
