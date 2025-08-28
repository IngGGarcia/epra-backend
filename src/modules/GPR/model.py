"""
Main implementation of the Gaussian Process Regression model.

This module contains the core GPR model class that handles training, prediction,
and model management.
"""

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .config import (
    DEFAULT_KERNEL,
    DEFAULT_LENGTH_SCALE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_NOISE_VARIANCE,
    DEFAULT_OPTIMIZER,
    DEFAULT_SIGNAL_VARIANCE,
)
from .kernels import MaternKernel, RBFKernel


class GaussianProcessRegressor(BaseEstimator, RegressorMixin):
    """
    Gaussian Process Regression model for predicting emotional responses from images.

    This implementation uses a probabilistic approach to predict valence or arousal
    values from vectorized image representations, providing both predictions and
    uncertainty estimates.
    """

    def __init__(
        self,
        kernel=DEFAULT_KERNEL,
        length_scale=DEFAULT_LENGTH_SCALE,
        signal_variance=DEFAULT_SIGNAL_VARIANCE,
        noise_variance=DEFAULT_NOISE_VARIANCE,
        max_iter=DEFAULT_MAX_ITERATIONS,
        optimizer=DEFAULT_OPTIMIZER,
        random_state=None,
    ):
        """
        Initialize the Gaussian Process Regressor.

        Args:
            kernel: Type of kernel to use ('rbf', 'matern', etc.)
            length_scale: Length scale parameter for the kernel
            signal_variance: Signal variance parameter
            noise_variance: Noise variance parameter
            max_iter: Maximum number of iterations for optimization
            optimizer: Optimization algorithm to use
            random_state: Random seed for reproducibility
        """
        self.kernel = self._initialize_kernel(kernel, length_scale, signal_variance)
        self.noise_variance = noise_variance
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.random_state = random_state

        # Model parameters
        self.X_train = None
        self.y_train = None
        self.alpha = None
        self.L = None

    def _initialize_kernel(self, kernel_type, length_scale, signal_variance):
        """
        Initialize the specified kernel type.

        Args:
            kernel_type: Type of kernel to initialize
            length_scale: Length scale parameter
            signal_variance: Signal variance parameter

        Returns:
            Initialized kernel object
        """
        if kernel_type == "rbf":
            return RBFKernel(length_scale=length_scale, signal_variance=signal_variance)
        if kernel_type == "matern":
            return MaternKernel(
                length_scale=length_scale, signal_variance=signal_variance
            )
        # Add more kernel types here
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    def fit(self, X, y):
        """
        Fit the Gaussian Process model to the training data.

        Args:
            X: Training features (vectorized images)
            y: Training targets (valence or arousal values)

        Returns:
            self: The fitted model
        """
        self.X_train = X
        self.y_train = y

        # Compute kernel matrix
        K = self.kernel(X, X)
        K += self.noise_variance * np.eye(len(X))

        # Compute Cholesky decomposition
        self.L = np.linalg.cholesky(K)

        # Solve for alpha
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))

        return self

    def predict(self, X, return_std=True):
        """
        Predict target values for new data points.

        Args:
            X: Test features (vectorized images)
            return_std: Whether to return standard deviations

        Returns:
            Tuple of (predictions, standard_deviations)
        """
        # Compute kernel between test and training points
        K_star = self.kernel(X, self.X_train)

        # Compute predictions
        f_star = K_star @ self.alpha

        if not return_std:
            return f_star, None

        # Compute standard deviations
        v = np.linalg.solve(self.L, K_star.T)
        var_star = self.kernel(X, X) - v.T @ v
        std_star = np.sqrt(np.diag(var_star))

        return f_star, std_star

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep: If True, will return the parameters for this estimator and
                 contained subobjects that are estimators.

        Returns:
            Parameter names mapped to their values
        """
        return {
            "kernel": self.kernel,
            "noise_variance": self.noise_variance,
            "max_iter": self.max_iter,
            "optimizer": self.optimizer,
            "random_state": self.random_state,
        }

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Args:
            **parameters: Parameter names mapped to their values

        Returns:
            self: The estimator instance
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path where to save the model
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "GaussianProcessRegressor":
        """
        Load a model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded model instance
        """
        return joblib.load(path)
