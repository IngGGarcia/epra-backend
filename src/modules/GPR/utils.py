"""
Utility functions for the Gaussian Process Regression module.

This module provides helper functions for data preprocessing, model evaluation,
and visualization of GPR results.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def preprocess_data(X, y, test_size=0.2, random_state=None):
    """
    Preprocess and split the data into training and testing sets.

    Args:
        X: Input features (vectorized images)
        y: Target values (valence or arousal)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple containing (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(y_true, y_pred, y_std=None):
    """
    Evaluate the model performance using various metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values
        y_std: Standard deviation of predictions (optional)

    Returns:
        Dictionary containing evaluation metrics
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }

    if y_std is not None:
        metrics["mean_std"] = np.mean(y_std)

    return metrics


def normalize_features(X):
    """
    Normalize input features using z-score normalization.

    Args:
        X: Input features to normalize

    Returns:
        Tuple containing (normalized_features, mean, std)
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std
