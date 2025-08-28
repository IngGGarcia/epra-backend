"""
Tests for the EEG-specific PSO optimizer.

This module contains tests to verify the functionality of the EEGPSOOptimizer
class, including optimization, prediction, and feature importance calculation.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error

from src.modules.PSO.eeg_optimizer import EEGPSOOptimizer


@pytest.fixture
def sample_data():
    """
    Crea datos sintéticos con relación lineal conocida entre features y targets.
    """
    np.random.seed(42)
    n_samples = 20
    n_features = 10
    # Features aleatorias
    image_features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    # Pesos verdaderos para valence y arousal
    true_weights_valence = np.random.uniform(-2, 2, n_features)
    true_weights_arousal = np.random.uniform(-2, 2, n_features)
    # Targets como combinación lineal + ruido
    valence_eeg = image_features.values @ true_weights_valence + np.random.normal(
        0, 0.5, n_samples
    )
    arousal_eeg = image_features.values @ true_weights_arousal + np.random.normal(
        0, 0.5, n_samples
    )
    eeg_responses = pd.DataFrame(
        {
            "valence_eeg": valence_eeg,
            "arousal_eeg": arousal_eeg,
        }
    )
    return image_features, eeg_responses


def test_eeg_optimizer_initialization(sample_data):
    """
    Test that the EEGPSOOptimizer initializes correctly.
    """
    image_features, eeg_responses = sample_data
    optimizer = EEGPSOOptimizer(
        image_features=image_features,
        eeg_responses=eeg_responses,
        n_particles=20,
        max_iter=50,
    )

    assert optimizer.image_features.shape == (20, 10)
    assert optimizer.eeg_responses.shape == (20, 2)
    assert optimizer.valence_pso is not None
    assert optimizer.arousal_pso is not None


def test_objective_function(sample_data):
    """
    Test that the objective function calculates errors correctly.
    """
    image_features, eeg_responses = sample_data
    optimizer = EEGPSOOptimizer(
        image_features=image_features,
        eeg_responses=eeg_responses,
        n_particles=20,
        max_iter=50,
    )

    # Test with random weights
    weights = np.random.randn(10)

    # Calculate error for valence
    valence_error = optimizer._objective_function(weights, "valence")
    weighted_features = np.dot(image_features, weights)
    expected_valence_error = mean_squared_error(
        weighted_features, eeg_responses["valence_eeg"]
    )
    assert np.isclose(valence_error, expected_valence_error)

    # Calculate error for arousal
    arousal_error = optimizer._objective_function(weights, "arousal")
    expected_arousal_error = mean_squared_error(
        weighted_features, eeg_responses["arousal_eeg"]
    )
    assert np.isclose(arousal_error, expected_arousal_error)


def test_optimization(sample_data):
    """
    Test that the optimization process returns expected results.
    """
    image_features, eeg_responses = sample_data
    optimizer = EEGPSOOptimizer(
        image_features=image_features,
        eeg_responses=eeg_responses,
        n_particles=20,
        max_iter=50,
    )

    results = optimizer.optimize()

    # Check that results contain expected keys
    assert "valence_weights" in results
    assert "arousal_weights" in results
    assert "valence_score" in results
    assert "arousal_score" in results
    assert "valence_importance" in results
    assert "arousal_importance" in results
    assert "convergence_curves" in results

    # Check that weights have correct shape
    assert len(results["valence_weights"]) == 10
    assert len(results["arousal_weights"]) == 10

    # Check that scores are positive
    assert results["valence_score"] > 0
    assert results["arousal_score"] > 0

    # Check that importance dictionaries have correct length
    assert len(results["valence_importance"]) == 10
    assert len(results["arousal_importance"]) == 10


def test_prediction(sample_data):
    """
    Test that the prediction function works correctly.
    """
    image_features, eeg_responses = sample_data
    optimizer = EEGPSOOptimizer(
        image_features=image_features,
        eeg_responses=eeg_responses,
        n_particles=20,
        max_iter=50,
    )

    # Get optimized weights
    results = optimizer.optimize()
    weights = {
        "valence_weights": results["valence_weights"],
        "arousal_weights": results["arousal_weights"],
    }

    # Make predictions
    predictions = optimizer.predict(weights, image_features)

    # Check predictions
    assert predictions.shape == (20, 2)
    assert "predicted_valence" in predictions.columns
    assert "predicted_arousal" in predictions.columns
    assert not predictions.isna().any().any()


def test_convergence(sample_data):
    """
    Test that the optimization process shows convergence.
    """
    image_features, eeg_responses = sample_data
    optimizer = EEGPSOOptimizer(
        image_features=image_features,
        eeg_responses=eeg_responses,
        n_particles=20,
        max_iter=50,
    )

    results = optimizer.optimize()
    convergence_curves = results["convergence_curves"]

    # Check that convergence curves exist and have correct length
    assert len(convergence_curves["valence"]) > 0
    assert len(convergence_curves["arousal"]) > 0

    # Check that at some point the error improved compared to the start
    assert min(convergence_curves["valence"]) <= convergence_curves["valence"][0]
    assert min(convergence_curves["arousal"]) <= convergence_curves["arousal"][0]
