"""
Tests for the Gaussian Process Regression model.

This module contains unit tests for the GPR model implementation,
including tests for initialization, fitting, and prediction.
"""

import numpy as np
import pytest

from src.modules.GPR.kernels import (
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
)
from src.modules.GPR.model import GaussianProcessRegressor


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(10, 5)  # 10 samples, 5 features
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(10)  # Simple function with noise
    return X, y


def test_model_initialization():
    """Test model initialization with different parameters."""
    # Test default initialization
    model = GaussianProcessRegressor()
    assert model.kernel is not None
    assert model.noise_variance > 0

    # Test custom initialization
    model = GaussianProcessRegressor(
        kernel="rbf", length_scale=2.0, signal_variance=1.5, noise_variance=0.2
    )
    assert isinstance(model.kernel, RBFKernel)
    assert model.noise_variance == 0.2


def test_model_fit(sample_data):
    """Test model fitting."""
    X, y = sample_data
    model = GaussianProcessRegressor()
    model.fit(X, y)

    assert model.X_train is not None
    assert model.y_train is not None
    assert model.alpha is not None
    assert model.L is not None


def test_model_predict(sample_data):
    """Test model prediction."""
    X, y = sample_data
    model = GaussianProcessRegressor()
    model.fit(X, y)

    # Test prediction
    X_test = np.random.randn(5, 5)
    y_pred, y_std = model.predict(X_test)

    assert y_pred.shape == (5,)
    assert y_std.shape == (5,)
    assert np.all(y_std >= 0)  # Standard deviations should be non-negative


def test_different_kernels(sample_data):
    """Test model with different kernel types."""
    X, y = sample_data

    # Test RBF kernel
    model_rbf = GaussianProcessRegressor(kernel="rbf")
    model_rbf.fit(X, y)
    y_pred_rbf, _ = model_rbf.predict(X)

    # Test Matern kernel
    model_matern = GaussianProcessRegressor(kernel="matern")
    model_matern.fit(X, y)
    y_pred_matern, _ = model_matern.predict(X)

    # Predictions should be different for different kernels
    assert not np.allclose(y_pred_rbf, y_pred_matern)


def test_model_parameters():
    """Test getting and setting model parameters."""
    model = GaussianProcessRegressor()

    # Test get_params
    params = model.get_params()
    assert "kernel" in params
    assert "noise_variance" in params

    # Test set_params
    model.set_params(noise_variance=0.5)
    assert model.noise_variance == 0.5
