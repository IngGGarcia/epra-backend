"""
Tests for kernel functions used in Gaussian Process Regression.

This module contains unit tests for various kernel functions,
including RBF, Matern, Linear, and Periodic kernels.
"""

import numpy as np
import pytest

from src.modules.GPR.kernels import (
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
)


@pytest.fixture
def sample_points():
    """Generate sample points for testing kernels."""
    np.random.seed(42)
    X = np.random.randn(5, 3)  # 5 points in 3D space
    Y = np.random.randn(3, 3)  # 3 points in 3D space
    return X, Y


def test_rbf_kernel(sample_points):
    """Test RBF kernel implementation."""
    X, Y = sample_points
    kernel = RBFKernel(length_scale=1.0, signal_variance=1.0)

    # Test kernel computation
    K = kernel(X, Y)
    assert K.shape == (5, 3)
    assert np.all(K >= 0)  # Kernel values should be non-negative

    # Test symmetry
    K_self = kernel(X)
    assert np.allclose(K_self, K_self.T)  # Should be symmetric

    # Test parameters
    params = kernel.get_params()
    assert "length_scale" in params
    assert "signal_variance" in params


def test_matern_kernel(sample_points):
    """Test Matern kernel implementation."""
    X, Y = sample_points
    kernel = MaternKernel(length_scale=1.0, signal_variance=1.0, nu=1.5)

    # Test kernel computation
    K = kernel(X, Y)
    assert K.shape == (5, 3)
    assert np.all(K >= 0)

    # Test different nu values
    kernel_nu_0_5 = MaternKernel(nu=0.5)
    kernel_nu_2_5 = MaternKernel(nu=2.5)

    K_0_5 = kernel_nu_0_5(X, Y)
    K_2_5 = kernel_nu_2_5(X, Y)

    assert not np.allclose(K_0_5, K_2_5)  # Different nu should give different results


def test_linear_kernel(sample_points):
    """Test Linear kernel implementation."""
    X, Y = sample_points
    kernel = LinearKernel(signal_variance=1.0, constant=1.0)

    # Test kernel computation
    K = kernel(X, Y)
    assert K.shape == (5, 3)

    # Test constant term
    K_no_constant = LinearKernel(constant=0.0)(X, Y)
    assert not np.allclose(K, K_no_constant)


def test_kernel_parameters():
    """Test getting and setting kernel parameters."""
    # Test RBF kernel
    kernel = RBFKernel(length_scale=2.0, signal_variance=1.5)
    params = kernel.get_params()
    assert params["length_scale"] == 2.0
    assert params["signal_variance"] == 1.5

    kernel.set_params(length_scale=3.0)
    assert kernel.length_scale == 3.0

    # Test Matern kernel
    kernel = MaternKernel(nu=1.5)
    params = kernel.get_params()
    assert params["nu"] == 1.5

    # Test Linear kernel
    kernel = LinearKernel(constant=1.0)
    params = kernel.get_params()
    assert params["constant"] == 1.0

    # Test Periodic kernel
    kernel = PeriodicKernel(period=2.0)
    params = kernel.get_params()
    assert params["period"] == 2.0
