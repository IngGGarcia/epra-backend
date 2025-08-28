"""
Periodic kernel implementation.

The Periodic kernel is useful for modeling functions that exhibit periodic
behavior. It can be used to capture repeating patterns in the data.
"""

import numpy as np


class PeriodicKernel:
    """
    Periodic kernel for Gaussian Process Regression.

    The Periodic kernel is defined as:
    k(x, y) = σ² * exp(-2 * sin²(π||x-y||/p) / l²)

    where:
    - σ² is the signal variance
    - p is the period
    - l is the length scale
    - ||x-y|| is the Euclidean distance
    """

    def __init__(self, length_scale=1.0, signal_variance=1.0, period=1.0):
        """
        Initialize the Periodic kernel.

        Args:
            length_scale: Length scale parameter(s). Can be a scalar or an array
                         for automatic relevance determination (ARD)
            signal_variance: Signal variance parameter (σ²)
            period: Period of the periodic function
        """
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.period = period

    def __call__(self, X, Y=None):
        """
        Compute the kernel matrix between X and Y.

        Args:
            X: First set of points
            Y: Second set of points. If None, computes kernel between X and itself

        Returns:
            Kernel matrix
        """
        if Y is None:
            Y = X

        # Compute squared distances
        X2 = np.sum(X**2, axis=1)[:, np.newaxis]
        Y2 = np.sum(Y**2, axis=1)[np.newaxis, :]
        XY = X @ Y.T

        distances = np.sqrt(X2 + Y2 - 2 * XY)

        # Compute periodic term
        sin_term = np.sin(np.pi * distances / self.period)
        K = np.exp(-2 * sin_term**2 / self.length_scale**2)

        return self.signal_variance * K

    def get_params(self):
        """
        Get the kernel parameters.

        Returns:
            Dictionary of kernel parameters
        """
        return {
            "length_scale": self.length_scale,
            "signal_variance": self.signal_variance,
            "period": self.period,
        }

    def set_params(self, **params):
        """
        Set the kernel parameters.

        Args:
            **params: Parameter names mapped to their values

        Returns:
            self: The kernel instance
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
