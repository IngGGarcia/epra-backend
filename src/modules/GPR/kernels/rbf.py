"""
Radial Basis Function (RBF) kernel implementation.

The RBF kernel, also known as the Gaussian kernel, is one of the most commonly
used kernels in Gaussian Process Regression. It is particularly well-suited for
smooth, continuous functions.
"""

import numpy as np


class RBFKernel:
    """
    Radial Basis Function (RBF) kernel for Gaussian Process Regression.

    The RBF kernel is defined as:
    k(x, y) = σ² * exp(-||x-y||² / (2 * l²))

    where:
    - σ² is the signal variance
    - l is the length scale
    - ||x-y||² is the squared Euclidean distance
    """

    def __init__(self, length_scale=1.0, signal_variance=1.0):
        """
        Initialize the RBF kernel.

        Args:
            length_scale: Length scale parameter(s). Can be a scalar or an array
                         for automatic relevance determination (ARD)
            signal_variance: Signal variance parameter (σ²)
        """
        self.length_scale = length_scale
        self.signal_variance = signal_variance

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

        distances = X2 + Y2 - 2 * XY

        # Apply length scale
        if isinstance(self.length_scale, np.ndarray):
            # ARD case
            distances = distances / (2 * self.length_scale**2)
        else:
            # Isotropic case
            distances = distances / (2 * self.length_scale**2)

        # Apply signal variance and exponential
        return self.signal_variance * np.exp(-distances)

    def get_params(self):
        """
        Get the kernel parameters.

        Returns:
            Dictionary of kernel parameters
        """
        return {
            "length_scale": self.length_scale,
            "signal_variance": self.signal_variance,
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
