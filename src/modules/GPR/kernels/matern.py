"""
Matern kernel implementation.

The Matern kernel is a generalization of the RBF kernel that allows for
controlling the smoothness of the resulting function. It is particularly
useful when the underlying function is expected to be less smooth than
what the RBF kernel assumes.
"""

import numpy as np
from scipy.special import gamma, kv


class MaternKernel:
    """
    Matern kernel for Gaussian Process Regression.

    The Matern kernel is defined as:
    k(x, y) = σ² * (2^(1-ν)/Γ(ν)) * (√(2ν)||x-y||/l)^ν * K_ν(√(2ν)||x-y||/l)

    where:
    - σ² is the signal variance
    - l is the length scale
    - ν is the smoothness parameter
    - K_ν is the modified Bessel function of the second kind
    - ||x-y|| is the Euclidean distance
    """

    def __init__(self, length_scale=1.0, signal_variance=1.0, nu=1.5):
        """
        Initialize the Matern kernel.

        Args:
            length_scale: Length scale parameter(s). Can be a scalar or an array
                         for automatic relevance determination (ARD)
            signal_variance: Signal variance parameter (σ²)
            nu: Smoothness parameter (ν). Common values are 0.5, 1.5, and 2.5
        """
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.nu = nu

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

        # Apply length scale
        if isinstance(self.length_scale, np.ndarray):
            # ARD case
            distances = distances / self.length_scale
        else:
            # Isotropic case
            distances = distances / self.length_scale

        # Compute Matern kernel
        const = 2 ** (1 - self.nu) / gamma(self.nu)
        scaled_dist = np.sqrt(2 * self.nu) * distances

        # Handle the case where distances are zero
        mask = distances > 0
        K = np.zeros_like(distances)

        if np.any(mask):
            K[mask] = (
                const * (scaled_dist[mask] ** self.nu) * kv(self.nu, scaled_dist[mask])
            )

        # Handle the case where distances are zero
        K[~mask] = self.signal_variance

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
            "nu": self.nu,
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
