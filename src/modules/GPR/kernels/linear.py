"""
Linear kernel implementation.

The Linear kernel is useful when the relationship between inputs and outputs
is expected to be linear. It can be combined with other kernels to model
both linear and non-linear relationships.
"""

import numpy as np


class LinearKernel:
    """
    Linear kernel for Gaussian Process Regression.

    The Linear kernel is defined as:
    k(x, y) = σ² * (x^T y + c)

    where:
    - σ² is the signal variance
    - c is the constant term
    """

    def __init__(self, signal_variance=1.0, constant=0.0):
        """
        Initialize the Linear kernel.

        Args:
            signal_variance: Signal variance parameter (σ²)
            constant: Constant term (c) added to the dot product
        """
        self.signal_variance = signal_variance
        self.constant = constant

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

        # Compute dot product
        K = X @ Y.T

        # Add constant term
        K += self.constant

        return self.signal_variance * K

    def get_params(self):
        """
        Get the kernel parameters.

        Returns:
            Dictionary of kernel parameters
        """
        return {"signal_variance": self.signal_variance, "constant": self.constant}

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
