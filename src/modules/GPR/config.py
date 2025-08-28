"""
Configuration settings for the Gaussian Process Regression module.

This module contains default parameters, constants, and configuration settings
used throughout the GPR implementation.
"""

# Default kernel parameters
DEFAULT_KERNEL = "rbf"
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_SIGNAL_VARIANCE = 1.0
DEFAULT_NOISE_VARIANCE = 0.1

# Model training parameters
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_OPTIMIZER = "lbfgs"
DEFAULT_RANDOM_STATE = 42

# Validation parameters
DEFAULT_CROSS_VALIDATION_FOLDS = 5
DEFAULT_TEST_SIZE = 0.2

# Output parameters
DEFAULT_CONFIDENCE_LEVEL = 0.95  # For prediction intervals

# File paths and naming conventions
MODEL_SAVE_DIR = "models"
MODEL_FILE_EXTENSION = ".gpr"
