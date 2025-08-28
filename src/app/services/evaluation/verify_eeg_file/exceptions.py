"""
EEG File Validation Exceptions

This module defines custom exceptions for EEG file validation errors.
"""


class EEGFileError(Exception):
    """Custom exception for EEG file validation errors."""

    def __init__(self, message: str = "Error verifying EEG file"):
        """
        Initialize EEGFileError with a default message.

        Args:
            message: Error message describing the validation failure
        """
        super().__init__(message)
