"""
Verify SAM File

This module verifies the SAM file for an evaluation.
"""

from datetime import datetime
from pathlib import Path
import re
from typing import BinaryIO

import pandas as pd

# Constants for SAM file validation
EXPECTED_ROWS_COUNT = 5
MIN_TIMESTAMP = 946684800  # Unix timestamp for 2000-01-01
MAX_TIMESTAMP = 4102444800  # Unix timestamp for 2100-01-01
SAM_SCALE_MIN = 1  # Minimum value on SAM scale
SAM_SCALE_MAX = 9  # Maximum value on SAM scale


class SAMFileError(Exception):
    """Custom exception for SAM file validation errors."""

    def __init__(self, message: str = "Error verifying SAM file"):
        """Initialize SAMFileError with a default message."""
        super().__init__(message)


def _validate_timestamp(timestamp_str: str) -> bool:
    """
    Validate that a string represents a valid Unix timestamp.

    Accepts both seconds and milliseconds timestamps.

    Args:
        timestamp_str: String containing the timestamp to validate

    Returns:
        bool: True if timestamp is valid, False otherwise
    """
    try:
        # Convert to integer first
        timestamp = int(timestamp_str)

        # Check if it's in milliseconds (13 digits) and convert to seconds
        if len(timestamp_str) == 13:
            timestamp = timestamp / 1000

        # Check if it's a reasonable timestamp (in seconds)
        if timestamp < MIN_TIMESTAMP or timestamp > MAX_TIMESTAMP:
            return False

        # Try to convert to datetime to ensure it's valid
        datetime.fromtimestamp(timestamp)
        return True
    except (ValueError, OSError, OverflowError):
        return False


def verify_sam_file(file_input: BinaryIO, filename: str) -> pd.DataFrame:
    """
    Verify and validate a SAM results CSV file from HTTP upload.

    This function validates that the CSV file:
    - Has the correct filename format (sam_results_<timestamp> or sam_results_<userId>_<timestamp>)
    - Contains a valid Unix timestamp (accepts both seconds and milliseconds)
    - Contains the required columns: image_id, valence, arousal
    - Has exactly 5 rows of data
    - Contains valid numeric data for valence and arousal

    Args:
        file_input: File-like object from HTTP upload (e.g., UploadFile.file)
        filename: Original filename for format validation

    Returns:
        pandas.DataFrame: The validated DataFrame containing the SAM data

    Raises:
        SAMFileError: If the file doesn't meet validation requirements

    Example:
        >>> df = verify_sam_file(upload_file.file, "sam_results_1747184865309.csv")
        >>> print(df.shape)  # (EXPECTED_ROWS_COUNT, 3)
        >>> print(df.columns.tolist())  # ['image_id', 'valence', 'arousal']

        >>> df = verify_sam_file(upload_file.file, "sam_results_123_1747184865309.csv")
        >>> print(df.shape)  # (EXPECTED_ROWS_COUNT, 3)
    """
    if filename is None:
        raise SAMFileError("Filename is required")

    # Get filename without extension for validation
    validation_filename = Path(filename).stem

    # Validate filename format and extract timestamp
    # Accept both formats: sam_results_<timestamp> and sam_results_<userId>_<timestamp>
    match = re.match(r"^sam_results_(?:\d+_)?(\d+)$", validation_filename)
    if not match:
        raise SAMFileError(
            f"Invalid filename format. Expected 'sam_results_<timestamp>' or 'sam_results_<userId>_<timestamp>', got '{validation_filename}'"
        )

    # Validate that the timestamp is valid
    timestamp_str = match.group(1)
    if not _validate_timestamp(timestamp_str):
        raise SAMFileError(
            f"Invalid timestamp in filename. '{timestamp_str}' is not a valid Unix timestamp"
        )

    try:
        # Reset file pointer to beginning
        if hasattr(file_input, "seek"):
            file_input.seek(0)

        # Read CSV from file-like object
        df = pd.read_csv(file_input)
    except Exception as e:
        raise SAMFileError(f"Error reading CSV file: {str(e)}") from e

    # Validate required columns
    required_columns = {"image_id", "valence", "arousal"}
    actual_columns = set(df.columns)

    if not required_columns.issubset(actual_columns):
        missing_columns = required_columns - actual_columns
        raise SAMFileError(
            f"Missing required columns: {missing_columns}. "
            f"Found columns: {list(actual_columns)}"
        )

    # Validate number of rows
    if len(df) != EXPECTED_ROWS_COUNT:
        raise SAMFileError(
            f"Expected exactly {EXPECTED_ROWS_COUNT} rows of data, found {len(df)} rows"
        )

    # Validate data types and values
    try:
        # Check that valence and arousal are numeric
        df["valence"] = pd.to_numeric(df["valence"], errors="raise")
        df["arousal"] = pd.to_numeric(df["arousal"], errors="raise")
    except ValueError as e:
        raise SAMFileError(
            f"Invalid numeric data in valence or arousal columns: {str(e)}"
        ) from e

    # Validate valence and arousal ranges (typically SAM scale is 1-9)
    if not df["valence"].between(SAM_SCALE_MIN, SAM_SCALE_MAX).all():
        raise SAMFileError(
            f"Valence values must be between {SAM_SCALE_MIN} and {SAM_SCALE_MAX}"
        )

    if not df["arousal"].between(SAM_SCALE_MIN, SAM_SCALE_MAX).all():
        raise SAMFileError(
            f"Arousal values must be between {SAM_SCALE_MIN} and {SAM_SCALE_MAX}"
        )

    # Ensure no null values
    if df.isnull().any().any():
        raise SAMFileError("CSV contains null/empty values")

    return df
