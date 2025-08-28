"""
EEG Data Validator

This module handles the validation of EEG DataFrame structure,
including column validation and timestamp conversion.
"""

from datetime import datetime

import pandas as pd

from .exceptions import EEGFileError

# Constants for timestamp validation (same as SAM file)
MIN_TIMESTAMP = 946684800  # Unix timestamp for 2000-01-01
MAX_TIMESTAMP = 4102444800  # Unix timestamp for 2100-01-01


def validate_eeg_columns(df: pd.DataFrame) -> None:
    """
    Validate that the DataFrame contains the required EEG columns.

    Args:
        df: DataFrame to validate

    Raises:
        EEGFileError: If required columns are missing
    """
    required_columns = {"Timestamp", "EEG.AF3", "EEG.F3", "EEG.AF4", "EEG.F4"}
    actual_columns = set(df.columns)

    if not required_columns.issubset(actual_columns):
        missing_columns = required_columns - actual_columns
        raise EEGFileError(
            f"Missing required EEG columns: {missing_columns}. "
            f"Found columns: {list(actual_columns)}"
        )


def _validate_unix_timestamp(timestamp: float) -> bool:
    """
    Validate that a numeric value represents a valid Unix timestamp.

    Accepts both seconds and milliseconds timestamps and normalizes them.

    Args:
        timestamp: Numeric timestamp value to validate

    Returns:
        bool: True if timestamp is valid, False otherwise
    """
    try:
        # Handle potential milliseconds timestamp (if value is very large)
        if timestamp > MAX_TIMESTAMP:
            timestamp = timestamp / 1000

        # Check if it's a reasonable timestamp (in seconds)
        if timestamp < MIN_TIMESTAMP or timestamp > MAX_TIMESTAMP:
            return False

        # Try to convert to datetime to ensure it's valid
        datetime.fromtimestamp(timestamp)
        return True
    except (ValueError, OSError, OverflowError):
        return False


def _normalize_timestamp(timestamp: float) -> int:
    """
    Normalize timestamp to Unix timestamp in seconds (integer).

    Args:
        timestamp: Numeric timestamp (could be seconds or milliseconds)

    Returns:
        int: Unix timestamp in seconds
    """
    # If timestamp is in milliseconds (very large number), convert to seconds
    if timestamp > MAX_TIMESTAMP:
        timestamp = timestamp / 1000

    # Round to nearest second for consistency
    return int(round(timestamp))


def validate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize timestamps to Unix format (integer seconds).

    This function:
    - Converts timestamps to numeric format
    - Validates they are reasonable Unix timestamps
    - Normalizes them to integer seconds for consistency with SAM files

    Args:
        df: DataFrame with timestamp column to validate

    Returns:
        pd.DataFrame: DataFrame with validated and normalized integer timestamps

    Raises:
        EEGFileError: If timestamps contain invalid values or are out of reasonable range
    """
    df_copy = df.copy()

    # Convert to numeric first
    df_copy["Timestamp"] = pd.to_numeric(df_copy["Timestamp"], errors="coerce")

    # Check for NaN values after conversion
    if df_copy["Timestamp"].isna().sum() > 0:
        raise EEGFileError(
            "EEG data contains invalid or NaN values in timestamps. Consider preprocessing."
        )

    # Validate each timestamp
    invalid_timestamps = []
    for idx, timestamp in enumerate(df_copy["Timestamp"]):
        if not _validate_unix_timestamp(timestamp):
            invalid_timestamps.append(idx)

    if invalid_timestamps:
        raise EEGFileError(
            f"Invalid Unix timestamps found at rows: {invalid_timestamps[:10]}... "
            f"Timestamps must be valid Unix timestamps between {MIN_TIMESTAMP} and {MAX_TIMESTAMP}"
        )

    # Normalize all timestamps to integer seconds
    df_copy["Timestamp"] = df_copy["Timestamp"].apply(_normalize_timestamp)

    return df_copy
