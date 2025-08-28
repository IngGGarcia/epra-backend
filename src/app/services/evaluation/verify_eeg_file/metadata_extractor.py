"""
EEG Metadata Extractor

This module handles the extraction of metadata from EEG CSV files,
specifically the sampling rate from the first line metadata.
"""

import re

from .exceptions import EEGFileError


def extract_sampling_rate_from_metadata(metadata_line: str) -> int:
    """
    Extract the EEG sampling rate from the metadata line of the CSV file.

    Args:
        metadata_line: The first line of the CSV file containing metadata

    Returns:
        int: The sampling rate in Hz

    Raises:
        EEGFileError: If the sampling rate cannot be extracted

    Example:
        >>> extract_sampling_rate_from_metadata("sampling rate: eeg_128")
        128.0
    """
    match = re.search(r"sampling rate:\s*eeg_(\d+)", metadata_line)
    if match:
        return int(match.group(1))
    else:
        raise EEGFileError(
            "Sampling rate not found in metadata. Ensure the file format is correct."
        )
