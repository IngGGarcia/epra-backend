"""
EEG Segment Extractor Module

This module provides functionality to extract EEG data segments corresponding
to image exposure periods from validated EEG data, excluding normalization
and response periods.

The module processes EEG data that follows a cyclic pattern:
- Normalization Time (5 seconds): Baseline period - excluded
- Image Exposure Time (30 seconds): Image display period - extracted
- Test SAM Response Time (15 seconds): Response evaluation period - excluded

Total cycle time: 50 seconds
"""

import pandas as pd

from .exceptions import EEGFileError

# Constants for experiment timing (in seconds)
NORMALIZATION_TIME = 5
IMAGE_EXPOSURE_TIME = 30
TEST_SAM_RESPONSE_TIME = 15
CYCLE_TIME = (
    NORMALIZATION_TIME + IMAGE_EXPOSURE_TIME + TEST_SAM_RESPONSE_TIME
)  # 50 seconds


def extract_exposure_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract EEG data segments corresponding only to image exposure periods.

    This function processes the validated EEG DataFrame and extracts only the
    segments where images are being displayed to participants, excluding
    normalization and response periods. Each exposure segment is labeled
    with an image_number starting from 1.

    Args:
        df: Validated DataFrame containing EEG data with columns:
            ['Timestamp', 'EEG.AF3', 'EEG.F3', 'EEG.AF4', 'EEG.F4']

    Returns:
        pandas.DataFrame: DataFrame containing only exposure segments with
                         an additional 'image_number' column indicating
                         the cycle number (1, 2, 3, ...)

    Raises:
        EEGFileError: If the DataFrame is empty, has insufficient data for
                     at least one complete cycle, or has invalid timestamps.

    Example:
        >>> df_validated = verify_eeg_file(file_input, filename)
        >>> df_exposure = extract_exposure_segments(df_validated)
        >>> print(df_exposure.columns.tolist())
        ['Timestamp', 'EEG.AF3', 'EEG.F3', 'EEG.AF4', 'EEG.F4', 'image_number']
    """
    if df.empty:
        raise EEGFileError("DataFrame is empty, cannot extract segments")

    if "Timestamp" not in df.columns:
        raise EEGFileError("DataFrame must contain 'Timestamp' column")

    # Calculate experiment duration and number of complete cycles
    start_time = df["Timestamp"].min()
    end_time = df["Timestamp"].max()
    total_duration = end_time - start_time

    if total_duration < CYCLE_TIME:
        raise EEGFileError(
            f"Insufficient data duration ({total_duration:.2f}s). "
            f"Need at least {CYCLE_TIME}s for one complete cycle."
        )

    num_cycles = int(total_duration / CYCLE_TIME)

    if num_cycles == 0:
        raise EEGFileError("No complete cycles found in the data")

    # Extract exposure segments for each cycle
    exposure_segments = []

    for cycle in range(num_cycles):
        # Calculate time boundaries for this cycle's exposure period
        cycle_start = start_time + cycle * CYCLE_TIME
        exposure_start = cycle_start + NORMALIZATION_TIME
        exposure_end = exposure_start + IMAGE_EXPOSURE_TIME

        # Extract the exposure segment for this cycle
        exposure_mask = (df["Timestamp"] >= exposure_start) & (
            df["Timestamp"] < exposure_end
        )

        exposure_segment = df[exposure_mask].copy()

        if not exposure_segment.empty:
            # Add image_number column (1-indexed)
            exposure_segment["image_number"] = cycle + 1
            exposure_segments.append(exposure_segment)

    if not exposure_segments:
        raise EEGFileError("No exposure segments could be extracted from the data")

    # Concatenate all exposure segments into a single DataFrame
    result_df = pd.concat(exposure_segments, ignore_index=True)

    # Ensure proper column order
    columns_order = [
        "Timestamp",
        "EEG.AF3",
        "EEG.F3",
        "EEG.AF4",
        "EEG.F4",
        "image_number",
    ]
    result_df = result_df[columns_order]

    return result_df
