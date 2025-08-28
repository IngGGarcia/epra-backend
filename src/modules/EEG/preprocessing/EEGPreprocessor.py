"""
EEG Data Preprocessing Module

This module provides functionality for preprocessing EEG data from CSV files, cleaning it,
and extracting structured EEG segments corresponding to different phases of an experiment
involving image exposure and Test SAM evaluation.

The class is designed to work with EEG data recorded in experiments where participants
are exposed to images and then evaluate their emotional responses using the Self-Assessment Manikin (Test SAM).
The EEG data is segmented into distinct phases based on predefined time intervals.

The CSV file must follow the following format:
    - Timestamp: Time in seconds since the beginning of the experiment.
    - EEG.AF3, EEG.F3, EEG.AF4, EEG.F4: EEG signals recorded from frontal electrodes.

The experiment consists of repeated cycles, each containing:
    - **Normalization Time (Baseline Period)**: A period where no image is shown,
      allowing the EEG signal to stabilize before exposure.
    - **Image Exposure Time**: The time during which an image is displayed to the participant.
    - **Test SAM Response Time**: The time allocated for the participant to assess the image
      using the Self-Assessment Manikin (SAM) scale, without further visual stimuli.

Attributes:
    csv_path (str): Path to the CSV file containing EEG data.
    full_data (bool): If True, includes the Test SAM response phase; otherwise,
                      extracts only the image exposure period.
    df (pd.DataFrame): Processed EEG data containing relevant columns.
    sampling_rate (float): Estimated sampling rate of the EEG data in Hz.
"""

import re

import pandas as pd


class EEGPreprocessor:
    """
    A class to preprocess EEG data from a CSV file, clean it, and extract structured EEG segments
    corresponding to different phases of an experiment involving image exposure and Test SAM evaluation.

    This class is designed to work with EEG data recorded in experiments where participants
    are exposed to images and then evaluate their emotional responses using the Self-Assessment Manikin (Test SAM).
    The EEG data is segmented into distinct phases based on predefined time intervals.

    The CSV file must follow the following format:
        - Timestamp: Time in seconds since the beginning of the experiment.
        - EEG.AF3, EEG.F3, EEG.AF4, EEG.F4: EEG signals recorded from frontal electrodes.

    The experiment consists of repeated cycles, each containing:
        - **Normalization Time (Baseline Period)**: A period where no image is shown,
          allowing the EEG signal to stabilize before exposure.
        - **Image Exposure Time**: The time during which an image is displayed to the participant.
        - **Test SAM Response Time**: The time allocated for the participant to assess the image
          using the Self-Assessment Manikin (SAM) scale, without further visual stimuli.

    Attributes:
        csv_path (str): Path to the CSV file containing EEG data.
        full_data (bool): If True, includes the Test SAM response phase; otherwise,
                          extracts only the image exposure period.
        df (pd.DataFrame): Processed EEG data containing relevant columns.
        sampling_rate (float): Estimated sampling rate of the EEG data in Hz.
    """

    # Constants for experiment timing (in seconds)
    NORMALIZATION_TIME = 5
    IMAGE_EXPOSURE_TIME = 30
    TEST_SAM_RESPONSE_TIME = 15
    CYCLE_TIME = (
        NORMALIZATION_TIME + IMAGE_EXPOSURE_TIME + TEST_SAM_RESPONSE_TIME
    )  # 50 seconds per cycle

    def __init__(self, csv_path: str, full_data: bool = False):
        """
        Initialize the EEGPreprocessor with the given CSV file path.

        Args:
            csv_path: Path to the CSV file containing EEG data.
            full_data: If True, includes the Test SAM response phase;
                      otherwise, only the image exposure period.
        """
        print("EEG API is running. Use /docs for documentation.")
        self.csv_path = csv_path
        self.full_data = full_data
        self.sampling_rate = self._extract_sampling_rate()
        self.df = self._load_eeg_data()
        self._validate_sampling_rate()

    def _extract_sampling_rate(self) -> float:
        """
        Extract the EEG sampling rate from the metadata row of the CSV file.

        Returns:
            float: The sampling rate in Hz.

        Raises:
            ValueError: If the sampling rate cannot be extracted.
            FileNotFoundError: If the CSV file is not found.
        """
        try:
            with open(self.csv_path, "r") as file:
                metadata_line = file.readline().strip()

            match = re.search(r"sampling rate:\s*eeg_(\d+)", metadata_line)
            print(match)
            if match:
                return float(match.group(1))
            else:
                raise ValueError(
                    "Sampling rate not found in metadata. Ensure the file format is correct."
                )

        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {self.csv_path}")

    def _load_eeg_data(self) -> pd.DataFrame:
        """
        Load EEG data from a CSV file, skipping the metadata row.

        Returns:
            pd.DataFrame: Processed DataFrame with selected relevant columns.

        Raises:
            ValueError: If required columns are missing or timestamps contain NaN values.
            FileNotFoundError: If the CSV file is not found.
        """
        try:
            df = pd.read_csv(self.csv_path, skiprows=1)
            relevant_columns = ["Timestamp", "EEG.AF3", "EEG.F3", "EEG.AF4", "EEG.F4"]

            if not set(relevant_columns).issubset(df.columns):
                missing_cols = set(relevant_columns) - set(df.columns)
                raise ValueError(
                    f"Missing expected columns in CSV file: {missing_cols}"
                )

            df_relevant = df[relevant_columns].copy()
            df_relevant["Timestamp"] = pd.to_numeric(
                df_relevant["Timestamp"], errors="coerce"
            )

            if df_relevant["Timestamp"].isna().sum() > 0:
                raise ValueError(
                    "EEG data contains NaN values in timestamps. Consider preprocessing."
                )

            return df_relevant

        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {self.csv_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file {self.csv_path} is empty or corrupted.")

    def _validate_sampling_rate(self):
        """
        Validate that the estimated sampling rate is consistent with the metadata.
        If the difference exceeds a predefined threshold, an exception is raised.

        Raises:
            ValueError: If the estimated sampling rate differs significantly from the metadata.
        """
        time_diffs = self.df["Timestamp"].diff().dropna()
        estimated_sampling_rate = 1 / time_diffs.mean()
        threshold = 0.5  # Acceptable difference in Hz
        if abs(self.sampling_rate - estimated_sampling_rate) > threshold:
            raise ValueError(
                f"Estimated sampling rate ({estimated_sampling_rate:.2f} Hz) "
                f"differs significantly from metadata ({self.sampling_rate} Hz). Check data integrity."
            )

    def extract_image_segments(self) -> list[pd.DataFrame]:
        """
        Extract EEG data segments corresponding to the image exposure period and optionally
        the Test SAM response phase.

        Returns:
            list[pd.DataFrame]: List of DataFrames containing EEG segments for each image exposure
                               and optionally the Test SAM response phase.
        """
        start_time = self.df["Timestamp"].min()
        end_time = self.df["Timestamp"].max()
        duration = end_time - start_time
        num_cycles = int(duration / self.CYCLE_TIME)
        image_data = []

        for i in range(num_cycles):
            start_img = start_time + i * self.CYCLE_TIME + self.NORMALIZATION_TIME
            end_img = start_img + self.IMAGE_EXPOSURE_TIME
            end_full = (
                end_img + self.TEST_SAM_RESPONSE_TIME if self.full_data else end_img
            )
            img_segment = self.df[
                (self.df["Timestamp"] >= start_img) & (self.df["Timestamp"] < end_full)
            ]
            image_data.append(img_segment)

        return image_data
