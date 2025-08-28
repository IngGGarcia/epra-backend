"""
This module provides functions to analyze the relationship between SAM (Self-Assessment Manikin)
and EEG data for emotional states, specifically focusing on valence and arousal dimensions.

The module implements various metrics to quantify the deviation between SAM and EEG measurements,
helping to identify systematic biases in the data collection process.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EmotionData:
    """Data class to store emotion measurements from SAM and EEG."""

    sam_valence: float
    eeg_valence: float
    sam_arousal: float
    eeg_arousal: float


def calculate_deviation(data: EmotionData) -> tuple[float, float]:
    """
    Calculate the deviation between SAM and EEG measurements for both valence and arousal.

    Args:
        data: An EmotionData object containing SAM and EEG measurements.

    Returns:
        A tuple containing (valence_deviation, arousal_deviation) where each value
        represents the absolute difference between SAM and EEG measurements.
    """
    valence_deviation = abs(data.sam_valence - data.eeg_valence)
    arousal_deviation = abs(data.sam_arousal - data.eeg_arousal)
    return valence_deviation, arousal_deviation


def analyze_emotion_data(
    data_list: list[EmotionData],
) -> dict[str, float | list[float] | bool]:
    """
    Analyze a list of emotion data points to identify systematic deviations between SAM and EEG.

    Args:
        data_list: A list of EmotionData objects containing multiple measurements.

    Returns:
        A dictionary containing:
        - mean_valence_deviation: Average deviation in valence measurements
        - mean_arousal_deviation: Average deviation in arousal measurements
        - valence_deviations: List of individual valence deviations
        - arousal_deviations: List of individual arousal deviations
        - systematic_bias: Boolean indicating if there's a consistent bias
    """
    valence_deviations = []
    arousal_deviations = []

    for data in data_list:
        v_dev, a_dev = calculate_deviation(data)
        valence_deviations.append(v_dev)
        arousal_deviations.append(a_dev)

    mean_valence_deviation = np.mean(valence_deviations)
    mean_arousal_deviation = np.mean(arousal_deviations)

    # Check for systematic bias by analyzing the sign of deviations
    valence_signs = [np.sign(data.sam_valence - data.eeg_valence) for data in data_list]
    arousal_signs = [np.sign(data.sam_arousal - data.eeg_arousal) for data in data_list]

    systematic_bias = (
        abs(np.mean(valence_signs))
        > 0.7  # If 70% of deviations are in the same direction
        or abs(np.mean(arousal_signs)) > 0.7
    )

    return {
        "mean_valence_deviation": mean_valence_deviation,
        "mean_arousal_deviation": mean_arousal_deviation,
        "valence_deviations": valence_deviations,
        "arousal_deviations": arousal_deviations,
        "systematic_bias": systematic_bias,
    }


def normalize_measurements(data: EmotionData) -> EmotionData:
    """
    Normalize the measurements to a common scale (1-9) for better comparison.
    Note: Both SAM and EEG data are already in 1-9 scale, so this function
    is mainly for consistency and future-proofing.

    Args:
        data: An EmotionData object containing raw measurements.

    Returns:
        A new EmotionData object with normalized values.
    """
    # Both SAM and EEG are already in 1-9 scale
    # This function is kept for consistency and future-proofing
    return EmotionData(
        sam_valence=data.sam_valence,
        eeg_valence=data.eeg_valence,
        sam_arousal=data.sam_arousal,
        eeg_arousal=data.eeg_arousal,
    )
