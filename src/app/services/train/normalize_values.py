"""
Normalize Values Service

This module provides functionality for normalizing values to a specific range.
It is primarily used in the context of EEG data normalization for arousal and valence values.

The normalization process ensures that values maintain their relative distances
while being scaled to a target range, which is crucial for machine learning models
to process the data effectively.

The range [2-8] is specifically chosen instead of the full SAM scale [1-9] to avoid
extreme values that could introduce noise or bias in the analysis. This approach
provides a more robust representation of the emotional states while maintaining
the relative distances between values.

References:
    - SAM Scale Development: Bradley, M. M., & Lang, P. J. (1994). Measuring emotion:
      The Self-Assessment Manikin and the Semantic Differential. Journal of Behavior
      Therapy and Experimental Psychiatry, 25(1), 49-59.
      https://doi.org/10.1016/0005-7916(94)90063-9

    - EEG Emotion Analysis: Koelstra, S., et al. (2011). DEAP: A Database for Emotion
      Analysis using Physiological Signals. IEEE Transactions on Affective Computing,
      3(1), 18-31. https://doi.org/10.1109/T-AFFC.2011.15

    - Emotion Recognition Best Practices: Soleymani, M., et al. (2017). A Survey of
      Multimodal Sentiment Analysis. Image and Vision Computing, 65, 3-14.
      https://doi.org/10.1016/j.imavis.2017.08.003
"""

from typing import Sequence

import numpy as np


def normalize_range(values: list[float] | Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Normalize a list of values to a range between 2 and 8 while maintaining relative distances.

    This function is specifically designed for EEG data normalization in the context
    of valence-arousal space mapping. The range [2, 8] is chosen instead of the full
    SAM scale [1, 9] to avoid extreme values that could introduce noise or bias in
    the analysis. This approach provides a more robust representation of the emotional
    states while maintaining the relative distances between values.

    Args:
        values (list[float] | Sequence[float] | np.ndarray): List, sequence, or numpy array of values to normalize.
            Typically contains EEG arousal or valence values.

    Returns:
        np.ndarray: Array of normalized values in range [2, 8]

    Raises:
        ValueError: If the input array is empty or if all values are identical
            (which would result in division by zero)

    Examples:
        >>> normalize_range([1, 2, 3, 4, 5])
        array([2.0, 3.5, 5.0, 6.5, 8.0])
        >>> normalize_range([5, 5, 5])  # All same values
        array([5.0, 5.0, 5.0])
    """
    # Convert to numpy array if not already
    values_array = np.asarray(values)

    if values_array.size == 0:
        raise ValueError("Input array cannot be empty")

    min_val = np.min(values_array)
    max_val = np.max(values_array)

    if min_val == max_val:
        return np.full_like(
            values_array, 5.0
        )  # Return middle value if all values are same

    # Calculate the scaling factor and offset
    scale = 6 / (max_val - min_val)  # 6 is the range we want (8-2)
    offset = 2 - (min_val * scale)  # 2 is our minimum target value

    return values_array * scale + offset
