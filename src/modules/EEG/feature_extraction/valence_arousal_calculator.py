"""
EEG-Based Valence and Arousal Calculator

This module implements state-of-the-art methods for calculating valence and arousal
from EEG data based on recent research findings. It includes multiple feature
extraction methods and machine learning approaches for emotion recognition.

Key Methods Implemented:
- Power Spectral Density (PSD) in frequency bands
- Differential Entropy (DE) - most widely used time-frequency feature
- Asymmetry features between left/right hemispheres
- Time domain features (first-order and second-order differences)
- Support Vector Machine (SVM) classification with RBF kernel

References:
- Li et al. (2023). CNN-KAN-F2CA model for emotion recognition
- Zheng et al. (2015). Differential entropy feature for EEG-based emotion classification
- Koelstra et al. (2012). DEAP: A database for emotion analysis using physiological signals
- Cai et al. (2020). Feature-level fusion approaches based on multimodal EEG data
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ValenceArousalCalculator:
    """
    Advanced EEG-based valence and arousal calculator implementing multiple
    state-of-the-art feature extraction and classification methods.

    This class provides comprehensive functionality for:
    - Multi-band power spectral density analysis
    - Differential entropy computation
    - Hemispheric asymmetry analysis
    - Time-domain feature extraction
    - Machine learning-based emotion classification

    Attributes:
        sampling_rate (int): EEG sampling rate in Hz (default: 128)
        window_size (float): Analysis window size in seconds (default: 1.0)
        overlap (float): Window overlap ratio (default: 0.5)
        frequency_bands (Dict): EEG frequency band definitions
        channels (List[str]): Channel names for analysis
        scaler (StandardScaler): Feature scaler for ML models
        valence_model (SVC): Trained SVM model for valence prediction
        arousal_model (SVC): Trained SVM model for arousal prediction
    """

    def __init__(
        self,
        sampling_rate: int = 128,
        window_size: float = 1.0,
        overlap: float = 0.5,
        channels: Optional[List[str]] = None,
    ):
        """
        Initialize the ValenceArousalCalculator.

        Args:
            sampling_rate: EEG sampling rate in Hz
            window_size: Analysis window size in seconds
            overlap: Window overlap ratio (0.0-1.0)
            channels: List of channel names (default: ['AF3', 'AF4', 'F3', 'F4'])
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.channels = channels or ["AF3", "AF4", "F3", "F4"]

        # EEG frequency bands based on literature
        self.frequency_bands = {
            "delta": (1, 4),  # Delta: 1-4 Hz
            "theta": (4, 8),  # Theta: 4-8 Hz
            "alpha": (8, 13),  # Alpha: 8-13 Hz
            "beta": (13, 30),  # Beta: 13-30 Hz
            "gamma": (30, 50),  # Gamma: 30-50 Hz
            "high_gamma": (
                50,
                80,
            ),  # High Gamma: 50-80 Hz (best for emotion recognition)
        }

        # Initialize ML models and scaler
        self.scaler = StandardScaler()
        self.valence_model = None
        self.arousal_model = None

        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore", category=UserWarning)

    def _bandpass_filter(
        self, data: np.ndarray, low_freq: float, high_freq: float, order: int = 4
    ) -> np.ndarray:
        """
        Apply bandpass filter to EEG data.

        Args:
            data: Input EEG signal
            low_freq: Lower cutoff frequency
            high_freq: Higher cutoff frequency
            order: Filter order

        Returns:
            Filtered EEG signal
        """
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        # Handle edge cases
        if low <= 0:
            low = 0.01
        if high >= 1:
            high = 0.99

        b, a = signal.butter(order, [low, high], btype="band")
        return signal.filtfilt(b, a, data)

    def extract_power_spectral_density(
        self, eeg_data: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract Power Spectral Density (PSD) features for each frequency band.

        PSD is one of the most fundamental features for EEG emotion recognition,
        providing information about the power distribution across frequencies.

        Args:
            eeg_data: Dictionary with channel names as keys and EEG signals as values

        Returns:
            Dictionary containing PSD values for each channel and frequency band
        """
        psd_features = {}

        for channel, data in eeg_data.items():
            psd_features[channel] = {}

            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # Filter data to specific frequency band
                filtered_data = self._bandpass_filter(data, low_freq, high_freq)

                # Calculate PSD using Welch's method
                freqs, psd = signal.welch(
                    filtered_data,
                    fs=self.sampling_rate,
                    window="hann",
                    nperseg=int(self.sampling_rate * self.window_size),
                    noverlap=int(self.sampling_rate * self.window_size * self.overlap),
                )

                # Extract power in the frequency band
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.trapz(psd[band_mask], freqs[band_mask])

                psd_features[channel][band_name] = float(band_power)

        return psd_features

    def extract_differential_entropy(
        self, eeg_data: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract Differential Entropy (DE) features for each frequency band.

        DE is the most widely used time-frequency feature for EEG emotion recognition.
        It measures the complexity and randomness of EEG signals in specific frequency bands.

        Formula: DE = (1/2) * log(2*π*e*σ²)
        where σ² is the variance of the signal in the frequency band.

        Args:
            eeg_data: Dictionary with channel names as keys and EEG signals as values

        Returns:
            Dictionary containing DE values for each channel and frequency band
        """
        de_features = {}

        for channel, data in eeg_data.items():
            de_features[channel] = {}

            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # Filter data to specific frequency band
                filtered_data = self._bandpass_filter(data, low_freq, high_freq)

                # Calculate differential entropy
                variance = np.var(filtered_data)
                if variance > 0:
                    de_value = 0.5 * np.log(2 * np.pi * np.e * variance)
                else:
                    de_value = 0.0

                de_features[channel][band_name] = float(de_value)

        return de_features

    def extract_asymmetry_features(
        self, eeg_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Extract hemispheric asymmetry features.

        Asymmetry features capture the difference in brain activity between
        left and right hemispheres, which is crucial for valence detection.
        Research shows that left hemisphere activation is associated with
        positive emotions, while right hemisphere activation is associated
        with negative emotions.

        Args:
            eeg_data: Dictionary with channel names as keys and EEG signals as values

        Returns:
            Dictionary containing asymmetry features for each frequency band
        """
        asymmetry_features = {}

        # Define electrode pairs for asymmetry calculation
        electrode_pairs = [
            ("F3", "F4"),  # Frontal asymmetry
            ("AF3", "AF4"),  # Anterior frontal asymmetry
        ]

        for left_ch, right_ch in electrode_pairs:
            if left_ch in eeg_data and right_ch in eeg_data:
                left_data = eeg_data[left_ch]
                right_data = eeg_data[right_ch]

                for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                    # Filter data for both channels
                    left_filtered = self._bandpass_filter(
                        left_data, low_freq, high_freq
                    )
                    right_filtered = self._bandpass_filter(
                        right_data, low_freq, high_freq
                    )

                    # Calculate power for both hemispheres
                    left_power = np.mean(left_filtered**2)
                    right_power = np.mean(right_filtered**2)

                    # Calculate asymmetry (log(right) - log(left))
                    # Positive values indicate right hemisphere dominance
                    # Negative values indicate left hemisphere dominance
                    if left_power > 0 and right_power > 0:
                        asymmetry = np.log(right_power) - np.log(left_power)
                    else:
                        asymmetry = 0.0

                    feature_name = f"{left_ch}_{right_ch}_{band_name}_asymmetry"
                    asymmetry_features[feature_name] = float(asymmetry)

        return asymmetry_features

    def extract_time_domain_features(
        self, eeg_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Extract time-domain features including statistical measures
        and first/second-order differences.

        Time-domain features provide complementary information to
        frequency-domain features and can improve emotion recognition accuracy.

        Args:
            eeg_data: Dictionary with channel names as keys and EEG signals as values

        Returns:
            Dictionary containing time-domain features for each channel
        """
        time_features = {}

        for channel, data in eeg_data.items():
            # Basic statistical features
            time_features[f"{channel}_mean"] = float(np.mean(data))
            time_features[f"{channel}_std"] = float(np.std(data))
            time_features[f"{channel}_var"] = float(np.var(data))
            time_features[f"{channel}_skewness"] = float(self._calculate_skewness(data))
            time_features[f"{channel}_kurtosis"] = float(self._calculate_kurtosis(data))

            # First-order difference features
            first_diff = np.diff(data)
            time_features[f"{channel}_first_diff_mean"] = float(np.mean(first_diff))
            time_features[f"{channel}_first_diff_std"] = float(np.std(first_diff))

            # Second-order difference features
            second_diff = np.diff(first_diff)
            time_features[f"{channel}_second_diff_mean"] = float(np.mean(second_diff))
            time_features[f"{channel}_second_diff_std"] = float(np.std(second_diff))

            # Zero-crossing rate
            zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
            time_features[f"{channel}_zero_crossing_rate"] = float(
                zero_crossings / len(data)
            )

        return time_features

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the signal."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the signal."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)

    def extract_all_features(self, eeg_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Extract all available features from EEG data.

        This method combines all feature extraction methods to create
        a comprehensive feature vector for emotion recognition.

        Args:
            eeg_data: Dictionary with channel names as keys and EEG signals as values

        Returns:
            Dictionary containing all extracted features
        """
        all_features = {}

        # Extract PSD features
        psd_features = self.extract_power_spectral_density(eeg_data)
        for channel, bands in psd_features.items():
            for band, value in bands.items():
                all_features[f"{channel}_{band}_psd"] = value

        # Extract DE features
        de_features = self.extract_differential_entropy(eeg_data)
        for channel, bands in de_features.items():
            for band, value in bands.items():
                all_features[f"{channel}_{band}_de"] = value

        # Extract asymmetry features
        asymmetry_features = self.extract_asymmetry_features(eeg_data)
        all_features.update(asymmetry_features)

        # Extract time-domain features
        time_features = self.extract_time_domain_features(eeg_data)
        all_features.update(time_features)

        return all_features

    def calculate_valence_arousal_heuristic(
        self, eeg_data: Dict[str, np.ndarray]
    ) -> Tuple[float, float]:
        """
        Calculate valence and arousal using heuristic methods based on
        established neuroscience research.

        Valence Calculation:
        - Based on frontal alpha asymmetry (Davidson's model)
        - Positive valence: left hemisphere alpha suppression (more activation)
        - Negative valence: right hemisphere alpha suppression (more activation)

        Arousal Calculation:
        - Based on beta/alpha ratio across frontal regions
        - Higher beta activity and lower alpha activity indicate higher arousal

        Args:
            eeg_data: Dictionary with channel names as keys and EEG signals as values

        Returns:
            Tuple containing (valence, arousal) values normalized to [1, 9] scale
        """
        # Extract features for heuristic calculation
        psd_features = self.extract_power_spectral_density(eeg_data)

        # Calculate valence using frontal alpha asymmetry
        valence = self._calculate_heuristic_valence(psd_features)

        # Calculate arousal using beta/alpha ratio
        arousal = self._calculate_heuristic_arousal(psd_features)

        return valence, arousal

    def _calculate_heuristic_valence(
        self, psd_features: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Calculate valence using frontal alpha asymmetry.

        Formula: Valence = log(F4_alpha) - log(F3_alpha)
        Positive values indicate positive valence (approach motivation)
        Negative values indicate negative valence (withdrawal motivation)
        """
        # Get alpha power for F3 and F4
        f3_alpha = psd_features.get("F3", {}).get("alpha", 1e-10)
        f4_alpha = psd_features.get("F4", {}).get("alpha", 1e-10)

        # Calculate asymmetry
        if f3_alpha > 0 and f4_alpha > 0:
            asymmetry = np.log(f4_alpha) - np.log(f3_alpha)
        else:
            asymmetry = 0.0

        # Normalize to [1, 9] scale (SAM scale)
        # Positive asymmetry -> higher valence
        valence = 5.0 + 2.0 * np.tanh(asymmetry)  # Maps to approximately [1, 9]
        return float(np.clip(valence, 1.0, 9.0))

    def _calculate_heuristic_arousal(
        self, psd_features: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Calculate arousal using beta/alpha ratio across frontal channels.

        Higher beta activity and lower alpha activity indicate higher arousal.
        """
        channels = ["F3", "F4", "AF3", "AF4"]
        beta_alpha_ratios = []

        for channel in channels:
            if channel in psd_features:
                alpha = psd_features[channel].get("alpha", 1e-10)
                beta = psd_features[channel].get("beta", 1e-10)

                if alpha > 0:
                    ratio = beta / alpha
                    beta_alpha_ratios.append(ratio)

        if beta_alpha_ratios:
            mean_ratio = np.mean(beta_alpha_ratios)
            # Normalize to [1, 9] scale
            arousal = 1.0 + 8.0 * (1 / (1 + np.exp(-np.log(mean_ratio))))
        else:
            arousal = 5.0  # Default middle value

        return float(np.clip(arousal, 1.0, 9.0))

    def prepare_features_for_ml(
        self, features_list: List[Dict[str, float]]
    ) -> np.ndarray:
        """
        Prepare feature vectors for machine learning models.

        Args:
            features_list: List of feature dictionaries

        Returns:
            Standardized feature matrix
        """
        if not features_list:
            return np.array([])

        # Convert to DataFrame for easier handling
        df = pd.DataFrame(features_list)

        # Fill NaN values with median
        df = df.fillna(df.median())

        # Fit scaler if not already fitted
        if not hasattr(self.scaler, "mean_"):
            self.scaler.fit(df.values)

        # Transform features
        return self.scaler.transform(df.values)

    def train_ml_models(
        self,
        features_list: List[Dict[str, float]],
        valence_labels: List[float],
        arousal_labels: List[float],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Train SVM models for valence and arousal prediction.

        Args:
            features_list: List of feature dictionaries
            valence_labels: List of valence values
            arousal_labels: List of arousal values
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing model performance metrics
        """
        # Prepare features
        X = self.prepare_features_for_ml(features_list)

        if X.size == 0:
            raise ValueError("No features available for training")

        # Convert continuous values to binary classification
        # High/Low valence and arousal (median split)
        valence_binary = np.array(valence_labels) > np.median(valence_labels)
        arousal_binary = np.array(arousal_labels) > np.median(arousal_labels)

        # Split data
        X_train, X_test, val_train, val_test, ar_train, ar_test = train_test_split(
            X,
            valence_binary,
            arousal_binary,
            test_size=test_size,
            random_state=random_state,
            stratify=valence_binary,
        )

        # Train valence model
        self.valence_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
        self.valence_model.fit(X_train, val_train)

        # Train arousal model
        self.arousal_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
        self.arousal_model.fit(X_train, ar_train)

        # Evaluate models
        val_pred = self.valence_model.predict(X_test)
        ar_pred = self.arousal_model.predict(X_test)

        return {
            "valence_accuracy": float(accuracy_score(val_test, val_pred)),
            "arousal_accuracy": float(accuracy_score(ar_test, ar_pred)),
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
        }

    def predict_valence_arousal_ml(
        self, eeg_data: Dict[str, np.ndarray]
    ) -> Tuple[float, float]:
        """
        Predict valence and arousal using trained ML models.

        Args:
            eeg_data: Dictionary with channel names as keys and EEG signals as values

        Returns:
            Tuple containing (valence, arousal) predictions
        """
        if self.valence_model is None or self.arousal_model is None:
            raise ValueError("Models not trained. Call train_ml_models() first.")

        # Extract features
        features = self.extract_all_features(eeg_data)
        X = self.prepare_features_for_ml([features])

        # Get predictions (probabilities for high class)
        valence_prob = self.valence_model.predict_proba(X)[0, 1]
        arousal_prob = self.arousal_model.predict_proba(X)[0, 1]

        # Convert probabilities to [1, 9] scale
        valence = 1.0 + 8.0 * valence_prob
        arousal = 1.0 + 8.0 * arousal_prob

        return float(valence), float(arousal)

    def calculate_valence_arousal(
        self, eeg_data: Dict[str, np.ndarray], method: str = "heuristic"
    ) -> Tuple[float, float]:
        """
        Calculate valence and arousal using the specified method.

        Args:
            eeg_data: Dictionary with channel names as keys and EEG signals as values
            method: Calculation method ('heuristic' or 'ml')

        Returns:
            Tuple containing (valence, arousal) values on [1, 9] scale
        """
        if method == "heuristic":
            return self.calculate_valence_arousal_heuristic(eeg_data)
        elif method == "ml":
            return self.predict_valence_arousal_ml(eeg_data)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'heuristic' or 'ml'.")
