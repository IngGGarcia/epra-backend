import numpy as np
import pywt
from scipy.signal import butter, lfilter


class EEGFeatureExtractor:
    """
    Standard EEG feature extractor based on established methodologies.

    This class applies predefined signal filtering, wavelet decomposition using Db4,
    and computes excitation and valence using standard electrode-based formulas.

    Attributes:
        sampling_rate (float): EEG signal sampling rate in Hz.
        wavelet (str): Wavelet type used for signal decomposition.
        lowcut (float): Lower bound frequency for bandpass filtering.
        highcut (float): Upper bound frequency for bandpass filtering.
        order (int): Order of the Butterworth filter.
    """

    def __init__(
        self,
        sampling_rate: float,
        wavelet: str = "db4",
        lowcut: float = 0.1,
        highcut: float = 60.0,
        order: int = 5,
    ):
        """
        Initializes the EEGFeatureExtractor with standard processing parameters.

        Parameters:
            sampling_rate (float): Sampling rate of the EEG data in Hz.
            wavelet (str, optional): Wavelet type for decomposition (default: 'db4').
            lowcut (float, optional): Lower bound for bandpass filtering (default: 0.1 Hz).
            highcut (float, optional): Upper bound for bandpass filtering (default: 60.0 Hz).
            order (int, optional): Order of the Butterworth filter (default: 5).
        """
        self.sampling_rate = sampling_rate
        self.wavelet = wavelet
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def _butter_bandpass(self):
        """
        Designs a Butterworth bandpass filter.

        Returns:
            tuple: Filter coefficients (b, a).
        """
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype="band")
        return b, a

    def apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Applies a Butterworth bandpass filter to the EEG signal.

        Parameters:
            data (np.ndarray): EEG signal array.

        Returns:
            np.ndarray: Filtered EEG signal.
        """
        b, a = self._butter_bandpass()
        return lfilter(b, a, data)

    def wavelet_decomposition(self, signal: np.ndarray, level: int = 4) -> dict:
        """
        Performs wavelet decomposition on an EEG signal.

        Parameters:
            signal (np.ndarray): EEG signal to decompose.
            level (int, optional): Number of decomposition levels (default: 4).

        Returns:
            dict: Dictionary containing decomposed wavelet coefficients.
        """
        coeffs = pywt.wavedec(signal, self.wavelet, level=level)
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
        return {bands[i]: coeffs[i] for i in range(len(bands))}

    def compute_excitation_valence(self, bands: dict) -> tuple:
        """
        Computes excitation and valence metrics based on standard formulas.

        Parameters:
            bands (dict): Dictionary containing wavelet decomposition bands.

        Returns:
            tuple: Excitation and valence values.
        """
        excitation = (
            np.mean(bands["beta"]["F3"])
            + np.mean(bands["beta"]["F4"])
            + np.mean(bands["beta"]["AF3"])
            + np.mean(bands["beta"]["AF4"])
        ) / (
            np.mean(bands["alpha"]["F3"])
            + np.mean(bands["alpha"]["F4"])
            + np.mean(bands["alpha"]["AF3"])
            + np.mean(bands["alpha"]["AF4"])
            + 1e-8
        )

        valence = np.mean(bands["alpha"]["F4"]) - np.mean(bands["alpha"]["F3"])
        return excitation, valence
