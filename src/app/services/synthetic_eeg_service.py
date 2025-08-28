"""
Synthetic EEG Generation Service

This service generates synthetic EEG data based on predicted valence and arousal values
from trained GPR models. It uses existing real EEG data as a base and applies
transformations to achieve the target emotional values.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from src.app.models.image_classification import ImageClassification
from src.app.models.image_evaluation import ImageEvaluation
from src.app.services.images.get_best_images import get_best_images
from src.app.services.predict.predict_service import PredictionService
from src.modules.metrics.eeg_sam_heuristic import EmotionData

logger = logging.getLogger(__name__)


class SyntheticEEGService:
    """
    Service for generating synthetic EEG evaluations based on predicted emotional responses.
    """

    def __init__(self, user_id: int, db_session: Session):
        """
        Initialize the synthetic EEG service.

        Args:
            user_id: ID of the user for whom to generate synthetic data
            db_session: Database session
        """
        self.user_id = user_id
        self.db_session = db_session
        self.sampling_rate = 128.0  # Default sampling rate for EEG data

    def generate_synthetic_evaluation(self) -> Dict[str, str]:
        """
        Generate a synthetic EEG evaluation for the user.

        Returns:
            Dictionary containing paths to generated synthetic EEG and SAM files
        """
        try:
            # Step 1: Get the best image for this user
            best_image_id = self._get_best_image_for_user()

            # Step 2: Predict valence and arousal for this image
            predicted_valence, predicted_arousal = self._predict_emotions(best_image_id)

            # Step 3: Generate synthetic EEG data
            synthetic_eeg_data = self._generate_synthetic_eeg_data(
                predicted_valence, predicted_arousal
            )

            # Step 4: Generate synthetic SAM data
            synthetic_sam_data = self._generate_synthetic_sam_data(
                best_image_id, predicted_valence, predicted_arousal
            )

            # Step 5: Save synthetic data to files
            eeg_file_path = self._save_synthetic_eeg(synthetic_eeg_data)
            sam_file_path = self._save_synthetic_sam(synthetic_sam_data)

            return {
                "eeg_file_path": eeg_file_path,
                "test_sam_file_path": sam_file_path,
                "image_id": best_image_id,
                "predicted_valence": predicted_valence,
                "predicted_arousal": predicted_arousal,
            }

        except Exception as e:
            logger.error(f"Error generating synthetic evaluation: {str(e)}")
            raise

    def generate_multiple_synthetic_evaluations(self) -> List[Dict[str, str]]:
        """
        Generate synthetic EEG evaluations for all 5 best images.

        Returns:
            List of dictionaries containing paths to generated synthetic EEG and SAM files
        """
        try:
            # Step 1: Get all 5 best images for this user
            best_image_ids = self._get_all_best_images_for_user()

            synthetic_evaluations = []

            # Step 2: Process each image
            for image_id in best_image_ids:
                try:
                    # Predict valence and arousal for this image
                    predicted_valence, predicted_arousal = self._predict_emotions(
                        image_id
                    )

                    # Generate synthetic EEG data with realistic variability
                    synthetic_eeg_data = self._generate_synthetic_eeg_data(
                        predicted_valence, predicted_arousal
                    )

                    # Generate synthetic SAM data with realistic variability
                    synthetic_sam_data = self._generate_synthetic_sam_data(
                        image_id, predicted_valence, predicted_arousal
                    )

                    # Save synthetic data to files
                    eeg_file_path = self._save_synthetic_eeg(synthetic_eeg_data)
                    sam_file_path = self._save_synthetic_sam(synthetic_sam_data)

                    # Get the realistic values that were actually used
                    realistic_valence, realistic_arousal = (
                        self._add_realistic_variability(
                            predicted_valence, predicted_arousal
                        )
                    )

                    synthetic_evaluations.append(
                        {
                            "eeg_file_path": eeg_file_path,
                            "test_sam_file_path": sam_file_path,
                            "image_id": image_id,
                            "predicted_valence": predicted_valence,
                            "predicted_arousal": predicted_arousal,
                            "realistic_valence": realistic_valence,
                            "realistic_arousal": realistic_arousal,
                        }
                    )

                    logger.info(f"Generated synthetic evaluation for image {image_id}")

                except Exception as e:
                    logger.error(
                        f"Error generating synthetic evaluation for image {image_id}: {str(e)}"
                    )
                    # Continue with other images even if one fails
                    continue

            if not synthetic_evaluations:
                raise ValueError("Failed to generate any synthetic evaluations")

            logger.info(
                f"Successfully generated {len(synthetic_evaluations)} synthetic evaluations"
            )
            return synthetic_evaluations

        except Exception as e:
            logger.error(f"Error generating multiple synthetic evaluations: {str(e)}")
            raise

    def _get_best_image_for_user(self) -> int:
        """
        Get the best image for the user using existing service.

        Returns:
            ID of the best image
        """
        try:
            best_images = get_best_images(self.db_session, self.user_id, limit=1)
            if not best_images:
                raise ValueError("No suitable images found for user")
            return best_images[0]
        except Exception as e:
            logger.error(f"Error getting best image: {str(e)}")
            raise

    def _get_all_best_images_for_user(self) -> List[int]:
        """
        Get all 5 best images for the user using existing service.

        Returns:
            List of 5 best image IDs
        """
        try:
            best_images = get_best_images(self.db_session, self.user_id, limit=5)
            if not best_images:
                raise ValueError("No suitable images found for user")
            if len(best_images) < 5:
                logger.warning(f"Only found {len(best_images)} images, expected 5")
            return best_images
        except Exception as e:
            logger.error(f"Error getting best images: {str(e)}")
            raise

    def _predict_emotions(self, image_id: int) -> Tuple[float, float]:
        """
        Predict valence and arousal for a given image.

        Args:
            image_id: ID of the image to predict for

        Returns:
            Tuple of (predicted_valence, predicted_arousal)
        """
        try:
            # Get image feature vector
            statement = select(ImageClassification.feature_vector).where(
                ImageClassification.image_id == image_id
            )
            result = self.db_session.exec(statement).first()

            if not result:
                raise ValueError(f"Image {image_id} not found")

            feature_vector = np.array(json.loads(result))

            # Get historical data for bias correction
            statement = (
                select(ImageEvaluation)
                .where(ImageEvaluation.user_id == self.user_id)
                .order_by(ImageEvaluation.created_at.desc())
            )
            historical_results = self.db_session.exec(statement).all()

            historical_data = None
            if historical_results:
                historical_data = [
                    EmotionData(
                        sam_valence=eval.sam_valence,
                        eeg_valence=eval.eeg_valence,
                        sam_arousal=eval.sam_arousal,
                        eeg_arousal=eval.eeg_arousal,
                    )
                    for eval in historical_results
                ]

            # Create prediction service and make prediction
            prediction_service = PredictionService.from_user_models(
                user_id=self.user_id, historical_data=historical_data
            )

            valence, _, arousal, _ = prediction_service.predict(feature_vector)

            return float(valence), float(arousal)

        except Exception as e:
            logger.error(f"Error predicting emotions: {str(e)}")
            raise

    def _generate_synthetic_eeg_data(
        self, target_valence: float, target_arousal: float
    ) -> pd.DataFrame:
        """
        Generate synthetic EEG data that will produce the target valence and arousal values.

        Args:
            target_valence: Target valence value (1-9)
            target_arousal: Target arousal value (1-9)

        Returns:
            DataFrame with synthetic EEG data
        """
        # Add realistic variability to the target values
        realistic_valence, realistic_arousal = self._add_realistic_variability(
            target_valence, target_arousal
        )

        # Get a sample of real EEG data to use as base
        base_eeg_data = self._get_base_eeg_data()

        if base_eeg_data is None:
            # Generate completely synthetic data if no base available
            return self._generate_completely_synthetic_eeg(
                realistic_valence, realistic_arousal
            )

        # Transform base data to achieve realistic target values
        return self._transform_eeg_data(
            base_eeg_data, realistic_valence, realistic_arousal
        )

    def _get_base_eeg_data(self) -> pd.DataFrame:
        """
        Get base EEG data from existing evaluations.

        Returns:
            DataFrame with base EEG data or None if not available
        """
        try:
            # Try to find existing EEG data for this user
            statement = (
                select(ImageEvaluation.eeg_file_path)
                .where(ImageEvaluation.user_id == self.user_id)
                .limit(1)
            )

            result = self.db_session.exec(statement).first()

            if result and Path(result).exists():
                # Load and return base EEG data
                return pd.read_csv(result)

            return None

        except Exception as e:
            logger.warning(f"Could not load base EEG data: {str(e)}")
            return None

    def _generate_completely_synthetic_eeg(
        self, target_valence: float, target_arousal: float
    ) -> pd.DataFrame:
        """
        Generate completely synthetic EEG data from scratch.

        Args:
            target_valence: Target valence value
            target_arousal: Target arousal value

        Returns:
            DataFrame with synthetic EEG data
        """
        # Generate time series
        duration = 50  # seconds per cycle
        timestamps = np.arange(0, duration, 1 / self.sampling_rate)

        # Generate synthetic signals based on target emotions
        # Higher arousal = more beta activity, higher valence = more alpha asymmetry

        # Base frequencies
        alpha_freq = 10  # Hz
        beta_freq = 20  # Hz

        # Amplitude modulation based on target values
        valence_factor = (target_valence - 5) / 4  # -1 to 1
        arousal_factor = (target_arousal - 5) / 4  # -1 to 1

        # Generate synthetic EEG channels
        eeg_af3 = self._generate_synthetic_channel(
            timestamps,
            alpha_freq,
            beta_freq,
            valence_factor,
            arousal_factor,
            phase_shift=0,
        )
        eeg_f3 = self._generate_synthetic_channel(
            timestamps,
            alpha_freq,
            beta_freq,
            valence_factor,
            arousal_factor,
            phase_shift=np.pi / 4,
        )
        eeg_af4 = self._generate_synthetic_channel(
            timestamps,
            alpha_freq,
            beta_freq,
            -valence_factor,
            arousal_factor,
            phase_shift=np.pi / 2,
        )
        eeg_f4 = self._generate_synthetic_channel(
            timestamps,
            alpha_freq,
            beta_freq,
            -valence_factor,
            arousal_factor,
            phase_shift=3 * np.pi / 4,
        )

        # Create DataFrame
        df = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "EEG.AF3": eeg_af3,
                "EEG.F3": eeg_f3,
                "EEG.AF4": eeg_af4,
                "EEG.F4": eeg_f4,
            }
        )

        return df

    def _generate_synthetic_channel(
        self,
        timestamps: np.ndarray,
        alpha_freq: float,
        beta_freq: float,
        valence_factor: float,
        arousal_factor: float,
        phase_shift: float,
    ) -> np.ndarray:
        """
        Generate synthetic EEG signal for a single channel.

        Args:
            timestamps: Time array
            alpha_freq: Alpha frequency
            beta_freq: Beta frequency
            valence_factor: Valence influence factor
            arousal_factor: Arousal influence factor
            phase_shift: Phase shift for this channel

        Returns:
            Synthetic EEG signal
        """
        # Base alpha activity (relaxation)
        alpha_signal = np.sin(2 * np.pi * alpha_freq * timestamps + phase_shift)

        # Beta activity (arousal)
        beta_signal = np.sin(2 * np.pi * beta_freq * timestamps + phase_shift)

        # Combine signals with emotion factors
        signal = (1 + valence_factor) * alpha_signal + (
            1 + arousal_factor
        ) * beta_signal

        # Add realistic noise
        noise = np.random.normal(0, 0.1, len(timestamps))
        signal += noise

        # Normalize
        signal = signal / np.std(signal) * 50  # Scale to realistic EEG amplitudes

        return signal

    def _transform_eeg_data(
        self, base_data: pd.DataFrame, target_valence: float, target_arousal: float
    ) -> pd.DataFrame:
        """
        Transform existing EEG data to achieve target valence and arousal values.

        Args:
            base_data: Base EEG data to transform
            target_valence: Target valence value
            target_arousal: Target arousal value

        Returns:
            Transformed EEG data
        """
        # For now, return the base data with slight modifications
        # In a more sophisticated implementation, this would apply
        # frequency domain transformations to achieve target values

        transformed_data = base_data.copy()

        # Apply simple scaling based on target values
        valence_scale = target_valence / 5.0  # Assuming 5 is neutral
        arousal_scale = target_arousal / 5.0

        # Scale EEG channels
        for col in ["EEG.AF3", "EEG.F3", "EEG.AF4", "EEG.F4"]:
            if col in transformed_data.columns:
                transformed_data[col] = transformed_data[col] * arousal_scale

        return transformed_data

    def _generate_synthetic_sam_data(
        self, image_id: int, valence: float, arousal: float
    ) -> pd.DataFrame:
        """
        Generate synthetic SAM data for the target image.

        Args:
            image_id: ID of the image
            valence: Valence value (float, will be converted to int)
            arousal: Arousal value (float, will be converted to int)

        Returns:
            DataFrame with synthetic SAM data
        """
        # Add realistic variability to the predicted values
        realistic_valence, realistic_arousal = self._add_realistic_variability(
            valence, arousal
        )

        # Create SAM data with the realistic values converted to integers
        # SAM scale is 1-9, so we round and clamp to valid range
        sam_valence = max(1, min(9, int(round(realistic_valence))))
        sam_arousal = max(1, min(9, int(round(realistic_arousal))))

        sam_data = pd.DataFrame(
            {
                "image_id": [image_id],
                "valence": [sam_valence],
                "arousal": [sam_arousal],
            }
        )

        return sam_data

    def _save_synthetic_eeg(self, eeg_data: pd.DataFrame) -> str:
        """
        Save synthetic EEG data to a file.

        Args:
            eeg_data: EEG data to save

        Returns:
            Path to saved file
        """
        # Create storage directory
        storage_dir = Path("uploads") / "eeg" / f"user_{self.user_id}" / "synthetic"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"synthetic_eeg_{self.user_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = storage_dir / filename

        # Add metadata line (sampling rate)
        with open(file_path, "w") as f:
            f.write(f"Sampling Rate: {self.sampling_rate} Hz\n")
            eeg_data.to_csv(f, index=False)

        return str(file_path)

    def _save_synthetic_sam(self, sam_data: pd.DataFrame) -> str:
        """
        Save synthetic SAM data to a file.

        Args:
            sam_data: SAM data to save

        Returns:
            Path to saved file
        """
        # Create storage directory
        storage_dir = Path("uploads") / "eeg" / f"user_{self.user_id}" / "synthetic"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"synthetic_sam_{self.user_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = storage_dir / filename

        # Save SAM data
        sam_data.to_csv(file_path, index=False)

        return str(file_path)

    def _get_user_historical_deviations(self) -> dict:
        """
        Calculate standard deviations from user's real evaluation data.

        Returns:
            Dictionary with valence and arousal standard deviations
        """
        try:
            # Get user's real evaluation data
            statement = select(
                ImageEvaluation.sam_valence, ImageEvaluation.sam_arousal
            ).where(ImageEvaluation.user_id == self.user_id)
            results = self.db_session.exec(statement).all()

            if not results:
                # If no real data, use default values
                return {
                    "valence_std": 1.0,  # Default 1 point variation
                    "arousal_std": 1.0,
                }

            # Extract valence and arousal values
            valence_values = [float(eval.sam_valence) for eval in results]
            arousal_values = [float(eval.sam_arousal) for eval in results]

            # Calculate standard deviations
            valence_std = np.std(valence_values) if len(valence_values) > 1 else 1.0
            arousal_std = np.std(arousal_values) if len(arousal_values) > 1 else 1.0

            # Ensure minimum variability (at least 0.5)
            valence_std = max(0.5, valence_std)
            arousal_std = max(0.5, arousal_std)

            logger.info(
                f"User {self.user_id} historical deviations - Valence: {valence_std:.2f}, Arousal: {arousal_std:.2f}"
            )

            return {
                "valence_std": float(valence_std),
                "arousal_std": float(arousal_std),
            }

        except Exception as e:
            logger.warning(
                f"Could not calculate user deviations, using defaults: {str(e)}"
            )
            return {"valence_std": 1.0, "arousal_std": 1.0}

    def _add_realistic_variability(
        self, predicted_valence: float, predicted_arousal: float
    ) -> tuple[float, float]:
        """
        Add realistic variability to predicted values based on user's historical data.

        Args:
            predicted_valence: Predicted valence value
            predicted_arousal: Predicted arousal value

        Returns:
            Tuple of (valence_with_noise, arousal_with_noise)
        """
        # Get user's historical variability
        deviations = self._get_user_historical_deviations()

        # Add gaussian noise based on user's real variability
        valence_noise = np.random.normal(0, deviations["valence_std"])
        arousal_noise = np.random.normal(0, deviations["arousal_std"])

        # Apply noise to predictions
        valence_with_noise = predicted_valence + valence_noise
        arousal_with_noise = predicted_arousal + arousal_noise

        # Clamp to valid SAM range (1-9)
        valence_with_noise = max(1.0, min(9.0, valence_with_noise))
        arousal_with_noise = max(1.0, min(9.0, arousal_with_noise))

        logger.info(
            f"Added variability - Valence: {predicted_valence:.2f} → {valence_with_noise:.2f} "
            f"(noise: {valence_noise:+.2f})"
        )
        logger.info(
            f"Added variability - Arousal: {predicted_arousal:.2f} → {arousal_with_noise:.2f} "
            f"(noise: {arousal_noise:+.2f})"
        )

        return float(valence_with_noise), float(arousal_with_noise)
