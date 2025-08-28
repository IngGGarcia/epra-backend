"""
Test EEG Metric Processing

This endpoint allows testing the EEG heuristic metric on a single file
without storing any results in the database. It's useful for testing and
validation purposes.

Follows clean architecture and reproducibility standards.
"""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
import pandas as pd

from src.modules.EEG.metrics.heuristic_metric import EEGHeuristicMetric

router = APIRouter()


@router.post(
    path="/test/",
    summary="Test EEG metric processing on a single file",
    description=(
        "Processes an EEG file and returns the computed metric without storing "
        "any results in the database. Useful for testing and validation."
    ),
)
async def test_heuristic_metric(
    file: Annotated[UploadFile, File(...)],
) -> dict:
    """
    Tests the heuristic metric on a single EEG file.

    Args:
        file: The EEG file to process.

    Returns:
        Dictionary with computed metric and validation information.
    """
    try:
        # Read the uploaded file
        df = pd.read_csv(file.file)

        # Validate required columns
        required_cols = {"EEG.F3", "EEG.F4", "EEG.AF3", "EEG.AF4", "Valence", "Arousal"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"EEG file missing required columns: {missing}.",
            )

        # Extract signals and SAM
        f3 = df["EEG.F3"].values
        f4 = df["EEG.F4"].values
        af3 = df["EEG.AF3"].values
        af4 = df["EEG.AF4"].values
        sam_valence = float(df["Valence"].iloc[0])
        sam_arousal = float(df["Arousal"].iloc[0])

        # Apply heuristic metric
        metric_calculator = EEGHeuristicMetric()
        metric_result, valence_eeg, arousal_eeg = metric_calculator.get_metric(
            f3, f4, af3, af4, sam_valence, sam_arousal
        )

        return {
            "file_name": file.filename,
            "heuristic_metric": metric_result,
            "valence_eeg": valence_eeg,
            "arousal_eeg": arousal_eeg,
            "sam_valence": sam_valence,
            "sam_arousal": sam_arousal,
            "message": "Heuristic metric computed successfully.",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}",
        ) from e
