"""
Apply Heuristic Metric to Single EEG Image Evaluation

This endpoint applies the EEG heuristic metric to a single EEGImageEvaluation instance,
calculates the metric based on preprocessed EEG data and SAM scores, updates the database,
and returns the result.

Follows clean architecture and reproducibility standards.
"""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi import Path as FastAPIPath
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from src.app.db.db_session import get_session
from src.app.models import EEGImageEvaluation
from src.modules.EEG.metrics.heuristic_metric import EEGHeuristicMetric

router = APIRouter()


@router.post(
    path="/heuristic/{image_evaluation_id}/",
    summary="Apply heuristic metric to a single EEG image evaluation",
    description=(
        "Calculates the EEG heuristic metric for a single EEGImageEvaluation instance, "
        "updates the metric_heurist field, and returns the computed metric."
    ),
)
async def apply_heuristic_metric_single(
    image_evaluation_id: Annotated[int, FastAPIPath(gt=0)],
    db: Annotated[Session, Depends(get_session)],
) -> dict:
    """
    Applies heuristic metric to a single EEG image evaluation.

    Args:
        image_evaluation_id: ID of the EEG image evaluation.
        db: Database session dependency.

    Returns:
        Dictionary with computed metric and confirmation message.
    """
    try:
        # Retrieve image evaluation
        image_eval = db.get(EEGImageEvaluation, image_evaluation_id)
        if not image_eval:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"EEG image evaluation with ID {image_evaluation_id} not found.",
            )

        # Check if EEG file exists
        eeg_path = Path(image_eval.eeg_file_path)
        if not eeg_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"EEG file not found at {image_eval.eeg_file_path}.",
            )

        # Load EEG data
        df = pd.read_csv(eeg_path)
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

        # Update database
        image_eval.metric_heurist = metric_result
        image_eval.valence_eeg = valence_eeg
        image_eval.arousal_eeg = arousal_eeg
        db.add(image_eval)
        db.commit()

        return {
            "image_evaluation_id": image_evaluation_id,
            "heuristic_metric": metric_result,
            "valence_eeg": valence_eeg,
            "arousal_eeg": arousal_eeg,
            "message": "Heuristic metric computed and updated successfully.",
        }

    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        ) from e

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        ) from e
