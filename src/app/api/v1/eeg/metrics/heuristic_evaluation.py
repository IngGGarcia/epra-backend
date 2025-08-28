"""
Apply Heuristic Metric to All EEG Image Evaluations in an Evaluation

This endpoint applies the EEG heuristic metric to all EEGImageEvaluation instances
associated with a specific EEGEvaluation. It calculates the metric for each image,
updates the metric_heurist field in the database, and returns a summary.

Follows clean architecture and reproducibility standards.
"""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi import Path as FastAPIPath
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from src.app.db.db_session import get_session
from src.app.models import EEGEvaluation, EEGImageEvaluation
from src.modules.EEG.metrics.heuristic_metric import EEGHeuristicMetric

router = APIRouter()


@router.post(
    path="/heuristic/evaluation/{evaluation_id}/",
    summary="Apply heuristic metric to all EEG image evaluations in a given evaluation",
    description=(
        "Calculates the EEG heuristic metric for all EEGImageEvaluation instances "
        "associated with a specific EEGEvaluation, updates each metric_heurist field, "
        "and returns a summary of the results."
    ),
)
async def apply_heuristic_metric_evaluation(
    evaluation_id: Annotated[int, FastAPIPath(gt=0)],
    db: Annotated[Session, Depends(get_session)],
) -> dict:
    """
    Applies heuristic metric to all EEG image evaluations in a specific evaluation.

    Args:
        evaluation_id: ID of the EEG evaluation.
        db: Database session dependency.

    Returns:
        Dictionary summarizing results per image evaluation.
    """
    try:
        # Retrieve evaluation
        evaluation = db.get(EEGEvaluation, evaluation_id)
        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"EEG evaluation with ID {evaluation_id} not found.",
            )

        # Retrieve all image evaluations linked to this evaluation
        image_evaluations = db.exec(
            select(EEGImageEvaluation).where(
                EEGImageEvaluation.evaluation_id == evaluation_id
            )
        ).all()

        if not image_evaluations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No EEG image evaluations found for evaluation ID {evaluation_id}.",
            )

        metric_calculator = EEGHeuristicMetric()
        processed = []

        for image_eval in image_evaluations:
            eeg_path = Path(image_eval.eeg_file_path)
            if not eeg_path.exists():
                continue  # Optionally log or handle missing file

            df = pd.read_csv(eeg_path)
            required_cols = {
                "EEG.F3",
                "EEG.F4",
                "EEG.AF3",
                "EEG.AF4",
                "Valence",
                "Arousal",
            }
            if not required_cols.issubset(df.columns):
                continue

            f3 = df["EEG.F3"].values
            f4 = df["EEG.F4"].values
            af3 = df["EEG.AF3"].values
            af4 = df["EEG.AF4"].values
            sam_valence = float(df["Valence"].iloc[0])
            sam_arousal = float(df["Arousal"].iloc[0])

            metric_result, valence_eeg, arousal_eeg = metric_calculator.get_metric(
                f3, f4, af3, af4, sam_valence, sam_arousal
            )
            image_eval.metric_heurist = metric_result
            image_eval.valence_eeg = valence_eeg
            image_eval.arousal_eeg = arousal_eeg
            db.add(image_eval)

            processed.append(
                {
                    "image_evaluation_id": image_eval.id,
                    "heuristic_metric": metric_result,
                    "valence_eeg": valence_eeg,
                    "arousal_eeg": arousal_eeg,
                }
            )

        db.commit()

        return {
            "evaluation_id": evaluation_id,
            "processed_images": processed,
            "message": f"Heuristic metric computed for {len(processed)} images.",
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
