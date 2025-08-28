"""
Apply Heuristic Metric to All EEG Image Evaluations

This endpoint applies the EEG heuristic metric to all EEGImageEvaluation instances
in the database. It calculates the metric for each image,
updates the metric_heurist field, and returns a summary.

Follows clean architecture and reproducibility standards.
"""

from collections import defaultdict
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from src.app.db.db_session import get_session
from src.app.models import EEGImageEvaluation
from src.modules.EEG.metrics.heuristic_metric import EEGHeuristicMetric

router = APIRouter()


@router.post(
    path="/heuristic/all/",
    summary="Apply heuristic metric to all EEG image evaluations",
    description=(
        "Calculates the EEG heuristic metric for all EEGImageEvaluation instances "
        "in the database, updates each metric_heurist field, "
        "and returns a summary of the results."
    ),
)
async def apply_heuristic_metric_all(
    db: Annotated[Session, Depends(get_session)],
) -> dict:
    """
    Applies heuristic metric to all EEG image evaluations.

    Args:
        db: Database session dependency.

    Returns:
        Dictionary summarizing results per image evaluation, grouped by user.
    """
    try:
        # Retrieve all image evaluations
        image_evaluations = db.exec(select(EEGImageEvaluation)).all()

        if not image_evaluations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No EEG image evaluations found.",
            )

        metric_calculator = EEGHeuristicMetric()
        processed_by_user = defaultdict(list)
        skipped_by_user = defaultdict(list)
        errors_by_user = defaultdict(list)
        user_stats = defaultdict(
            lambda: {
                "total_images": 0,
                "successfully_processed": 0,
                "skipped": 0,
                "errors": 0,
                "average_metric": 0,
                "average_valence": 0,
                "average_arousal": 0,
            }
        )

        for image_eval in image_evaluations:
            user_id = image_eval.user_id
            user_stats[user_id]["total_images"] += 1

            try:
                eeg_path = Path(image_eval.eeg_file_path)
                if not eeg_path.exists():
                    skipped_by_user[user_id].append(
                        {
                            "image_evaluation_id": image_eval.id,
                            "reason": "EEG file not found",
                            "file_path": str(eeg_path),
                        }
                    )
                    user_stats[user_id]["skipped"] += 1
                    continue

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
                    missing = required_cols - set(df.columns)
                    skipped_by_user[user_id].append(
                        {
                            "image_evaluation_id": image_eval.id,
                            "reason": "Missing required columns",
                            "missing_columns": list(missing),
                        }
                    )
                    user_stats[user_id]["skipped"] += 1
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

                processed_by_user[user_id].append(
                    {
                        "image_evaluation_id": image_eval.id,
                        "evaluation_id": image_eval.evaluation_id,
                        "image_id": image_eval.image_id,
                        "heuristic_metric": metric_result,
                        "valence": sam_valence,
                        "arousal": sam_arousal,
                        "valence_eeg": valence_eeg,
                        "arousal_eeg": arousal_eeg,
                        "file_path": str(eeg_path),
                    }
                )
                user_stats[user_id]["successfully_processed"] += 1
            except Exception as e:
                errors_by_user[user_id].append(
                    {"image_evaluation_id": image_eval.id, "error": str(e)}
                )
                user_stats[user_id]["errors"] += 1

        # Calculate averages for each user
        for user_id in user_stats:
            processed = processed_by_user[user_id]
            if processed:
                user_stats[user_id]["average_metric"] = sum(
                    p["heuristic_metric"] for p in processed
                ) / len(processed)
                user_stats[user_id]["average_valence"] = sum(
                    p["valence"] for p in processed
                ) / len(processed)
                user_stats[user_id]["average_arousal"] = sum(
                    p["arousal"] for p in processed
                ) / len(processed)

        db.commit()

        # Calculate global statistics
        total_processed = sum(
            len(processed) for processed in processed_by_user.values()
        )
        total_skipped = sum(len(skipped) for skipped in skipped_by_user.values())
        total_errors = sum(len(errors) for errors in errors_by_user.values())
        total_images = sum(stats["total_images"] for stats in user_stats.values())

        return {
            "results_by_user": {
                str(user_id): {
                    "processed_images": processed_by_user[user_id],
                    "skipped_images": skipped_by_user[user_id],
                    "errors": errors_by_user[user_id],
                    "statistics": user_stats[user_id],
                }
                for user_id in user_stats
            },
            "global_summary": {
                "total_images": total_images,
                "successfully_processed": total_processed,
                "skipped": total_skipped,
                "errors": total_errors,
                "average_metric": sum(
                    stats["average_metric"] * stats["successfully_processed"]
                    for stats in user_stats.values()
                )
                / total_processed
                if total_processed > 0
                else 0,
            },
            "message": f"Heuristic metric computed for {total_processed} images across {len(user_stats)} users. "
            f"{total_skipped} skipped, {total_errors} errors.",
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
