"""
EEG Preprocessing Endpoint with Image-Level Storage

Processes EEG evaluations by evaluation ID, extracting structured EEG segments,
aligning them with Test SAM responses, and storing each preprocessed segment
under a user-specific directory, named by image ID.

For each image, a corresponding database entry is created, capturing the user ID,
image ID, evaluation ID, valence, arousal, and the path to the clean EEG data.

This endpoint follows strict modularity, atomicity, and scientific reproducibility standards.
"""

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi import Path as FastAPIPath
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from src.app.db.db_session import get_session
from src.app.models import EEGEvaluation, EEGImageEvaluation, ProcessingStatus
from src.modules.EEG.preprocessing.EEGPreprocessor import EEGPreprocessor

router = APIRouter()
logging.basicConfig(level=logging.INFO)


@router.post(
    "/{evaluation_id}/",
    summary="Preprocess EEG evaluation and store segments per image",
    description=(
        "Processes an EEG evaluation by extracting image-level EEG segments, "
        "aligning them with Test SAM responses, storing preprocessed files "
        "per image ID under user-specific directories, and creating "
        "corresponding database records."
    ),
)
async def preprocess_eeg_evaluation(
    evaluation_id: Annotated[int, FastAPIPath(gt=0)],
    db: Annotated[Session, Depends(get_session)],
) -> dict:
    """
    Processes an existing EEG evaluation by extracting segments, aligning them with
    Test SAM data, and creating detailed records per image evaluation.

    Args:
        evaluation_id: ID of the EEG evaluation to process.
        db: Database session dependency.

    Returns:
        Dictionary summarizing the processing outcome.
    """
    try:
        # Retrieve evaluation
        evaluation = db.get(EEGEvaluation, evaluation_id)
        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"EEG evaluation with ID {evaluation_id} not found.",
            )

        if evaluation.processed == ProcessingStatus.PROCESSED:
            return {
                "evaluation_id": evaluation_id,
                "message": "Evaluation already processed.",
            }

        # Set status to processing
        evaluation.processed = ProcessingStatus.PENDING
        evaluation.error_message = None
        db.add(evaluation)
        db.commit()

        # Read Test SAM file
        test_sam_path = Path(evaluation.test_sam_file_path)
        if not test_sam_path.exists():
            evaluation.processed = ProcessingStatus.ERROR
            evaluation.error_message = (
                f"Test SAM file not found at {evaluation.test_sam_file_path}."
            )
            db.add(evaluation)
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=evaluation.error_message,
            )

        test_sam_df = pd.read_csv(test_sam_path)
        required_cols = {"image_id", "valence", "arousal"}
        if not required_cols.issubset(test_sam_df.columns):
            missing = required_cols - set(test_sam_df.columns)
            evaluation.processed = ProcessingStatus.ERROR
            evaluation.error_message = (
                f"Test SAM file missing required columns: {missing}"
            )
            db.add(evaluation)
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=evaluation.error_message,
            )

        # Process EEG file
        preprocessor = EEGPreprocessor(evaluation.eeg_file_path)
        segments = preprocessor.extract_image_segments()

        if len(segments) != len(test_sam_df):
            evaluation.processed = ProcessingStatus.ERROR
            evaluation.error_message = (
                f"Mismatch between EEG segments ({len(segments)}) "
                f"and Test SAM entries ({len(test_sam_df)})."
            )
            db.add(evaluation)
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=evaluation.error_message,
            )

        # Prepare storage base path
        user_id = evaluation.user_id
        base_preprocessed_dir = Path("uploads") / "preprocessed_eeg" / f"user_{user_id}"
        base_preprocessed_dir.mkdir(parents=True, exist_ok=True)

        # Process each image segment
        for idx, segment_df in enumerate(segments):
            test_sam_row = test_sam_df.iloc[idx]
            image_id = str(test_sam_row["image_id"])
            valence = float(test_sam_row["valence"])
            arousal = float(test_sam_row["arousal"])

            # Define file path per image
            eeg_segment_path = base_preprocessed_dir / f"image_{image_id}.csv"

            # Store EEG segment with valence/arousal columns
            segment_df["Valence"] = valence
            segment_df["Arousal"] = arousal
            segment_df.to_csv(eeg_segment_path, index=False)

            # Create record in EEGImageEvaluation table
            image_eval = EEGImageEvaluation(
                user_id=user_id,
                evaluation_id=evaluation_id,
                image_id=image_id,
                eeg_file_path=str(eeg_segment_path),
                valence=valence,
                arousal=arousal,
                metric_heurist=0.0,
                metric_v2=0.0,
            )
            db.add(image_eval)

        # Mark evaluation as processed
        evaluation.processed = ProcessingStatus.PROCESSED
        evaluation.error_message = None
        db.add(evaluation)
        db.commit()

        return {
            "evaluation_id": evaluation_id,
            "num_segments": len(segments),
            "message": "EEG evaluation processed and image-level records created successfully.",
        }

    except SQLAlchemyError as e:
        db.rollback()
        logging.error(
            f"Database error processing EEG evaluation {evaluation_id}: {str(e)}"
        )
        if evaluation:
            evaluation.processed = ProcessingStatus.ERROR
            evaluation.error_message = f"Database error: {str(e)}"
            db.add(evaluation)
            db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error processing EEG evaluation {evaluation_id}.",
        ) from e
    except Exception as e:
        db.rollback()
        logging.error(f"Error processing EEG evaluation {evaluation_id}: {str(e)}")
        if evaluation:
            evaluation.processed = ProcessingStatus.ERROR
            evaluation.error_message = f"Processing error: {str(e)}"
            db.add(evaluation)
            db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from e
