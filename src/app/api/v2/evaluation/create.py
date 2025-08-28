"""
Create Evaluation API

This module handles the creation of evaluations, including the storage of associated CSV files.
"""

from datetime import datetime
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlmodel import Session

from src.app.clients.storage import LocalStorageClient
from src.app.db import get_session
from src.app.models.evaluation import Evaluation, EvaluationRead
from src.app.models.image_evaluation import ImageEvaluation
from src.app.services.evaluation import verify_eeg_file, verify_sam_file

router = APIRouter()
storage_client = LocalStorageClient()
logger = logging.getLogger(__name__)


@router.post("/create")
async def create_evaluation(
    user_id: Annotated[int, Form()],
    eeg_file: Annotated[UploadFile, File(..., description="EEG CSV file")],
    sam_file: Annotated[UploadFile, File(..., description="SAM CSV file")],
    session: Annotated[Session, Depends(get_session)],
) -> EvaluationRead:
    """
    Create a new evaluation with associated CSV files.

    Args:
        user_id: The ID of the user creating the evaluation
        eeg_file: EEG CSV file to be stored
        sam_file: SAM CSV file to be stored
        session: Database session

    Returns:
        The created evaluation with updated file paths
    """
    # Verify the EEG file
    eeg_file_data = verify_eeg_file(eeg_file.file, eeg_file.filename)

    # Verify the SAM file
    sam_file_data = verify_sam_file(sam_file.file, sam_file.filename)

    # Create evaluation record first to get the UUID
    evaluation = Evaluation(user_id=user_id)
    session.add(evaluation)
    session.commit()
    session.refresh(evaluation)

    # Store the CSV files with evaluation UUID in the path
    eeg_file_path = storage_client.upload(
        eeg_file.file, f"evaluations/{user_id}/{evaluation.id}/eeg.csv"
    )
    sam_file_path = storage_client.upload(
        sam_file.file, f"evaluations/{user_id}/{evaluation.id}/sam.csv"
    )

    # Update evaluation with file paths
    evaluation.eeg_file_path = eeg_file_path
    evaluation.sam_file_path = sam_file_path
    session.add(evaluation)
    session.commit()
    session.refresh(evaluation)

    # Create image evaluations - combine EEG and SAM data
    # EEG data has: image_number (sequential order), valence, arousal
    # SAM data has: image_id (actual image ID), valence, arousal
    # We match by position: first EEG image with first SAM row, second with second, etc.

    image_evaluations_created = 0

    # Sort both DataFrames to ensure consistent order
    eeg_file_data_sorted = eeg_file_data.sort_values("image_number").reset_index(
        drop=True
    )
    sam_file_data_sorted = sam_file_data.reset_index(drop=True)

    # Match by position/index
    for index in range(min(len(eeg_file_data_sorted), len(sam_file_data_sorted))):
        eeg_row = eeg_file_data_sorted.iloc[index]
        sam_row = sam_file_data_sorted.iloc[index]

        actual_image_id = sam_row["image_id"]  # This is the real image ID from SAM

        try:
            # Create ImageEvaluation record using the actual image_id from SAM
            image_evaluation = ImageEvaluation(
                evaluation_id=evaluation.id,
                user_id=user_id,
                image_id=int(actual_image_id),  # Use the real image ID from SAM
                sam_valence=int(sam_row["valence"]),
                sam_arousal=int(sam_row["arousal"]),
                eeg_valence=float(eeg_row["valence"]),
                eeg_arousal=float(eeg_row["arousal"]),
            )

            session.add(image_evaluation)
            image_evaluations_created += 1

        except Exception as e:
            logger.error(
                f"Error creating ImageEvaluation at position {index + 1}: {str(e)}"
            )
            raise

    # Commit all image evaluations
    session.commit()

    return evaluation
