"""
EEG Evaluation Creation Endpoint with File Upload

Provides a POST endpoint for creating new EEG evaluations, including:
- EEG file
- Test SAM file

Implements dependency injection, file storage, and SQLModel-based database interaction.
Ensures reproducibility, atomicity, and robust error handling aligned with scientific standards.
"""

from datetime import UTC, date, datetime
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from src.app.db.db_session import get_session
from src.app.models.epra_evaluation import (
    EEGEvaluation,
    EEGEvaluationCreate,
    EEGEvaluationRead,
)
from src.modules.storage import StorageManager

# Configure logging
logging.basicConfig(level=logging.INFO)

router = APIRouter()

# Initialize storage manager
storage_manager = StorageManager(sub_storage_path="eeg")


@router.post(
    path="/",
    response_model=EEGEvaluationRead,
    summary="Create a new EEG evaluation with file upload",
    description="Creates a new EEG evaluation and uploads EEG and Test SAM CSV files.",
)
async def create_evaluation(
    user_id: Annotated[int, Form(description="User identifier")],
    eeg_file: Annotated[UploadFile, File(description="EEG CSV file")],
    test_sam_file: Annotated[UploadFile, File(description="Test SAM CSV file")],
    db: Annotated[Session, Depends(get_session)],
    evaluation_date: Annotated[
        date | None, Form(description="Date in YYYY-MM-DD format")
    ] = None,
) -> EEGEvaluationRead:
    """
    Creates a new EEG evaluation in the database and stores EEG and Test SAM files.

    Args:
        user_id: User ID for the EEG evaluation.
        evaluation_date: Optional evaluation date in YYYY-MM-DD format.
        eeg_file: EEG CSV file.
        test_sam_file: Test SAM CSV file.
        db: Database session dependency.

    Returns:
        The created EEG evaluation in read format.
    """
    try:
        # Default to current date if not provided
        final_date = evaluation_date or datetime.now(UTC).date()

        # Store EEG and Test SAM files
        stored_files = storage_manager.store_files(
            user_id,
            {"eeg_data.csv": eeg_file, "test_sam.csv": test_sam_file},
            final_date.isoformat(),
        )

        # Create evaluation data object
        evaluation_data = EEGEvaluationCreate(
            user_id=user_id,
            evaluation_date=final_date,
            eeg_file_path=stored_files["eeg_data.csv"],
            test_sam_file_path=stored_files["test_sam.csv"],
        )

        # Validate and persist evaluation
        new_evaluation = EEGEvaluation.model_validate(evaluation_data)
        db.add(new_evaluation)
        db.commit()
        db.refresh(new_evaluation)

        return new_evaluation

    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error creating EEG evaluation: {str(e)}",
        ) from e

    except Exception as e:
        db.rollback()
        logging.error(f"Error uploading EEG evaluation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from e
