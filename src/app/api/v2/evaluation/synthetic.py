"""
Synthetic EEG Evaluation Endpoint V2

This endpoint generates synthetic EEG evaluations based on predicted emotional responses
from trained GPR models. It creates synthetic EEG data that, when processed, will
produce the predicted valence and arousal values.
"""

import logging
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from src.app.db import get_session
from src.app.models.evaluation import Evaluation, EvaluationRead
from src.app.models.image_evaluation import ImageEvaluation
from src.app.services.synthetic_eeg_service import SyntheticEEGService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    path="/synthetic/{user_id}",
    response_model=List[EvaluationRead],
    summary="Generate synthetic EEG evaluations",
    description="Generates 5 synthetic EEG evaluations based on predicted emotional responses from trained models.",
)
async def create_synthetic_evaluations(
    user_id: Annotated[
        int,
        Path(
            title="User ID",
            description="The ID of the user for whom to generate synthetic evaluations.",
            examples=[1],
            gt=0,
        ),
    ],
    db: Annotated[Session, Depends(get_session)],
) -> List[EvaluationRead]:
    """
    Generates 5 synthetic EEG evaluations for a specific user.

    This endpoint:
    1. Finds the 5 best images for the user using existing services
    2. Predicts valence and arousal for each image using trained GPR models
    3. Generates synthetic EEG data that will produce those predicted values
    4. Creates 5 synthetic evaluation records in the database
    5. Creates ImageEvaluation records so they appear in statistics

    Args:
        user_id: ID of the user for whom to generate synthetic evaluations
        db: Database session dependency

    Returns:
        List of the created synthetic EEG evaluations in read format

    Raises:
        HTTPException: If user has no evaluations, models not trained, or generation fails
    """
    try:
        # Initialize synthetic EEG service
        synthetic_service = SyntheticEEGService(user_id, db)

        # Generate synthetic evaluations for all 5 images
        logger.info(f"Generating 5 synthetic evaluations for user {user_id}")
        synthetic_evaluations = (
            synthetic_service.generate_multiple_synthetic_evaluations()
        )

        created_evaluations = []

        # Process each synthetic evaluation
        for synthetic_data in synthetic_evaluations:
            # Create evaluation data object
            evaluation_data = Evaluation(
                user_id=user_id,
                eeg_file_path=synthetic_data["eeg_file_path"],
                sam_file_path=synthetic_data["test_sam_file_path"],
            )

            # Validate and persist evaluation
            db.add(evaluation_data)
            db.commit()
            db.refresh(evaluation_data)

            # Create ImageEvaluation record so it appears in statistics
            image_evaluation = ImageEvaluation(
                evaluation_id=evaluation_data.id,
                user_id=user_id,
                image_id=synthetic_data["image_id"],
                sam_valence=int(round(synthetic_data["predicted_valence"])),
                sam_arousal=int(round(synthetic_data["predicted_arousal"])),
                eeg_valence=synthetic_data[
                    "predicted_valence"
                ],  # Synthetic EEG should match predictions
                eeg_arousal=synthetic_data[
                    "predicted_arousal"
                ],  # Synthetic EEG should match predictions
            )

            db.add(image_evaluation)
            db.commit()

            created_evaluations.append(evaluation_data)

            logger.info(
                f"Created synthetic evaluation {evaluation_data.id} "
                f"for user {user_id} with image {synthetic_data['image_id']}"
            )

        logger.info(
            f"Successfully created {len(created_evaluations)} synthetic evaluations "
            f"for user {user_id} with ImageEvaluation records for statistics"
        )

        return created_evaluations

    except ValueError as e:
        # Handle validation errors (no evaluations, models not trained, etc.)
        logger.warning(f"Validation error for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except SQLAlchemyError as e:
        db.rollback()
        logger.error(
            f"Database error creating synthetic evaluations for user {user_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error creating synthetic evaluations: {str(e)}",
        ) from e

    except Exception as e:
        db.rollback()
        logger.error(
            f"Unexpected error creating synthetic evaluations for user {user_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during synthetic evaluation generation.",
        ) from e
