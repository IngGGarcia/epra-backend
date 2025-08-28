"""
EEG Evaluation Pre-creation Package

This package handles the creation of preliminary evaluation setups,
including image selection for the evaluation session.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlmodel import Session

from src.app.db.db_session import get_session

from .image_selector import select_evaluation_images

# Configure router
router = APIRouter()


class EvaluationPreCreate(BaseModel):
    """
    Model for preliminary evaluation creation request.
    """

    user_id: int = Field(..., description="User identifier")
    image_number: int = Field(
        ..., description="Total number of images for the evaluation", gt=0
    )
    image_negative_number: int = Field(
        ..., description="Number of negative images", ge=0
    )

    @field_validator("image_negative_number")
    @classmethod
    def validate_negative_images(cls, v: int, info) -> int:
        """
        Validates that the number of negative images is not greater than the total number of images.
        """
        if "image_number" in info.data and v > info.data["image_number"]:
            raise ValueError(
                "Number of negative images cannot be greater than total number of images"
            )
        return v


class EvaluationPreCreateResponse(BaseModel):
    """
    Model for preliminary evaluation creation response.
    """

    images: list[str] = Field(..., description="List of image IDs for the evaluation")


@router.post(
    path="/pre-create",
    summary="Create preliminary evaluation setup",
    description="Creates a preliminary evaluation setup with selected images.",
)
async def pre_create_evaluation(
    evaluation_data: EvaluationPreCreate,
    db: Annotated[Session, Depends(get_session)],
) -> EvaluationPreCreateResponse:
    """
    Creates a preliminary evaluation setup with selected images.

    Args:
        evaluation_data: Evaluation pre-creation data including user ID and image counts.
        db: Database session dependency.

    Returns:
        EvaluationPreCreateResponse containing the list of image IDs.

    Raises:
        HTTPException: If there's an error in the evaluation setup or if no images are available.
    """
    try:
        # Select images based on user history and classification rules
        selected_images = select_evaluation_images(
            db=db,
            user_id=evaluation_data.user_id,
            total_images=evaluation_data.image_number,
            negative_images=evaluation_data.image_negative_number,
        )

        if not selected_images:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No available images for evaluation. User may have viewed all images or no images match the criteria.",
            )

        return EvaluationPreCreateResponse(images=selected_images)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating preliminary evaluation: {str(e)}",
        ) from e
