"""
Pre-Create Evaluation API

This endpoint checks if a user has already evaluated an image.
Implements dependency injection and SQLModel-based queries.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session

from src.app.db import get_session
from src.app.services.evaluation import is_first_evaluation
from src.app.services.images import get_best_images, get_random_images

router = APIRouter()

LIMIT_IMAGES = 5


@router.get("/pre-create")
async def pre_create_evaluation(
    user_id: Annotated[int, Query()],
    session: Annotated[Session, Depends(get_session)],
    limit: Annotated[int, Query(ge=LIMIT_IMAGES)] = LIMIT_IMAGES,
) -> list[int]:
    """
    Check if this is the first evaluation for a user.

    Args:
        user_id: The ID of the user to check.
        session: Database session dependency.

    Returns:
        bool: True if this is the first evaluation, False otherwise.
    """
    if is_first_evaluation(user_id, session):
        images = get_random_images(session, limit)
    else:
        images = get_best_images(
            session=session,
            user_id=user_id,
            limit=limit,
        )
    return images
