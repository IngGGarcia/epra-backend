"""
Is First Evaluation Service

This service checks if a user has already evaluated an image.
Implements dependency injection and SQLModel-based queries.
"""

from typing import Annotated

from fastapi import Depends
from sqlmodel import Session, select

from src.app.db import get_session
from src.app.models.evaluation import Evaluation


def is_first_evaluation(
    user_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> bool:
    """
    Check if the user has already evaluated the image.

    Args:
        user_id: The ID of the user to check.
        session: Database session dependency.

    Returns:
        bool: True if this is the first evaluation, False otherwise.
    """
    evaluation = session.exec(
        select(Evaluation).where(Evaluation.user_id == user_id)
    ).first()
    if evaluation:
        return False
    return True
