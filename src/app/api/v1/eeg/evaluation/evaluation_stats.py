"""
EEG Evaluation Statistics Endpoint

Provides GET endpoint to retrieve statistics about EEG evaluations:
- Total number of evaluations
- Total number of unique users
- Total number of pending evaluations

Implements dependency injection, SQLModel-based queries, and robust error handling.
Ensures reproducibility, atomicity, and clarity aligned with scientific standards.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from src.app.db.db_session import get_session
from src.app.models.epra_evaluation import EEGEvaluation

# Configure logging
logging.basicConfig(level=logging.INFO)

router = APIRouter()


@router.get(
    path="/stats",
    summary="Get evaluation statistics",
    description="Retrieves statistics about EEG evaluations including total evaluations, unique users, and pending evaluations.",
)
async def get_evaluation_stats(
    db: Annotated[Session, Depends(get_session)],
) -> dict:
    """
    Retrieves statistics about EEG evaluations.

    Args:
        db: Database session dependency.

    Returns:
        dict: A dictionary containing:
            - total_evaluations: Total number of evaluations
            - total_users: Total number of unique users
            - pending_evaluations: Total number of pending evaluations

    Raises:
        HTTPException: If there's a database error
    """
    try:
        # Get total evaluations
        total_evaluations = db.exec(select(func.count(EEGEvaluation.id))).first()

        # Get total unique users
        total_users = db.exec(
            select(func.count(func.distinct(EEGEvaluation.user_id)))
        ).first()

        # Get pending evaluations
        pending_evaluations = db.exec(
            select(func.count(EEGEvaluation.id)).where(
                EEGEvaluation.processed != "processed"
            )
        ).first()

        return {
            "total_evaluations": total_evaluations or 0,
            "total_users": total_users or 0,
            "pending_evaluations": pending_evaluations or 0,
        }

    except SQLAlchemyError as e:
        logging.error(f"Database error while retrieving evaluation stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving evaluation statistics",
        )
