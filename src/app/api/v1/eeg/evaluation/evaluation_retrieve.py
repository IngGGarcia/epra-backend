"""
EEG Session Retrieval Endpoints

Provides GET endpoints to retrieve EEG sessions from the database:
- Retrieve all EEG sessions with optional filtering by processing status.
- Retrieve all EEG sessions associated with a specific user, with optional filtering.

Implements dependency injection, SQLModel-based queries, and robust error handling.
Ensures reproducibility, atomicity, and clarity aligned with scientific standards.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from src.app.db.db_session import get_session
from src.app.models.epra_evaluation import EEGEvaluation, EEGEvaluationRead

# Configure logging
logging.basicConfig(level=logging.INFO)

router = APIRouter()


@router.get(
    path="/",
    response_model=list[EEGEvaluationRead],
    summary="Retrieve all EEG sessions",
    description="Retrieves all EEG sessions from the database, with optional filtering by processing status.",
)
async def get_sessions(
    db: Annotated[Session, Depends(get_session)],
    pending_only: Annotated[
        bool,
        Query(
            title="Pending Only",
            description="If True, returns only unprocessed EEG sessions.",
            examples=[False],
        ),
    ] = False,
) -> list[EEGEvaluationRead]:
    """
    Retrieves all EEG sessions from the database.

    Args:
        db: Database session dependency.
        pending_only: If True, returns only unprocessed sessions.

    Returns:
        List of EEG sessions in read format.
    """
    try:
        statement = select(EEGEvaluation)
        if pending_only:
            statement = statement.where(EEGEvaluation.processed.is_(False))

        sessions = db.exec(statement).all()

        logging.info(f"Retrieved {len(sessions)} EEG sessions.")
        return sessions

    except SQLAlchemyError as e:
        logging.error(f"Database error retrieving EEG sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error retrieving EEG sessions: {str(e)}",
        ) from e

    except Exception as e:
        logging.error(f"Unexpected error retrieving EEG sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from e


@router.get(
    path="/user/{user_id}",
    response_model=list[EEGEvaluationRead],
    summary="Retrieve user's EEG sessions",
    description="Retrieves all EEG sessions for a specific user, with optional filtering by processing status.",
)
async def get_user_sessions(
    user_id: Annotated[
        int,
        Path(
            title="User ID",
            description="The ID of the user whose EEG sessions are to be retrieved.",
            examples=[1],
            gt=0,
        ),
    ],
    db: Annotated[Session, Depends(get_session)],
    pending_only: Annotated[
        bool,
        Query(
            title="Pending Only",
            description="If True, returns only unprocessed EEG sessions.",
            examples=[False],
        ),
    ] = False,
) -> list[EEGEvaluationRead]:
    """
    Retrieves all EEG sessions for a specific user.

    Args:
        user_id: ID of the user whose sessions are to be retrieved.
        db: Database session dependency.
        pending_only: If True, returns only unprocessed sessions.

    Returns:
        List of user's EEG sessions in read format.
    """
    try:
        statement = select(EEGEvaluation).where(EEGEvaluation.user_id == user_id)
        if pending_only:
            statement = statement.where(EEGEvaluation.processed.is_(False))

        sessions = db.exec(statement).all()

        logging.info(f"Retrieved {len(sessions)} EEG sessions for user {user_id}.")
        return sessions

    except SQLAlchemyError as e:
        logging.error(
            f"Database error retrieving EEG sessions for user {user_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error retrieving EEG sessions for user {user_id}: {str(e)}",
        ) from e

    except Exception as e:
        logging.error(
            f"Unexpected error retrieving EEG sessions for user {user_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from e
