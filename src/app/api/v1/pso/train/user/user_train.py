"""
This module contains the FastAPI endpoints to interact with the image feature optimizer.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlmodel import Session

from src.app.db.db_session import get_session

from .get_user_data import get_user_data

router = APIRouter()


@router.post("/{user_id}")
def train_user_model(
    user_id: int,
    session: Annotated[Session, Depends(get_session)],
):
    """
    Train a PSO model for a specific user using their EEG image evaluations.
    """
    user_data = get_user_data(user_id, session)
    return "Ok"
