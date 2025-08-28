"""
Train by User API

This API provides functionality for training a model by user.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlmodel import Session

from src.app.db import get_session
from src.app.services.train import train_by_user_service

router = APIRouter()


@router.post("/{user_id}")
def train_by_user_endpoint(
    user_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> str:
    """
    Train arousal and valence models for a specific user.

    Args:
        user_id: The ID of the user to train models for
        session: Database session

    Returns:
        ModelsTrained containing the trained models
    """
    train_by_user_service(user_id, session)
    return f"Models trained for user {user_id}"
