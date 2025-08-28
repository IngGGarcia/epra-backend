"""
List Evaluations API

This API provides functionality for listing evaluations.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from src.app.db import get_session
from src.app.models.evaluation import Evaluation, EvaluationRead

router = APIRouter()


@router.get("/list")
def list_evaluations(
    session: Annotated[Session, Depends(get_session)],
) -> list[EvaluationRead]:
    """
    List evaluations for a user.
    """
    statement = select(Evaluation).order_by(Evaluation.created_at.desc())
    return session.exec(statement).all()
