"""
EEG Evaluation Deletion Endpoint

Provides a DELETE endpoint to remove EEG evaluations from the database.
Only allows deletion of evaluations that have errors in their processing.
Implements dependency injection, SQLModel-based queries, and robust error handling.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from src.app.db.db_session import get_session
from src.app.models.epra_evaluation import EEGEvaluation, ProcessingStatus

# Configure logging
logging.basicConfig(level=logging.INFO)

router = APIRouter()


@router.delete(
    path="/{evaluation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete EEG evaluation with error",
    description="Deletes an EEG evaluation that has an error status. Evaluations with pending or processed status cannot be deleted.",
)
async def delete_evaluation(
    evaluation_id: Annotated[
        int,
        Path(
            title="Evaluation ID",
            description="The ID of the EEG evaluation to delete.",
            examples=[1],
            gt=0,
        ),
    ],
    db: Annotated[Session, Depends(get_session)],
) -> None:
    """
    Deletes an EEG evaluation that has an error status.

    Args:
        evaluation_id: ID of the EEG evaluation to delete.
        db: Database session dependency.

    Raises:
        HTTPException: If the evaluation doesn't exist or doesn't have an error status.
    """
    try:
        # Get the evaluation
        statement = select(EEGEvaluation).where(EEGEvaluation.id == evaluation_id)
        evaluation = db.exec(statement).first()

        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontró la evaluación EEG con ID {evaluation_id}. Por favor, verifique el ID proporcionado.",
            )

        if evaluation.processed != ProcessingStatus.ERROR:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Solo se pueden eliminar evaluaciones que tengan estado de error. Esta evaluación tiene estado: "
                + evaluation.processed.value,
            )

        # Delete the evaluation
        db.delete(evaluation)
        db.commit()

        logging.info(f"Successfully deleted EEG evaluation with ID {evaluation_id}.")

    except SQLAlchemyError as e:
        logging.error(f"Database error deleting EEG evaluation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al acceder a la base de datos. Por favor, intente nuevamente más tarde. Si el problema persiste, contacte al administrador del sistema.",
        ) from e

    except Exception as e:
        logging.error(f"Unexpected error deleting EEG evaluation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ocurrió un error inesperado al procesar su solicitud. Por favor, intente nuevamente más tarde. Si el problema persiste, contacte al administrador del sistema.",
        ) from e
