from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, func, select

from src.app.db import get_session
from src.app.models import EEGImageEvaluation, PredictionTrain

router = APIRouter()


class UserInfo(BaseModel):
    user_id: int
    evaluation_quantity: int
    prediction_images_quantity: int
    prediction_created_at: datetime | None
    prediction_updated_at: datetime | None


@router.get("/", response_model=list[UserInfo])
async def get_users_info(
    session: Annotated[Session, Depends(get_session)],
) -> list[UserInfo]:
    stmt = (
        select(
            # Siempre mostramos el user_id, ya sea de PredictionTrain o de EEGImageEvaluation
            func.coalesce(PredictionTrain.user_id, EEGImageEvaluation.user_id).label(
                "user_id"
            ),
            # Conteo distinto de evaluation_id (0 si no hay)
            func.count(func.distinct(EEGImageEvaluation.evaluation_id)).label(
                "evaluation_quantity"
            ),
            # Longitud del array de imágenes, o 0 si no existe fila de PredictionTrain
            func.coalesce(
                func.json_array_length(PredictionTrain.images_for_training), 0
            ).label("prediction_images_quantity"),
            # Estas columnas quedarán NULL si no hay fila de PredictionTrain
            PredictionTrain.created_at.label("prediction_created_at"),
            PredictionTrain.updated_at.label("prediction_updated_at"),
        )
        .join(
            PredictionTrain,
            EEGImageEvaluation.user_id == PredictionTrain.user_id,
            isouter=True,  # LEFT JOIN
        )
        .group_by(
            PredictionTrain.user_id,
            EEGImageEvaluation.user_id,
            PredictionTrain.images_for_training,
            PredictionTrain.created_at,
            PredictionTrain.updated_at,
        )
        .order_by("user_id")
    )

    rows = session.exec(stmt).all()
    return [UserInfo.model_validate(row._mapping) for row in rows]
