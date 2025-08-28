"""
Get Image Vector Service

This service provides functionality for getting the vector of an image.
"""

from functools import lru_cache
import json

from sqlmodel import Session, select

from src.app.models.image_classification import ImageClassification


@lru_cache(maxsize=128)
def get_image_vector(session: Session, image_id: int) -> list[float]:
    """
    Get the vector of an image.
    """
    statement = select(ImageClassification.feature_vector).where(
        ImageClassification.image_id == image_id,
    )
    return session.exec(statement).first()


def get_image_vector_batch(
    session: Session, image_ids: list[int]
) -> list[tuple[int, list[float]]]:
    """
    Get the vectors of a list of images.
    """
    statement = select(
        ImageClassification.image_id,
        ImageClassification.feature_vector,
    ).where(
        ImageClassification.image_id.in_(image_ids),
    )
    result = session.exec(statement).all()
    return [(id, json.loads(v)) for id, v in result]
