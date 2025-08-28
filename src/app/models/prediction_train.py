"""
This module contains stats for the prediction training by user.
"""

from datetime import datetime
import json
from typing import ClassVar

from pydantic import field_validator
from sqlmodel import Field, SQLModel


class PredictionTrainBase(SQLModel):
    """
    Logical base model for prediction training.
    """

    user_id: int
    images_for_training: list[int]
    precidction_vector: list[float]


class PredictionTrain(PredictionTrainBase, table=True):
    """
    Physical table model adapted for SQLite (stores lists as JSON strings).
    """

    __tablename__: ClassVar[str] = "prediction_train"

    id: int = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Override field types for SQLite compatibility
    images_for_training: str = Field(
        description="Images for training (IDs) in JSON format"
    )
    precidction_vector: str = Field(description="Precidction vector in JSON format")

    @field_validator("images_for_training", "precidction_vector", mode="before")
    @classmethod
    def serialize_lists(cls, v):
        return json.dumps(v) if isinstance(v, list) else v

    def decoded_images(self) -> list[int]:
        return json.loads(self.images_for_training)

    def decoded_vector(self) -> list[float]:
        return json.loads(self.precidction_vector)


class PredictionTrainCreate(PredictionTrainBase):
    """
    Model used for creation (input).
    """

    pass


class PredictionTrainRead(PredictionTrainBase):
    """
    Model used for reading (output).
    """

    id: int
    created_at: datetime
    updated_at: datetime

    @field_validator("images_for_training", "precidction_vector", mode="before")
    @classmethod
    def deserialize_lists(cls, v):
        return json.loads(v) if isinstance(v, str) else v
