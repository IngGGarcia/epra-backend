"""
EEG Evaluation Models

This module defines the models for EEG evaluation data management, adhering to SQLModel standards.
It includes the base model, database model, and serializers for input (POST) and output (GET) operations.

The separation ensures modularity, allowing controlled exposure of fields and facilitating reproducible
data handling for EEG research.
"""

from datetime import UTC, date, datetime
from enum import Enum

from sqlmodel import Field, SQLModel


class ProcessingStatus(str, Enum):
    """
    Enum representing the possible states of EEG evaluation processing.
    """

    PENDING = "pending"
    PROCESSED = "processed"
    ERROR = "error"


class EEGEvaluationBase(SQLModel):
    """
    Base model for EEG evaluation data.
    Contains common attributes shared across different operations.
    """

    user_id: int = Field(
        index=True,
        description="Identifier of the user associated with this EEG evaluation.",
    )
    eeg_file_path: str = Field(description="File path to the stored EEG data.")
    test_sam_file_path: str = Field(
        description="File path to the stored test SAM data."
    )


class EEGEvaluation(EEGEvaluationBase, table=True):
    """
    Database model representing an EEG evaluation stored in the database.
    Includes evaluation ID, evaluation date, processing status, and error message.
    """

    __tablename__ = "eeg_evaluation"

    id: int | None = Field(default=None, primary_key=True)
    evaluation_date: date = Field(
        default_factory=lambda: datetime.now(UTC).date(),
        description="Date when the EEG evaluation was recorded.",
    )
    processed: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current status of the EEG data processing.",
    )
    error_message: str | None = Field(
        default=None,
        description="Message describing the error that occurred during preprocessing.",
    )


class EEGEvaluationCreate(EEGEvaluationBase):
    """
    Serializer for creating a new EEG evaluation (POST request).
    The evaluation_date is optional; if not provided, defaults to current date.
    """

    evaluation_date: date | None = Field(
        default_factory=lambda: datetime.now(UTC).date(),
        description="Date when the EEG evaluation was recorded. Defaults to today if not provided.",
    )


class EEGEvaluationRead(EEGEvaluationBase):
    """
    Serializer for reading EEG evaluation data (GET request).
    Includes evaluation ID, evaluation date, processing status, and error message for external consumers.
    """

    id: int
    evaluation_date: date
    processed: ProcessingStatus
    error_message: str | None
