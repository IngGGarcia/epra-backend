"""
Image Evaluation Models

This module defines the data models responsible for managing detailed image evaluation results.
Each record captures the association between a specific user, a particular image stimulus, and the corresponding
preprocessed EEG segment. Additionally, subjective responses from the Self-Assessment Manikin (SAM) test
(valence and arousal scores) are recorded, alongside heuristic metrics intended for further computational analysis.
"""

from datetime import datetime
from uuid import uuid4

from sqlmodel import Field, SQLModel


class ImageEvaluationBase(SQLModel):
    """
    Base model defining the core attributes of an image evaluation entry.
    """

    evaluation_id: str = Field(
        description="Identifier referencing the corresponding EEG evaluation session."
    )
    user_id: int = Field(
        description="Identifier of the user associated with this image evaluation session."
    )
    image_id: int = Field(
        description="Unique identifier or label for the stimulus image presented to the user."
    )
    sam_valence: int = Field(
        description="Valence score from the SAM scale reflecting the user's emotional assessment."
    )
    sam_arousal: int = Field(
        description="Arousal score from the SAM scale reflecting the user's emotional assessment."
    )
    eeg_valence: float = Field(
        description="EEG-derived valence score based on signal analysis."
    )
    eeg_arousal: float = Field(
        description="EEG-derived arousal score based on signal analysis."
    )


class ImageEvaluation(ImageEvaluationBase, table=True):
    """
    Database model representing the persistent storage structure for image evaluations.
    """

    __tablename__ = "image_evaluation"

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this image evaluation record.",
        primary_key=True,
    )
    created_at: datetime = Field(
        description="Timestamp when the image evaluation was created.",
        default_factory=datetime.now,
    )
    updated_at: datetime = Field(
        description="Timestamp when the image evaluation was last updated.",
        default_factory=datetime.now,
    )


IMAGE_EVALUATION_MIGRATION = """
CREATE TABLE image_evaluation (
    id TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    sam_valence INTEGER NOT NULL,
    sam_arousal INTEGER NOT NULL,
    eeg_valence REAL NOT NULL,
    eeg_arousal REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_image_evaluation_user_id ON image_evaluation(user_id);
"""
