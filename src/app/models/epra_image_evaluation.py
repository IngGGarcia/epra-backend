"""
EEG Image Evaluation Models

This module defines the data models responsible for managing detailed EEG evaluation results at the image level.
Each record captures the association between a specific user, a particular image stimulus, and the corresponding
preprocessed EEG segment. Additionally, subjective responses from the Self-Assessment Manikin (SAM) test
(valence and arousal scores) are recorded, alongside heuristic metrics intended for further computational analysis.

Design decisions emphasize modularity, atomic responsibility, and weakly linked relational references. Explicit
foreign keys are intentionally avoided to promote flexibility, relying on controlled logical consistency enforced
at the application level.
"""

from sqlmodel import Field, SQLModel


class EEGImageEvaluationBase(SQLModel):
    """
    Base model defining the core attributes of an image-level EEG evaluation entry.

    Attributes:
        user_id (int): Identifier of the user associated with the image evaluation session.
        evaluation_id (int): Identifier referencing the parent EEG evaluation session.
        image_id (str): Unique identifier or label for the stimulus image presented during the experiment.
        eeg_file_path (str): File path pointing to the preprocessed EEG segment corresponding to the image.
        valence (float): Subjective valence rating based on the SAM scale (typically ranging from 1 to 9).
        arousal (float): Subjective arousal rating based on the SAM scale (typically ranging from 1 to 9).
        metric_heurist (float): Heuristic metric derived from signal analysis, initialized with default value.
        metric_v2 (float): Additional metric reserved for future analytical purposes, initialized with default value.
        valence_eeg (float): EEG-derived valence score based on signal analysis.
        arousal_eeg (float): EEG-derived arousal score based on signal analysis.
    """

    user_id: int = Field(
        description="Identifier of the user associated with this image evaluation session.",
        index=True,
    )
    evaluation_id: int = Field(
        description="Identifier referencing the corresponding EEG evaluation session.",
        index=True,
    )
    image_id: str = Field(
        description="Unique identifier or label for the stimulus image presented to the user."
    )
    eeg_file_path: str = Field(
        description="Absolute or relative path to the preprocessed EEG segment file for this image."
    )
    valence: float = Field(
        description="Valence score from the SAM scale reflecting the user's emotional assessment."
    )
    arousal: float = Field(
        description="Arousal score from the SAM scale reflecting the user's emotional assessment."
    )
    metric_heurist: float = Field(
        default=0.0,
        description="Heuristic metric derived from EEG analysis or computational models.",
    )
    metric_v2: float = Field(
        default=0.0,
        description="Secondary metric reserved for extended analytical evaluations.",
    )
    valence_eeg: float = Field(
        default=0.0,
        description="EEG-derived valence score based on signal analysis.",
    )
    arousal_eeg: float = Field(
        default=0.0,
        description="EEG-derived arousal score based on signal analysis.",
    )


class EEGImageEvaluation(EEGImageEvaluationBase, table=True):
    """
    Database model representing the persistent storage structure for image-level EEG evaluations.

    This table captures a single record per image stimulus, including the user identifier, evaluation session,
    preprocessed EEG data path, subjective response scores, and analytical metrics. Relationships to users or
    EEG sessions are intentionally implemented without explicit foreign key constraints, ensuring logical flexibility
    and controlled consistency managed at the application layer.
    """

    __tablename__ = "eeg_image_evaluation"

    id: int = Field(
        default=None,
        primary_key=True,
        description="Primary key identifier for the image evaluation entry.",
    )


class EEGImageEvaluationCreate(EEGImageEvaluationBase):
    """
    Serializer model for creating new image-level EEG evaluation records (POST operations).

    Ensures that all required attributes are provided for successful database insertion,
    excluding the auto-generated primary key.
    """

    pass


class EEGImageEvaluationRead(EEGImageEvaluationBase):
    """
    Serializer model for reading image-level EEG evaluation records (GET operations).

    Exposes all relevant attributes, including the primary key, for external data consumers.
    """

    id: int
