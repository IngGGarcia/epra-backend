"""
Image Classification Models

This module defines the data models responsible for managing image classification data.
Each record captures the original and final classification of images regarding violence content.
"""

from sqlmodel import Field, SQLModel


class ImageClassificationBase(SQLModel):
    """
    Base model defining the core attributes of an image classification entry.

    Attributes:
        image_id (str): Unique identifier for the image.
        is_violent_original (bool): Original classification of the image regarding violence content.
        violent_final_classification (bool): Final classification of the image regarding violence content.
        feature_vector (str): Vector of image features stored as a JSON string.
    """

    image_id: str = Field(
        description="Unique identifier for the image.",
        index=True,
        primary_key=True,
    )
    is_violent_original: bool = Field(
        description="Original classification of the image regarding violence content."
    )
    violent_final_classification: int = Field(
        description="Final classification of the image regarding violence content."
    )
    feature_vector: str = Field(
        default="", description="Vector of image features stored as a JSON string."
    )


class ImageClassification(ImageClassificationBase, table=True):
    """
    Database model representing the persistent storage structure for image classifications.

    This table captures a single record per image, including both original and final classifications
    regarding violence content and its feature vector.
    """

    __tablename__ = "image_classification"


class ImageClassificationCreate(ImageClassificationBase):
    """
    Serializer model for creating new image classification records (POST operations).

    Ensures that all required attributes are provided for successful database insertion,
    excluding the auto-generated primary key.
    """

    pass


class ImageClassificationRead(ImageClassificationBase):
    """
    Serializer model for reading image classification records (GET operations).

    Exposes all relevant attributes, including the primary key, for external data consumers.
    """

    id: int
