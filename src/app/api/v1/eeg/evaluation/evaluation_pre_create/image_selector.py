"""
Image Selection Logic for EEG Evaluation

This module handles the selection of images for EEG evaluation sessions,
implementing specific business rules for image distribution based on
violence classification and user history.
"""

import random
from typing import Final

from sqlmodel import Session, select

from src.app.models.epra_image_evaluation import EEGImageEvaluation
from src.app.models.image_classification import ImageClassification

# Configuration Constants
NON_VIOLENT_CLASSIFICATIONS: Final[list[int]] = [0, 1, 2, 3]
VIOLENT_CLASSIFICATION: Final[int] = 4
MIN_IMAGES_PER_CLASSIFICATION: Final[int] = 1
DEFAULT_RANDOM_SEED: Final[int] = 42


def get_user_viewed_images(db: Session, user_id: int) -> list[str]:
    """
    Retrieves all images that a user has already viewed in previous evaluations.

    Args:
        db: Database session
        user_id: User identifier

    Returns:
        List of image IDs that the user has already viewed
    """
    statement = select(EEGImageEvaluation.image_id).where(
        EEGImageEvaluation.user_id == user_id
    )
    results = db.exec(statement).all()
    return [result for result in results]


def calculate_image_distribution(
    total_images: int, negative_images: int
) -> tuple[int, int]:
    """
    Calculates the distribution of images between negative and non-negative categories.

    Args:
        total_images: Total number of images needed
        negative_images: Number of negative images requested

    Returns:
        Tuple containing (non_negative_images, negative_images)
    """
    non_negative_images = total_images - negative_images
    return non_negative_images, negative_images


def get_images_by_classification(
    db: Session, classification: int, limit: int, exclude_images: list[str]
) -> list[str]:
    """
    Retrieves images filtered by their final classification.

    Args:
        db: Database session
        classification: Classification value to filter by (0-4)
        limit: Maximum number of images to retrieve
        exclude_images: List of image IDs to exclude from results

    Returns:
        List of image IDs matching the criteria
    """
    # Get total count of available images for this classification
    count_statement = (
        select(ImageClassification.image_id)
        .where(ImageClassification.violent_final_classification == classification)
        .where(ImageClassification.image_id.not_in(exclude_images))
    )
    total_available = len(db.exec(count_statement).all())

    if total_available == 0:
        return []

    # Calculate random offset
    max_offset = max(0, total_available - limit)
    random_offset = random.randint(0, max_offset)

    # Get images with random offset
    statement = (
        select(ImageClassification.image_id)
        .where(ImageClassification.violent_final_classification == classification)
        .where(ImageClassification.image_id.not_in(exclude_images))
        .offset(random_offset)
        .limit(limit)
    )
    results = db.exec(statement).all()
    return [result for result in results]


def get_non_violent_images(
    db: Session, limit: int, exclude_images: list[str]
) -> list[str]:
    """
    Retrieves non-violent images with simple distribution across classifications.

    Args:
        db: Database session
        limit: Maximum number of images to retrieve
        exclude_images: List of image IDs to exclude

    Returns:
        List of selected image IDs
    """
    selected_images = []
    remaining_limit = limit

    # Calculate base distribution
    base_images_per_class = limit // len(NON_VIOLENT_CLASSIFICATIONS)
    extra_images = limit % len(NON_VIOLENT_CLASSIFICATIONS)

    # Distribute images evenly
    for classification in NON_VIOLENT_CLASSIFICATIONS:
        if remaining_limit <= 0:
            break

        # Calculate images for this classification
        images_needed = base_images_per_class
        if extra_images > 0:
            images_needed += 1
            extra_images -= 1

        images = get_images_by_classification(
            db, classification, images_needed, exclude_images
        )

        if images:
            selected_images.extend(images)
            exclude_images.extend(images)
            remaining_limit -= len(images)

    # If we still have remaining limit, try to fill with any available classification
    if remaining_limit > 0:
        for classification in NON_VIOLENT_CLASSIFICATIONS:
            if remaining_limit <= 0:
                break
            images = get_images_by_classification(
                db, classification, remaining_limit, exclude_images
            )
            if images:
                selected_images.extend(images)
                exclude_images.extend(images)
                remaining_limit -= len(images)

    return selected_images


def select_evaluation_images(
    db: Session, user_id: int, total_images: int, negative_images: int
) -> list[str]:
    """
    Selects images for an EEG evaluation session based on user history and classification rules.

    Args:
        db: Database session
        user_id: User identifier
        total_images: Total number of images needed
        negative_images: Number of negative images requested

    Returns:
        List of selected image IDs for the evaluation session, randomly ordered
    """
    if total_images < negative_images:
        raise ValueError(
            "Total images must be greater than or equal to negative images"
        )

    # Get images already viewed by the user
    viewed_images = get_user_viewed_images(db, user_id)

    # If user has viewed all images, return empty list
    if len(viewed_images) >= total_images:
        return []

    # Calculate distribution
    non_negative_images = total_images - negative_images
    selected_images = []
    excluded_images = viewed_images.copy()

    # Get negative images (classification 4)
    negative_selected = get_images_by_classification(
        db, VIOLENT_CLASSIFICATION, negative_images, excluded_images
    )
    selected_images.extend(negative_selected)
    excluded_images.extend(negative_selected)

    # Get non-negative images with simple distribution
    non_negative_selected = get_non_violent_images(
        db, non_negative_images, excluded_images
    )
    selected_images.extend(non_negative_selected)

    # Randomize the order of selected images
    random.shuffle(selected_images)
    return selected_images
