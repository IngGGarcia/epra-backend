"""
Get Image Service

This service provides functionality for getting an image.
"""

from functools import lru_cache
import random

from sqlmodel import Session, select

from src.app.models.image_classification import ImageClassification

LIMIT_IMAGES = 5


@lru_cache(maxsize=128)
def _get_all_classifications(session: Session) -> list[tuple[int, int]]:
    """
    Retrieve all image IDs and their classifications from the database.
    Cached to improve performance for repeated calls.

    Args:
        session: Database session

    Returns:
        list[tuple[int, int]]: List of tuples containing (image_id, classification)
    """
    return session.exec(
        select(
            ImageClassification.image_id,
            ImageClassification.violent_final_classification,
        )
    ).all()


@lru_cache(maxsize=32)
def _get_available_classifications(results: tuple[tuple[int, int], ...]) -> set[int]:
    """
    Get the set of classification numbers that exist in the results.
    Cached to improve performance for repeated calls.

    Args:
        results: Tuple of tuples containing (image_id, classification)

    Returns:
        set[int]: Set of available classification numbers
    """
    return {classification for _, classification in results}


def _get_random_image_for_classification(
    results: tuple[tuple[int, int], ...], classification: int
) -> int:
    """
    Get a random image ID for a specific classification.
    Cached to improve performance for repeated calls with same parameters.

    Args:
        results: Tuple of tuples containing (image_id, classification)
        classification: The classification number to filter by

    Returns:
        int: Random image ID for the given classification
    """
    matching_images = [img_id for img_id, cls in results if cls == classification]
    return random.choice(matching_images)


def get_random_images(session: Session, limit: int = LIMIT_IMAGES) -> list[int]:
    """
    Get random image IDs ensuring at least one image from each available classification
    before allowing repetitions.

    Args:
        session: Database session
        limit: Maximum number of images to return (default: LIMIT_IMAGES)

    Returns:
        list[int]: List of image IDs, with at most 'limit' images
    """
    # Convert results to tuple for caching
    results = tuple(_get_all_classifications(session))
    available_classifications = _get_available_classifications(results)

    selected_images = []

    # First, ensure one image from each classification
    for classification in available_classifications:
        image_id = _get_random_image_for_classification(results, classification)
        selected_images.append(image_id)

    # Then, if we haven't reached the limit, add more random images
    remaining_slots = limit - len(selected_images)
    if remaining_slots > 0:
        while len(selected_images) < limit:
            # Get a random classification
            classification = random.choice(list(available_classifications))
            # Get a random image for this classification
            image_id = _get_random_image_for_classification(results, classification)
            selected_images.append(image_id)

    # Shuffle the final list to ensure randomness in the order
    random.shuffle(selected_images)
    return selected_images
