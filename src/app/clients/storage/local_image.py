"""
Local Image Client

This module provides a client for handling local image storage operations.
"""

from pathlib import Path

BASE_PATH = "images"


class LocalImageClient:
    """
    Client for handling local image storage operations.
    """

    def __init__(self, base_path: str = BASE_PATH):
        """
        Initialize the LocalImageClient.
        """
        self.base_path = Path(base_path)

    def get_image(self, image_id: int):
        """
        Get an image from the images directory.
        """

        if not (self.base_path / f"{image_id}.jpg").exists():
            raise Exception("Image not found")

        return {
            "path": self.base_path / f"{image_id}.jpg",
            "media_type": "image/jpeg",
        }
