"""
Image Manager Module

This module provides a specialized manager for serving images from a specific directory.
"""

from pathlib import Path

from fastapi import HTTPException, status
from fastapi.responses import FileResponse


class ImageManager:
    """
    A specialized manager for serving images from a specific directory.
    """

    ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

    def __init__(self, images_path: str = "images"):
        """
        Initializes the ImageManager with a specified images directory.

        Args:
            images_path (str): The directory where images are stored.
        """
        self.images_path = Path(images_path).resolve()
        if not self.images_path.exists():
            self.images_path.mkdir(parents=True, exist_ok=True)

    def serve_image(self, image_path: str) -> FileResponse:
        """
        Serves an image from the images directory.

        Args:
            image_path (str): Path to the image relative to the images directory.

        Returns:
            FileResponse: The image file response.

        Raises:
            HTTPException: If the image is not found or access is denied.
        """
        try:
            # Construct the full path to the image
            full_path = self.images_path / image_path

            # Check if the file exists
            if not full_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Image not found: {image_path}",
                )

            # Check if the file is within the images directory
            if not str(full_path).startswith(str(self.images_path)):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Invalid path",
                )

            # Check if the file is an image
            if full_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file type: {full_path.suffix}. Allowed: {self.ALLOWED_EXTENSIONS}",
                )

            # Determine the media type based on file extension
            media_type = (
                "image/jpeg"
                if full_path.suffix.lower() in {".jpg", ".jpeg"}
                else "image/png"
            )

            return FileResponse(
                path=full_path,
                media_type=media_type,
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error serving image: {str(e)}",
            ) from e
