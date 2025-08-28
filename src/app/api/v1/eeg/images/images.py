"""
EEG Image Serving Endpoints

Provides endpoints for serving images from the images directory.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status
from fastapi.responses import FileResponse
from sqlmodel import Session

from src.app.db.db_session import get_session
from src.modules.storage.image_manager import ImageManager

# Configure logging
logging.basicConfig(level=logging.INFO)

router = APIRouter()


def get_image_manager() -> ImageManager:
    return ImageManager()


@router.get(
    path="/{image_path:path}",
    summary="Serve an image",
    description="Serves an image from the images directory.",
)
async def serve_image(
    image_path: Annotated[
        str, Path(description="Path to the image relative to the images directory")
    ],
    db: Annotated[Session, Depends(get_session)],
    image_manager: Annotated[ImageManager, Depends(get_image_manager)],
) -> FileResponse:
    """
    Serves an image from the images directory.

    Args:
        image_path: Path to the image relative to the images directory.
        db: Database session dependency.
        image_manager: Image manager dependency.

    Returns:
        FileResponse containing the image.
    """
    try:
        return image_manager.serve_image(image_path)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error serving image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error serving image: {str(e)}",
        ) from e
