"""
Get Image API

This module provides an endpoint for getting an image.
"""

from fastapi import APIRouter
from fastapi.responses import FileResponse

from src.app.clients.storage import LocalImageClient

router = APIRouter()

local_image_client = LocalImageClient()


@router.get("/{image_id}")
async def get_image(image_id: int) -> FileResponse:
    """
    Get an image from the images directory.
    """
    image_metadata = local_image_client.get_image(image_id)
    return FileResponse(**image_metadata)
