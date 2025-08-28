"""
Images API

This module contains the images API routes.
"""

from fastapi import APIRouter

from .get_image import router as get_image_router

router = APIRouter(prefix="/images", tags=["Images"])

router.include_router(get_image_router)
