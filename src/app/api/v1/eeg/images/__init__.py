"""
EEG Images endpoints
"""

from fastapi import APIRouter

from .images import router as image_router

router = APIRouter(prefix="")

router.include_router(image_router)

# Import routes from other modules here
# from . import routes

# Export the router for use in the main app
__all__ = ["router"]
