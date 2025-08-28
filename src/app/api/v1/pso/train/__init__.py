"""
Training module for image feature optimization using PSO.

This module exposes the router and endpoints to interact with the optimizer.
"""

from fastapi import APIRouter

from .user import router as user_router

router = APIRouter(prefix="/train")

# Include user router
router.include_router(user_router)

__all__ = ["router"]
