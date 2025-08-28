"""
Router for user endpoints.
This module defines the router for handling user-related API endpoints.
"""

from fastapi import APIRouter

from .get_users_info import router as get_users_info_router

router = APIRouter(prefix="/users", tags=["Users"])

router.include_router(get_users_info_router)
