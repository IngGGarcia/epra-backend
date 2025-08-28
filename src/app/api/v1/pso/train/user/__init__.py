"""
This module contains the FastAPI endpoints to interact with the image feature optimizer.
"""

from fastapi import APIRouter

from .user_train import router as user_train_router

router = APIRouter(
    prefix="/user",
)

router.include_router(user_train_router)
