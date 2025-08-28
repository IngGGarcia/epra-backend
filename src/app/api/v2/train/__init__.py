"""
Train API
"""

from fastapi import APIRouter

from .train_by_user import router as train_by_user_router

router = APIRouter(prefix="/train", tags=["Train"])

router.include_router(train_by_user_router)
