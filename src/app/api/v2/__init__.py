"""
API v2

This module contains the v2 API routes.
"""

from fastapi import APIRouter

from .evaluation import router as evaluation_router
from .images import router as images_router
from .predict import router as predict_router
from .train import router as train_router

router = APIRouter(prefix="/v2")

router.include_router(evaluation_router)
router.include_router(images_router)
router.include_router(train_router)
router.include_router(predict_router)
