"""
PSO module for image feature optimization.

This module exposes the training and prediction routers to interact with the optimizer.
"""

from fastapi import APIRouter

from .predict import router as predict_router
from .train import router as train_router

router = APIRouter(prefix="/pso", tags=["PSO"])

router.include_router(train_router)
router.include_router(predict_router)

__all__ = ["router"]
