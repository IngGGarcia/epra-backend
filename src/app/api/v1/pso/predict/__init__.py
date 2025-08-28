"""
Prediction module for image feature optimization using PSO.

This module exposes the router and endpoints to predict subjective responses using optimization vectors.
"""

from fastapi import APIRouter

from .predict import router as predict_router

router = APIRouter(prefix="/predict")

router.include_router(predict_router)

__all__ = ["router"]
