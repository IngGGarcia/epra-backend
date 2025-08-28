"""
Prediction API Module

This module provides endpoints for making predictions using trained GPR models.
"""

from fastapi import APIRouter

from .predict import router as predict_router

router = APIRouter(prefix="/predict", tags=["Predict"])

router.include_router(predict_router)
