"""
EEG Preprocessing Router Initialization.

This module consolidates the EEG preprocessing endpoint into a single APIRouter instance.
It adheres to clean architecture principles, ensuring modularity, atomic endpoint definition,
and ease of testing and extension.
"""

from fastapi import APIRouter

# Preprocessing router
from .preprocess_eeg import router as preprocess_eeg_router

router = APIRouter()

# Include preprocessing router
router.include_router(preprocess_eeg_router, prefix="")
