"""
EEG Evaluation Module

This module contains all evaluation-related endpoints, including:
- Evaluation creation
- Preliminary evaluation setup
- Evaluation statistics
- Evaluation deletion
"""

from fastapi import APIRouter

# Evaluation routers
from .evaluation_create import router as evaluation_router
from .evaluation_delete import router as evaluation_delete_router
from .evaluation_pre_create import router as evaluation_pre_router
from .evaluation_retrieve import router as evaluation_retrieve_router
from .evaluation_stats import router as evaluation_stats_router

router = APIRouter()

# Include evaluation sub-routers
router.include_router(evaluation_router, prefix="")
router.include_router(evaluation_pre_router, prefix="")
router.include_router(
    evaluation_stats_router, prefix=""
)  # Stats router before retrieve
router.include_router(evaluation_retrieve_router, prefix="")
router.include_router(evaluation_delete_router, prefix="")

__all__ = ["router"]
