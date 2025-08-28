"""
EEG Metrics Router Initialization.

This module consolidates all EEG metrics endpoints into a single APIRouter instance.
It adheres to clean architecture principles, allowing each endpoint to be defined atomically
and facilitating isolated testing and maintenance.
"""

from fastapi import APIRouter

# Heuristic metric routers
from .heuristic_all import router as heuristic_all_router
from .heuristic_evaluation import router as heuristic_evaluation_router
from .heuristic_single import router as heuristic_single_router
from .heuristic_user import router as heuristic_user_router
from .test_metric import router as test_metric_router

router = APIRouter()

# Include metric routers
router.include_router(heuristic_all_router, prefix="")
router.include_router(heuristic_evaluation_router, prefix="")
router.include_router(heuristic_user_router, prefix="")
router.include_router(heuristic_single_router, prefix="")
router.include_router(test_metric_router, prefix="")
