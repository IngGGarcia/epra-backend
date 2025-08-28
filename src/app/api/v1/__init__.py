"""
API v1.

Este módulo expone los routers de la API v1.
"""

from fastapi import APIRouter

# EEG related router
from .eeg import router as eeg_router
from .gpr import router as gpr_router
from .pso import router as pso_router
from .users import router as users_router

router = APIRouter(prefix="/v1")

# Include routers
router.include_router(eeg_router)
router.include_router(pso_router)
router.include_router(users_router)
router.include_router(gpr_router, prefix="/gpr", tags=["GPR"])

__all__ = ["router"]
