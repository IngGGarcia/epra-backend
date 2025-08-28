"""
Evaluation API
"""

from fastapi import APIRouter

from .create import router as create_router
from .list import router as list_router
from .pre_create import router as pre_create_router
from .stats import router as stats_router
from .synthetic import router as synthetic_router

router = APIRouter(prefix="/evaluation", tags=["Evaluation V2"])

router.include_router(create_router)
router.include_router(pre_create_router)
router.include_router(list_router)
router.include_router(stats_router)
router.include_router(synthetic_router)
