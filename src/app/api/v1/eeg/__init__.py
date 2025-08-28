"""
EEG Module Router Initialization.

This module initializes and aggregates all EEG-related routers. It follows a modular structure,
grouping all EEG-related functionality including:
- Evaluation creation and management
- Image serving
- Preprocessing
- Metrics calculation
"""

from fastapi import APIRouter

# from .calculate_valence_arousal import router as valence_arousal_router  # Disabled - V1 deprecated, use V2
from .evaluation import router as evaluation_router
from .images import router as images_router
from .metrics import router as metrics_router
from .preprocess import router as preprocess_router

router = APIRouter(prefix="/eeg")

# Include sub-routers
router.include_router(
    evaluation_router, prefix="/evaluation", tags=["EEG - Evaluation"]
)
router.include_router(images_router, prefix="/images", tags=["EEG - Images"])
router.include_router(
    preprocess_router, prefix="/preprocess", tags=["EEG - Preprocess"]
)
router.include_router(metrics_router, prefix="/metrics", tags=["EEG - Metrics"])
# router.include_router(
#     valence_arousal_router, prefix="/valence-arousal", tags=["EEG - Valence & Arousal"]
# )  # Disabled - V1 deprecated, use V2
