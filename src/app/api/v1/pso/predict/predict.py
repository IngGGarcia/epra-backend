"""
Endpoint for predicting valence and arousal using optimization vectors.

This module contains the FastAPI endpoint to predict subjective responses
for a user's image using their optimization vector.
Vectors are read from src/modules/PSO/data/optimization_vectors.parquet
at the project root.
"""

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd

from src.modules.violence_model.image_features import load_image_features

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)


def get_project_root():
    """Return the absolute path to the real project root directory."""
    return (
        Path(__file__).resolve().parents[6]
    )  # src/app/api/v1/pso/predict/predict.py -> project root


DATA_DIR = get_project_root() / "src/modules/PSO/data"
VECTORS_PATH = DATA_DIR / "optimization_vectors.parquet"


def normalize_image_id(image_id):
    """Remove extension from image_id if it exists."""
    return os.path.splitext(str(image_id))[0]


def scale_to_sam_range(value: float) -> float:
    """
    Scale a value from [2,8] range to [1,9] range to match SAM scale.

    Args:
        value: Value in range [2,8]

    Returns:
        Scaled value in range [1,9]
    """
    # Scale from [2,8] to [0,1]
    normalized = (value - 2) / 6
    # Scale from [0,1] to [1,9]
    return 1 + (normalized * 8)


@router.get("/{user_id}/{image_id}")
async def predict_user_response(
    user_id: int,
    image_id: str,
) -> dict[str, float]:
    """
    Predict valence and arousal that a user would assign to an image
    using their optimization vectors saved in Parquet.
    Values are scaled from [2,8] to [1,9] to match the SAM scale.

    Args:
        user_id: ID of the user to predict for
        image_id: ID of the image to predict for

    Returns:
        dict[str, float]: Dictionary containing predicted valence and arousal values
                         scaled to range [1,9]

    Raises:
        HTTPException: If user vector not found or image features not available
    """
    try:
        # Read optimization vectors
        if not VECTORS_PATH.exists():
            raise ValueError("Optimization vectors file does not exist.")
        df = pd.read_parquet(VECTORS_PATH)
        row = df[df.user_id == user_id]
        if row.empty:
            raise ValueError(f"Optimization vector not found for user {user_id}")

        # Load separate weights for valence and arousal
        valence_weights = json.loads(row.iloc[0]["valence_weights"])
        arousal_weights = json.loads(row.iloc[0]["arousal_weights"])

        # Load image features and normalize image_id
        image_features = load_image_features().copy()
        image_features["image_id"] = image_features["image_id"].apply(
            normalize_image_id
        )
        logging.info(
            "First image_ids after normalization:",
            image_features["image_id"].head(10).tolist(),
        )
        image_id_norm = normalize_image_id(image_id)
        logging.info("Looking for image_id:", image_id_norm)
        image_row = image_features[image_features["image_id"] == image_id_norm]
        logging.info("Filtered DataFrame shape:", image_row.shape)
        if image_row.empty:
            raise ValueError(f"No features found for image {image_id}")

        # Get feature vector
        feature_columns = image_row.select_dtypes(include="number").columns
        logging.info("Selected numeric columns:", feature_columns.tolist())
        features = image_row[feature_columns].values[0]

        # Convert feature weights dicts to lists in the same order as feature_columns
        valence_weight_array = [
            valence_weights[f"feature_{i}"] for i in range(len(feature_columns))
        ]
        arousal_weight_array = [
            arousal_weights[f"feature_{i}"] for i in range(len(feature_columns))
        ]

        # Calculate predictions using separate weights
        valence_pred = float(np.dot(features, valence_weight_array))
        arousal_pred = float(np.dot(features, arousal_weight_array))

        # Scale predictions from [2,8] to [1,9] to match SAM scale
        valence = scale_to_sam_range(valence_pred)
        arousal = scale_to_sam_range(arousal_pred)

        return {
            "valence": valence,
            "arousal": arousal,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
