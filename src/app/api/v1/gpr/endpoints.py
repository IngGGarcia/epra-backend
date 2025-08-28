"""
GPR API endpoints for model training and prediction.

This module provides endpoints for training Gaussian Process Regression models,
making predictions, and managing trained models.
"""

from datetime import datetime
from typing import List
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
import numpy as np

from src.modules.GPR.model import GaussianProcessRegressor
from src.modules.GPR.utils import evaluate_model, normalize_features, preprocess_data

from .schemas import (
    GPRModelInfo,
    GPRPredictionRequest,
    GPRPredictionResponse,
    GPRTrainingRequest,
    GPRTrainingResponse,
)

router = APIRouter()

# In-memory storage for trained models (in production, use a proper database)
trained_models = {}


@router.post("/train", response_model=GPRTrainingResponse)
async def train_model(request: GPRTrainingRequest):
    """
    Train a new GPR model with the provided data and parameters.

    Args:
        request: Training request containing features, targets, and model parameters

    Returns:
        Training response with model ID and metrics
    """
    try:
        # Convert input data to numpy arrays
        X = np.array(request.features)
        y = np.array(request.targets)

        # Normalize features
        X_norm, mean, std = normalize_features(X)

        # Split data for validation
        X_train, X_val, y_train, y_val = preprocess_data(X_norm, y)

        # Initialize and train the model
        model = GaussianProcessRegressor(
            kernel=request.kernel,
            length_scale=request.length_scale,
            signal_variance=request.signal_variance,
            noise_variance=request.noise_variance,
            max_iter=request.max_iter,
            optimizer=request.optimizer,
        )

        model.fit(X_train, y_train)

        # Make predictions on validation set
        y_pred, y_std = model.predict(X_val)

        # Calculate metrics
        metrics = evaluate_model(y_val, y_pred, y_std)

        # Generate unique model ID
        model_id = str(uuid.uuid4())

        # Store model and metadata
        trained_models[model_id] = {
            "model": model,
            "mean": mean,
            "std": std,
            "parameters": {
                "kernel": request.kernel,
                "length_scale": request.length_scale,
                "signal_variance": request.signal_variance,
                "noise_variance": request.noise_variance,
                "max_iter": request.max_iter,
                "optimizer": request.optimizer,
            },
            "created_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }

        return GPRTrainingResponse(
            model_id=model_id,
            metrics=metrics,
            parameters=trained_models[model_id]["parameters"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/{model_id}", response_model=GPRPredictionResponse)
async def predict(model_id: str, request: GPRPredictionRequest):
    """
    Make predictions using a trained GPR model.

    Args:
        model_id: ID of the trained model to use
        request: Prediction request containing features to predict

    Returns:
        Prediction response with predicted values and standard deviations
    """
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        model_data = trained_models[model_id]
        model = model_data["model"]
        mean = model_data["mean"]
        std = model_data["std"]

        # Convert input data to numpy array and normalize
        X = np.array(request.features)
        X_norm = (X - mean) / std

        # Make predictions
        predictions, std_devs = model.predict(X_norm, return_std=request.return_std)

        return GPRPredictionResponse(
            predictions=predictions.tolist(),
            standard_deviations=std_devs.tolist() if std_devs is not None else None,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=List[GPRModelInfo])
async def list_models():
    """
    List all trained GPR models.

    Returns:
        List of model information
    """
    return [
        GPRModelInfo(
            model_id=model_id,
            kernel=data["parameters"]["kernel"],
            parameters=data["parameters"],
            created_at=data["created_at"],
            metrics=data["metrics"],
        )
        for model_id, data in trained_models.items()
    ]


@router.get("/models/{model_id}", response_model=GPRModelInfo)
async def get_model_info(model_id: str):
    """
    Get information about a specific trained model.

    Args:
        model_id: ID of the model to retrieve

    Returns:
        Model information
    """
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    data = trained_models[model_id]
    return GPRModelInfo(
        model_id=model_id,
        kernel=data["parameters"]["kernel"],
        parameters=data["parameters"],
        created_at=data["created_at"],
        metrics=data["metrics"],
    )


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a trained model.

    Args:
        model_id: ID of the model to delete

    Returns:
        Success message
    """
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    del trained_models[model_id]
    return JSONResponse(content={"message": "Model deleted successfully"})
