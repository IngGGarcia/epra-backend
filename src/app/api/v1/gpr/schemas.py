"""
Schema definitions for the GPR API endpoints.

This module contains Pydantic models for request and response validation
in the GPR API endpoints.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class GPRTrainingRequest(BaseModel):
    """
    Request model for training a GPR model.
    """

    features: List[List[float]] = Field(
        ..., description="Input features (vectorized images)"
    )
    targets: List[float] = Field(..., description="Target values (valence or arousal)")
    kernel: str = Field(default="rbf", description="Type of kernel to use")
    length_scale: float = Field(default=1.0, description="Length scale parameter")
    signal_variance: float = Field(default=1.0, description="Signal variance parameter")
    noise_variance: float = Field(default=0.1, description="Noise variance parameter")
    max_iter: int = Field(default=1000, description="Maximum number of iterations")
    optimizer: str = Field(default="lbfgs", description="Optimization algorithm")


class GPRPredictionRequest(BaseModel):
    """
    Request model for making predictions with a GPR model.
    """

    features: List[List[float]] = Field(..., description="Input features to predict")
    return_std: bool = Field(
        default=True, description="Whether to return standard deviations"
    )


class GPRPredictionResponse(BaseModel):
    """
    Response model for GPR predictions.
    """

    predictions: List[float] = Field(..., description="Predicted values")
    standard_deviations: Optional[List[float]] = Field(
        None, description="Standard deviations of predictions"
    )


class GPRTrainingResponse(BaseModel):
    """
    Response model for GPR training results.
    """

    model_id: str = Field(..., description="Unique identifier for the trained model")
    metrics: dict = Field(..., description="Training metrics")
    parameters: dict = Field(..., description="Model parameters")


class GPRModelInfo(BaseModel):
    """
    Model for GPR model information.
    """

    model_id: str = Field(..., description="Unique identifier for the model")
    kernel: str = Field(..., description="Type of kernel used")
    parameters: dict = Field(..., description="Model parameters")
    created_at: str = Field(..., description="Creation timestamp")
    metrics: Optional[dict] = Field(None, description="Model performance metrics")
