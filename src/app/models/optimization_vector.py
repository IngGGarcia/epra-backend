"""
Modelo para almacenar los vectores de optimización de características de imágenes.

Este módulo define el modelo SQLModel para la tabla de vectores de optimización
que se utilizan para predecir las respuestas subjetivas de los usuarios.
"""

import json
from typing import List

import numpy as np
from pydantic import validator
from sqlalchemy import String
from sqlmodel import Field, SQLModel


class OptimizationVector(SQLModel, table=True):
    """
    Modelo para almacenar los vectores de optimización de características de imágenes.

    Attributes:
        user_id: ID del usuario al que pertenece el vector
        feature_weights: Vector de pesos para las características
        best_score: Mejor score obtenido durante la optimización
        important_features: Lista de características más importantes
        valence_correlation: Correlación con valencia
        arousal_correlation: Correlación con arousal
        heuristic_correlation: Correlación con la heurística
    """

    __tablename__ = "optimization_vectors"

    user_id: int = Field(primary_key=True)
    feature_weights: str = Field(sa_type=String)
    best_score: float
    important_features: str = Field(sa_type=String)
    valence_correlation: float
    arousal_correlation: float
    heuristic_correlation: float

    @validator("feature_weights", pre=True)
    def validate_feature_weights(cls, v):
        """Valida y convierte los pesos a JSON string."""
        if isinstance(v, list):
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Todos los pesos deben ser números")
            return json.dumps(v)
        return v

    @validator("important_features", pre=True)
    def validate_important_features(cls, v):
        """Valida y convierte las características importantes a JSON string."""
        if isinstance(v, list):
            if not all(isinstance(x, str) for x in v):
                raise ValueError("Todas las características deben ser strings")
            return json.dumps(v)
        return v

    def get_feature_weights(self) -> List[float]:
        """Obtiene los pesos de características como lista."""
        return json.loads(self.feature_weights)

    def get_important_features(self) -> List[str]:
        """Obtiene las características importantes como lista."""
        return json.loads(self.important_features)

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """
        Predice la valencia y arousal para un vector de características.

        Args:
            features: Vector de características de la imagen

        Returns:
            Tuple con (valencia, arousal) predichas
        """
        weights = self.get_feature_weights()
        weighted_features = np.dot(features, weights)
        return (
            weighted_features,
            weighted_features,
        )  # Por ahora retornamos el mismo valor

    class Config:
        """Configuración del modelo."""

        json_encoders = {
            List[float]: lambda v: json.dumps(v),
            List[str]: lambda v: json.dumps(v),
        }
        json_decoders = {
            List[float]: lambda v: json.loads(v) if isinstance(v, str) else v,
            List[str]: lambda v: json.loads(v) if isinstance(v, str) else v,
        }
