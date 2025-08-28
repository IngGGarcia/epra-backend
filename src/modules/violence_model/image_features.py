"""
Módulo para cargar y manejar características de imágenes.

Este módulo proporciona funciones para cargar y procesar las características
numéricas de las imágenes utilizadas en el modelo de violencia. Al cargar,
expande la columna 'features' (lista) en columnas numéricas separadas.
"""

from pathlib import Path

import pandas as pd


def load_image_features() -> pd.DataFrame:
    """
    Carga las características de las imágenes desde el archivo parquet y expande la columna 'features'.

    Returns:
        pd.DataFrame: DataFrame con las características numéricas de las imágenes, una columna por feature.

    Raises:
        FileNotFoundError: Si no se encuentra el archivo de características
    """
    # Construir la ruta al archivo
    file_path = Path(__file__).parent / "image_features.parquet"

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de características en {file_path}"
        )

    # Cargar características
    features = pd.read_parquet(file_path)

    # Expandir la columna 'features' en columnas numéricas separadas
    features_expanded = features["features"].apply(pd.Series)
    features_expanded = features_expanded.add_prefix("feature_")
    features_final = pd.concat([features[["image_id"]], features_expanded], axis=1)

    return features_final
