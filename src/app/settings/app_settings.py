"""
Application Settings Management for EPRA Backend

This module centralizes and manages environment-specific settings for the EEG data processing backend.
Using Pydantic's BaseSettings, it defines clear and scalable configuration profiles for development and production,
facilitating maintainability, reproducibility, and adaptability in different deployment contexts.

Methodology:
- Two distinct settings classes: DevSettings and ProdSettings.
- Common parameters inherited from a shared BaseSettingsConfig.
- Environment variable-driven selection for flexible deployment.
"""

import os

from pydantic_settings import BaseSettings


class BaseSettingsConfig(BaseSettings):
    """
    Base configuration class containing common settings.
    """

    PROJECT_NAME: str = "EPRA - EEG Processing System"
    VERSION: str = "1.0.0"
    DATABASE_URL: str = "sqlite:///database.db"
    ENVIRONMENT: str = "development"  # Options: development, production

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class DevSettings(BaseSettingsConfig):
    """
    Development environment settings.
    """

    DEBUG: bool = True
    DATABASE_URL: str = "sqlite:///dev_database.db"


class ProdSettings(BaseSettingsConfig):
    """
    Production environment settings.
    """

    DEBUG: bool = False
    DATABASE_URL: str = "sqlite:///prod_database.db"  # Replace with actual production DB (e.g., PostgreSQL)


# ------------------- Settings Loader -------------------


def get_settings() -> BaseSettingsConfig:
    """
    Dynamically loads the appropriate settings based on the ENVIRONMENT variable.

    Returns:
        BaseSettingsConfig: The loaded settings instance.
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    if environment == "production":
        return ProdSettings()
    return DevSettings()


# Singleton instance used across the application
app_settings = get_settings()
