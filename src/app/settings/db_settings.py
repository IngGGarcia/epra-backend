"""
Database Settings Configuration.

Defines functions to create the database engine and access model metadata.
"""

from sqlalchemy.engine import Engine
from sqlmodel import SQLModel, create_engine


def get_engine(database_url: str) -> Engine:
    """
    Creates and returns the SQLModel engine using the provided database URL.
    """
    return create_engine(database_url, echo=True)


def get_metadata():
    """
    Returns the SQLModel metadata.
    """
    return SQLModel.metadata
