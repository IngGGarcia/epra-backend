"""
Database Session Dependency.

Provides the SQLModel session dependency in a straightforward, maintainable way
without introducing unnecessary complexity.
"""

from typing import Generator

from sqlmodel import Session

from src.app.settings.app_settings import app_settings
from src.app.settings.db_settings import get_engine

# Initialize engine here (single source)
engine = get_engine(app_settings.DATABASE_URL)


def get_session() -> Generator:
    """
    Dependency function yielding database sessions.
    """
    with Session(engine) as session:
        yield session
