"""
Database module initialization.

This module exposes the database session and engine for use throughout the application.
"""

from .db_session import engine, get_session

__all__ = ["get_session", "engine"]
