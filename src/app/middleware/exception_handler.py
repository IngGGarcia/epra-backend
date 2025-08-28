"""
Global Exception Handler Middleware

This module provides a global exception handler for the FastAPI application.
It centralizes the handling of HTTP exceptions and other errors,
providing consistent error responses across the application.
"""

import logging
from typing import Callable

from fastapi import Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for HTTP exceptions and other errors.

    Args:
        request: The incoming request.
        exc: The exception that was raised.

    Returns:
        JSONResponse: A JSON response with the error details.
    """
    # Log the error with more context
    logger.error(
        f"Error processing request {request.method} {request.url}: {str(exc)}",
        exc_info=True,
    )

    # Handle SQLAlchemy errors
    if isinstance(exc, SQLAlchemyError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Database error occurred",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "source": "database",
            },
        )

    # Handle custom application exceptions (like SAMFileError)
    # Check if it's a custom exception by looking at the module
    exc_module = type(exc).__module__
    if exc_module and exc_module.startswith("src.app"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(exc),
                "error_type": type(exc).__name__,
                "source": exc_module.replace("src.app.", "").replace(".", "/"),
            },
        )

    # Handle other exceptions
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "source": "system",
        },
    )
