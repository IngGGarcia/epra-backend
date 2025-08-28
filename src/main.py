"""
Main API Application for EPRA System

This module initializes the FastAPI application, setting up core middleware and configuration parameters
for the EPRA backend system. EPRA encompasses various modules, including but not limited to EEG data processing,
ensuring a modular, scalable, and reproducible architecture for advanced data management and analysis.

Methodology:
- FastAPI is used to build a RESTful API, providing endpoints for multiple modules.
- CORS middleware is configured to allow cross-origin communication, adjustable per environment.
- Application settings are dynamically loaded based on the deployment context (development, production).
- Database migrations and schema control are managed externally to guarantee consistency and traceability.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.app.api import router
from src.app.middleware.exception_handler import http_exception_handler
from src.app.settings.app_settings import app_settings

# ------------------- Initialize FastAPI Application -------------------

app = FastAPI(
    title=app_settings.PROJECT_NAME,
    version=app_settings.VERSION,
    description="EPRA System Backend - Managing multi-module data processing and analysis.",
    debug=app_settings.DEBUG,
)

# ------------------- Configure CORS Middleware -------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
    if app_settings.DEBUG
    else ["https://your-production-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Configure Exception Handler -------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for all unhandled exceptions.
    """
    return await http_exception_handler(request, exc)


# ------------------- Root Endpoint -------------------


@app.get("/")
def root():
    """
    Root endpoint providing system status and environment details.
    """
    return {
        "message": f"{app_settings.PROJECT_NAME} is operational. Access /docs for API documentation.",
        "version": app_settings.VERSION,
        "environment": app_settings.ENVIRONMENT,
    }


# ------------------- Include API Routes -------------------

app.include_router(router)
