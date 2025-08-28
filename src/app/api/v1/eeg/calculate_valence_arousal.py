"""
EEG Valence and Arousal Calculation API Endpoint

This endpoint provides functionality to calculate valence and arousal values
from EEG files using state-of-the-art research-based methods. It returns
a DataFrame structure similar to SAM results but with image_number instead
of image_id, containing EEG-derived emotional measurements.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
import pandas as pd

from src.app.services.evaluation.verify_eeg_file import EEGFileError, verify_eeg_file
from src.app.services.evaluation.verify_eeg_file.eeg_valence_arousal_service import (
    EEGValenceArousalService,
)

router = APIRouter()


@router.post(
    "/calculate/",
    summary="Calculate valence and arousal from EEG file",
    description=(
        "Processes an EEG CSV file and calculates valence and arousal values "
        "for each image exposure segment using research-based methods. "
        "Returns a DataFrame with image_number, valence, arousal columns "
        "similar to SAM results but derived from EEG signal analysis."
    ),
    responses={
        200: {
            "description": "Successfully calculated valence and arousal",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": [
                            {"image_number": 1, "valence": 6.23, "arousal": 5.78},
                            {"image_number": 2, "valence": 4.91, "arousal": 7.12},
                            {"image_number": 3, "valence": 7.45, "arousal": 3.67},
                        ],
                        "summary": {
                            "total_images": 3,
                            "method": "heuristic",
                            "sampling_rate": 128,
                            "valence_range": [4.91, 7.45],
                            "arousal_range": [3.67, 7.12],
                        },
                    }
                }
            },
        },
        400: {"description": "Invalid EEG file or processing error"},
        422: {"description": "Invalid file format or missing required fields"},
    },
)
async def calculate_valence_arousal_endpoint(
    file: Annotated[UploadFile, File(description="EEG CSV file to process")],
    method: Annotated[str, Form(description="Calculation method")] = "heuristic",
) -> JSONResponse:
    """
    Calculate valence and arousal from uploaded EEG file.

    This endpoint processes EEG CSV files and extracts emotional measurements
    using state-of-the-art signal processing methods including:
    - Power Spectral Density (PSD) analysis
    - Differential Entropy (DE) features
    - Hemispheric asymmetry analysis
    - Heuristic or machine learning-based classification

    Args:
        file: EEG CSV file with required columns: Timestamp, EEG.AF3, EEG.F3, EEG.AF4, EEG.F4
        method: Calculation method - 'heuristic' (default) or 'ml'

    Returns:
        JSON response containing valence/arousal data and processing summary

    Raises:
        HTTPException: If file processing fails or invalid parameters provided
    """
    # Validate method parameter
    if method not in ["heuristic", "ml"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Method must be 'heuristic' or 'ml'",
        )

    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No file provided"
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File must be a CSV file",
        )

    try:
        # Calculate valence and arousal using integrated verify_eeg_file
        df_results = verify_eeg_file(file.file, file.filename)

        # Convert DataFrame to list of dictionaries for JSON response
        results_data = df_results.to_dict("records")

        # Generate summary statistics
        summary = {
            "total_images": len(df_results),
            "method": method,
            "valence_range": [
                float(df_results["valence"].min()),
                float(df_results["valence"].max()),
            ],
            "arousal_range": [
                float(df_results["arousal"].min()),
                float(df_results["arousal"].max()),
            ],
            "valence_mean": float(df_results["valence"].mean()),
            "arousal_mean": float(df_results["arousal"].mean()),
            "valence_std": float(df_results["valence"].std()),
            "arousal_std": float(df_results["arousal"].std()),
        }

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "data": results_data,
                "summary": summary,
                "message": f"Successfully calculated valence/arousal for {len(df_results)} images using {method} method",
            },
        )

    except EEGFileError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"EEG file processing error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


@router.post(
    "/analyze-features/",
    summary="Analyze EEG features for debugging and research",
    description=(
        "Provides detailed analysis of EEG features extracted from the file "
        "for research and debugging purposes. Shows feature counts, examples, "
        "and technical details about the signal processing."
    ),
)
async def analyze_eeg_features_endpoint(
    file: Annotated[UploadFile, File(description="EEG CSV file to analyze")],
    method: Annotated[str, Form(description="Analysis method")] = "heuristic",
) -> JSONResponse:
    """
    Analyze EEG features from uploaded file for research purposes.

    This endpoint provides detailed technical information about the
    feature extraction process, including:
    - Sampling rate estimation
    - Feature counts by type
    - Example feature values
    - Signal processing parameters

    Args:
        file: EEG CSV file to analyze
        method: Analysis method ('heuristic' or 'ml')

    Returns:
        JSON response with detailed feature analysis
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Valid CSV file required",
        )

    try:
        service = EEGValenceArousalService(method=method)
        feature_summary = service.get_feature_summary(file.file, file.filename)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "analysis": feature_summary,
                "message": "Feature analysis completed successfully",
            },
        )

    except EEGFileError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"EEG file analysis error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis error: {str(e)}",
        )
