@echo off
echo.
echo ================================================
echo  Edge Detection Project - Batch Runner
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if required Python packages are installed
echo Checking Python dependencies...
python -c "import cv2, numpy, matplotlib, scipy" 2>nul
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install opencv-python-headless numpy scipy matplotlib
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install required packages.
        pause
        exit /b 1
    )
)

REM Check if SAM2 is available (optional)
echo Checking SAM2 availability...
python -c "import sys; sys.path.insert(0, 'sam2'); import torch; from sam2.build_sam import build_sam2; from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator" 2>nul
if %errorlevel% equ 0 (
    echo SAM2 is available - will use real model inference.
) else (
    echo SAM2 not found - will use simulation mode.
)

echo.
echo ================================================
echo  Running Edge Detection Pipeline
echo ================================================
echo.

REM Create directories if they don't exist
if not exist "input_images" mkdir input_images
if not exist "results" mkdir results

echo Step 1: Running Canny + Fill segmentation...
echo ----------------------------------------------
python canny_fill_thermal.py
if %errorlevel% neq 0 (
    echo ERROR: Canny + Fill segmentation failed.
    pause
    exit /b 1
)
echo.

echo Step 2: Running SAM2 thermal segmentation...
echo --------------------------------------------
python sam2_thermal.py
if %errorlevel% neq 0 (
    echo ERROR: SAM2 segmentation failed.
    pause
    exit /b 1
)
echo.

echo Step 3: Running comparison analysis...
echo --------------------------------------
python sam2_canny_comparison.py
if %errorlevel% neq 0 (
    echo ERROR: Comparison analysis failed.
    pause
    exit /b 1
)
echo.

echo ================================================
echo  Pipeline completed successfully!
echo ================================================
echo.
echo Output files saved to: results/
echo.
echo Files generated:
echo   - canny_fill_edge_detection.png
echo   - canny_fill_mask.png
echo   - canny_fill_overlay.png
echo   - sam2_segmentation_steps.png
echo   - sam2_mask.png
echo   - sam2_overlay.png
echo   - sam2_edges.png
echo   - sam2_vs_canny_overview.png
echo   - sam2_vs_canny_metrics.png
echo.
echo To view results, check the results/ folder.
echo.
echo Press any key to open the results folder...
pause >nul
explorer results