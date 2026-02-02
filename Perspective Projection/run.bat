@echo off
chcp 65001 >nul
color 0a
title Real-World Object Dimension Estimation Tool

echo.
echo ===========================================
echo   Real-World Object Dimension Estimation
echo           Using Perspective Projection
echo ===========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher from https://python.org
    echo.
    pause
    exit /b 1
)

REM Check if requirements.txt exists
if not exist requirements.txt (
    echo ERROR: requirements.txt not found
    echo Please ensure requirements.txt is in the current directory
    echo.
    pause
    exit /b 1
)

echo Step 1: Checking virtual environment...
if exist venv (
    echo Virtual environment already exists.
) else (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Please check Python installation and permissions
        echo.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Trying alternative activation method...
    call venv\Scripts\activate
)

echo.
echo Step 3: Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Please check internet connection and try again
    echo.
    pause
    exit /b 1
)
echo Dependencies installed successfully.

echo.
echo Step 4: Starting measurement tool...
echo ===========================================
echo.

REM Check if measure_object.py exists
if exist measure_object.py (
    python measure_object.py
) else (
    echo ERROR: measure_object.py not found
    echo Please ensure measure_object.py is in the current directory
    echo.
    pause
    exit /b 1
)

echo.
echo ===========================================
echo   Measurement tool execution completed
echo ===========================================
echo.
echo Press any key to exit...
pause >nul
