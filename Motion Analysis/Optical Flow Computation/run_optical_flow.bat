@echo off
:: ============================================================================
::  run_optical_flow.bat
::  One-click launcher for the Optical Flow Visualisation script.
::
::  Usage:
::    run_optical_flow.bat                        – auto-detect video in folder
::    run_optical_flow.bat my_video.mp4           – specify an input video
::    run_optical_flow.bat my_video.mp4 out.mp4   – specify input + output
::
::  All outputs are written to the  Output\  sub-folder automatically.
::
::  The script will:
::    1. Verify Python is installed.
::    2. Create the virtual environment (venv\) if it does not exist.
::    3. Install/upgrade all required packages from requirements.txt.
::    4. Run optical_flow.py with any arguments you pass to this batch file.
:: ============================================================================

setlocal EnableDelayedExpansion

:: ── Change to the folder this .bat lives in ──────────────────────────────────
cd /d "%~dp0"

echo.
echo ============================================================
echo  OPTICAL FLOW COMPUTATION  -  Launcher
echo ============================================================

:: ── 1. Check Python ──────────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo         Please install Python 3.9+ from https://python.org
    echo         and make sure "Add Python to PATH" is checked.
    goto :error
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [INFO]  Found: %PYVER%

:: ── 2. Create virtual environment if missing ─────────────────────────────────
if not exist "venv\Scripts\activate.bat" (
    echo [INFO]  Creating virtual environment ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        goto :error
    )
    echo [INFO]  Virtual environment created.
) else (
    echo [INFO]  Virtual environment already exists.
)

:: ── 3. Activate virtual environment ──────────────────────────────────────────
call venv\Scripts\activate.bat

:: ── 4. Install / upgrade dependencies ────────────────────────────────────────
echo [INFO]  Installing / verifying dependencies ...
if exist "requirements.txt" (
    pip install --quiet --upgrade -r requirements.txt
) else (
    pip install --quiet --upgrade opencv-python numpy matplotlib scipy
)
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    goto :error
)
echo [INFO]  Dependencies OK.

:: ── 5. Build argument list ────────────────────────────────────────────────────
::
::  Argument mapping:
::    %1  ->  --input          (optional: path to input video)
::    %2  ->  --output         (optional: main HSV flow video)
::    %3  ->  --plot           (optional: magnitude time-series PNG)
::    %4  ->  --heatmap        (optional: heatmap overlay video)
::    %5  ->  --bbox           (optional: bounding-box video)
::    %6  ->  --polar          (optional: polar histogram PNG)
::    %7  ->  --energy         (optional: accumulated energy heatmap PNG)
::    %8  ->  --arrow-interval (optional: integer, default 5)
::
set ARGS=
if not "%~1"=="" set ARGS=!ARGS! --input "%~1"
if not "%~2"=="" set ARGS=!ARGS! --output "%~2"
if not "%~3"=="" set ARGS=!ARGS! --plot "%~3"
if not "%~4"=="" set ARGS=!ARGS! --heatmap "%~4"
if not "%~5"=="" set ARGS=!ARGS! --bbox "%~5"
if not "%~6"=="" set ARGS=!ARGS! --polar "%~6"
if not "%~7"=="" set ARGS=!ARGS! --energy "%~7"
if not "%~8"=="" set ARGS=!ARGS! --arrow-interval %~8

:: ── 6. Run the script ─────────────────────────────────────────────────────────
echo.
echo [INFO]  Starting optical flow analysis ...
echo [INFO]  All outputs will be saved to: Output\
echo ============================================================
echo.

python optical_flow.py%ARGS%
set EXITCODE=%ERRORLEVEL%

echo.
echo ============================================================
if %EXITCODE%==0 (
    echo  Done!  All files saved to the  Output\  folder:
    echo.
    echo    Videos
    echo    ^• Output\optical_flow_output.mp4        - main HSV flow + region-mask video
    echo    ^• Output\flow_heatmap_output.mp4        - magnitude heatmap overlay video
    echo    ^• Output\motion_bbox_output.mp4         - motion bounding-box video
    echo.
    echo    Graphs / Charts
    echo    ^• Output\flow_magnitude_plot.png        - per-frame magnitude time-series
    echo    ^• Output\polar_direction_histogram.png  - polar flow-direction rose chart
    echo    ^• Output\accumulated_motion_energy.png  - spatial motion energy heatmap
) else (
    echo  Script exited with code %EXITCODE%.  See error messages above.
)
echo ============================================================
echo.

:: ── Deactivate venv ───────────────────────────────────────────────────────────
call venv\Scripts\deactivate.bat 2>nul

pause
exit /b %EXITCODE%

:error
echo.
echo [FATAL] Setup failed. Aborting.
pause
exit /b 1
