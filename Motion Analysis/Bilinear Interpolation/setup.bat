@echo off
REM ============================================================
REM  setup.bat  —  Create venv and install dependencies
REM ============================================================

echo.
echo [1/3] Creating virtual environment ...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Could not create virtual environment.
    echo Make sure Python 3.9+ is installed and on PATH.
    pause
    exit /b 1
)

echo.
echo [2/3] Activating virtual environment ...
call venv\Scripts\activate.bat

echo.
echo [3/3] Installing dependencies ...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  Run modes:
echo    run.bat                            synthetic frames
echo    run.bat --video path\to\clip.mp4   single video
echo    run.bat --video a.mp4 --video b.mp4  multiple videos
echo.
echo  Each video produces 10 outputs in output\:
echo    Static PNGs:
echo      lk_theory.png
echo      lk_tracking_<tag>.png
echo      lk_bilinear_detail_<tag>.png
echo      lk_errors_<tag>.png
echo      lk_dense_hsv_<tag>.png
echo      lk_quiver_<tag>.png
echo      lk_eigenvalue_scatter_<tag>.png
echo    Video MP4s:
echo      vid_dense_hsv_<tag>.mp4
echo      vid_trails_<tag>.mp4
echo      vid_quiver_<tag>.mp4
echo ============================================================
pause
