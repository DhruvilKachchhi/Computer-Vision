@echo off
setlocal EnableDelayedExpansion

:: Enable UTF-8 for Unicode console output
chcp 65001 > nul

title Uncalibrated Stereo Vision - Setup and Run

:: Force Python to use UTF-8 for stdout/stderr
set PYTHONUTF8=1

set "BASE=%~dp0"
set "VENV=%BASE%venv"
set "PY=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"

echo.
echo ════════════════════════════════════════════════════════════════════════
echo   Uncalibrated Stereo Vision  ^|  Samsung Galaxy M34
echo   Baseline = 1 ft  ^|  Ground-Truth Distance = 7 ft
echo ════════════════════════════════════════════════════════════════════════
echo.

:: ────────────────────────────────────────────────────────────────────────────
:: [1]  Create virtual environment (skip if it already exists)
:: ────────────────────────────────────────────────────────────────────────────
if not exist "%PY%" (
    echo [1]  Creating virtual environment in:
    echo      %VENV%
    python -m venv "%VENV%"
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR]  Failed to create virtual environment.
        echo          Ensure Python 3.9+ is installed and on the system PATH.
        pause
        exit /b 1
    )
    echo      Done.
) else (
    echo [1]  Virtual environment already exists — skipping creation.
)

:: ────────────────────────────────────────────────────────────────────────────
:: [2]  Activate virtual environment
:: ────────────────────────────────────────────────────────────────────────────
echo [2]  Activating virtual environment …
call "%VENV%\Scripts\activate.bat"
if !errorlevel! neq 0 (
    echo [ERROR]  Could not activate virtual environment.
    pause
    exit /b 1
)

:: ────────────────────────────────────────────────────────────────────────────
:: [3]  Install / upgrade dependencies
:: ────────────────────────────────────────────────────────────────────────────
echo [3]  Installing / verifying dependencies from requirements.txt …
"%PIP%" install --quiet --upgrade pip
"%PIP%" install --quiet -r "%BASE%requirements.txt"
if !errorlevel! neq 0 (
    echo.
    echo [ERROR]  pip install failed.  Check requirements.txt and your internet connection.
    pause
    exit /b 1
)
echo      Done.

:: ────────────────────────────────────────────────────────────────────────────
:: [4]  Run the stereo-vision implementation
:: ────────────────────────────────────────────────────────────────────────────
echo.
echo [4]  Running stereo_vision.py …
echo ────────────────────────────────────────────────────────────────────────
"%PY%" "%BASE%stereo_vision.py"
if !errorlevel! neq 0 (
    echo.
    echo [ERROR]  stereo_vision.py terminated with an error.
    pause
    exit /b 1
)
echo ────────────────────────────────────────────────────────────────────────

:: ────────────────────────────────────────────────────────────────────────────
:: [5]  Generate the Word derivation report
:: ────────────────────────────────────────────────────────────────────────────
echo.
echo [5]  Generating Word report (create_word_report.py) …
"%PY%" "%BASE%create_word_report.py"
if !errorlevel! neq 0 (
    echo [WARNING]  Word report generation encountered an error.
    echo            Stereo output (output_stereo.png) was still produced.
) else (
    echo      Done.
)

:: ────────────────────────────────────────────────────────────────────────────
:: Summary
:: ────────────────────────────────────────────────────────────────────────────
echo.
echo ════════════════════════════════════════════════════════════════════════
echo   [DONE]  All steps completed.
echo.
echo   Output files:
echo     • output_stereo.png   — annotated stereo pair with detected square
echo     • results.json        — computed matrices (F, E, R, t) and metrics
echo     • stereo_report.docx  — derivation report (Word)
echo.
echo   Location: %BASE%
echo ════════════════════════════════════════════════════════════════════════
echo.
pause
