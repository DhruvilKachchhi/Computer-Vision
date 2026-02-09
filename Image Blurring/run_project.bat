@echo off
REM Image Blurring and Convolution Theorem Demonstration
REM Windows batch file to set up virtual environment and run the project

echo.
echo ================================================
echo Image Blurring and Convolution Theorem Project
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not available in PATH
    echo Please install Python 3.7 or higher and ensure it's in your system PATH
    pause
    exit /b 1
)

echo Python version: %python --version%
echo.

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        echo Please ensure Python is installed and available in PATH
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    echo Trying alternative activation method...
    call venv\Scripts\activate
    if errorlevel 1 (
        echo Error: Virtual environment activation failed
        pause
        exit /b 1
    )
)

REM Install requirements
echo Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install requirements
    echo Please check requirements.txt and internet connection
    pause
    exit /b 1
)

echo Requirements installed successfully.

REM Run the main script
echo.
echo Running main.py...
echo.
python src/main.py %*
if errorlevel 1 (
    echo Error: Failed to run main.py
    echo Please check the script and dependencies
    pause
    exit /b 1
)

echo.
echo ================================================
echo Project execution completed successfully!
echo Results saved to the 'Results' folder.
echo ================================================
echo.

REM Pause to allow user to read output
pause
