@echo off
echo ================================================
echo Camera Calibration and 3D Visualization Project
echo ================================================
echo.

echo Activating virtual environment...
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate
    echo Virtual environment activated.
) else (
    echo Warning: Virtual environment not found. Using system Python.
)

echo.
echo Step 1: Running Camera Calibration...
echo ----------------------------------------
python camera_callibration.py

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Camera calibration failed!
    echo Make sure all required packages are installed:
    echo - opencv-python
    echo - numpy
    echo - matplotlib
    echo - plotly
    pause
    exit /b 1
)

echo.
echo Step 2: Creating 3D Visualization of Extrinsic Parameters...
echo ------------------------------------------------------------
python visualize_extrinsic_parameters.py

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: 3D visualization failed!
    pause
    exit /b 1
)

echo.
echo Step 3: Creating 3D Graph of Undistorted Images...
echo --------------------------------------------------
python create_3d_graph_2d_plot.py

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: 3D graph creation failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo Project completed successfully!
echo ================================================
echo.
echo Output files created in:
echo - Results/npy_files/     (calibration parameters)
echo - Results/Undistorted_Images/  (processed images)
echo - Results/3D_Plot/       (3D visualizations)
echo.
echo Press any key to open the Results folder...
pause >nul
explorer "Results"
