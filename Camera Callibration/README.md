# Camera Calibration and 3D Visualization Project

This project performs camera calibration using chessboard images and creates comprehensive 3D visualizations of the calibration results.

## Quick Start

**One-Click Operation:**
Double-click `run_camera_calibration.bat` to run the entire project automatically!

## Project Structure

```
Camera Callibration/
├── Callibration Images/          # Input chessboard images
├── Results/                      # Output directory
│   ├── npy_files/               # Calibration parameters
│   ├── Undistorted_Images/      # Processed images
│   └── 3D_Plot/                 # 3D visualizations
├── camera_callibration.py       # Main calibration script
├── visualize_extrinsic_parameters.py  # 3D extrinsic visualization
├── create_3d_graph_2d_plot.py   # 3D graph of undistorted images
├── run_camera_calibration.bat   # One-click execution script
└── README.md                    # This file
```

## Features

### 1. Camera Calibration
- **Chessboard Detection:** Automatically detects chessboard corners in calibration images
- **Parameter Calculation:** Computes intrinsic and extrinsic camera parameters
- **Image Processing:** Undistorts and saves all calibration images with detected corners
- **Data Organization:** Saves all parameters in organized folder structure

### 2. 3D Visualization of Extrinsic Parameters
- **Proper 3D Orientation:** Each chessboard plane is oriented exactly as observed in the original photos
- **Camera Coordinate Frame:** Shows camera position and orientation at origin
- **Multiple Views:** Top, front, and side orthogonal views
- **Spatial Relationships:** Demonstrates relative camera viewpoints

### 3. 3D Graph of Undistorted Images
- **3D Surface Plots:** Shows pixel intensity as height in 3D space
- **Multiple Visualization Methods:** Matplotlib and Plotly outputs
- **2D Projections:** Different perspectives of the 3D data

## Output Files

### Calibration Parameters (`Results/npy_files/`)
- `camera_matrix.npy` - Intrinsic camera matrix
- `dist_coeffs.npy` - Distortion coefficients
- `rotation_vectors.npy` - Extrinsic rotation vectors
- `translation_vectors.npy` - Extrinsic translation vectors
- `object_points.npy` - World coordinates of chessboard corners
- `image_points.npy` - Detected corner coordinates in images

### Processed Images (`Results/Undistorted_Images/`)
- `undistorted_Image (1).jpeg` through `undistorted_Image (18).jpeg`
- All images with chessboard corners detected and drawn

### 3D Visualizations (`Results/3D_Plot/`)
- `extrinsic_visualization.png` - Main 3D extrinsic parameter visualization
- `extrinsic_top_view_(x-y).png` - Top view of extrinsic parameters
- `extrinsic_front_view_(y-z).png` - Front view of extrinsic parameters
- `extrinsic_side_view_(x-z).png` - Side view of extrinsic parameters
- Various 3D graphs and plots of undistorted images

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- Plotly

Install dependencies:
```bash
pip install opencv-python numpy matplotlib plotly
```

## Usage

### Option 1: One-Click (Recommended)
1. Place your chessboard calibration images in the `Callibration Images/` folder
2. Double-click `run_camera_calibration.bat`
3. The script will run calibration, create visualizations, and open the Results folder

### Option 2: Manual Execution
1. **Run Calibration:**
   ```bash
   python camera_callibration.py
   ```

2. **Create 3D Visualization:**
   ```bash
   python visualize_extrinsic_parameters.py
   ```

3. **Create 3D Graphs:**
   ```bash
   python create_3d_graph_2d_plot.py
   ```

## Technical Details

### Chessboard Configuration
- **Size:** 9x6 internal corners
- **Square Size:** 2.5 cm (0.025 meters)
- **Images:** 18 calibration photos provided

### Coordinate Systems
- **World Coordinates:** Chessboard plane with origin at first corner
- **Camera Coordinates:** Camera-centered coordinate system
- **Transformation:** `P_camera = R * P_world + t`

### Visualization Features
- **Equal Aspect Ratio:** Ensures accurate spatial representation
- **Color Coding:** Each plane has unique color for distinction
- **Proper Scaling:** All measurements in meters
- **Camera Frame:** Colored axes (X=red, Y=green, Z=blue)

## Troubleshooting

### Common Issues
1. **No Chessboard Detected:** Ensure good lighting and clear chessboard pattern
2. **Missing Dependencies:** Run `pip install -r requirements.txt`
3. **Permission Errors:** Run as administrator if needed

### Error Messages
- **"No valid chessboard corners found":** Check image quality and chessboard visibility
- **"File not found":** Ensure calibration images are in the correct folder
- **"ImportError":** Install missing Python packages

## Project Files

### Main Scripts
- `camera_callibration.py` - Complete calibration pipeline
- `visualize_extrinsic_parameters.py` - 3D extrinsic visualization
- `create_3d_graph_2d_plot.py` - 3D image analysis

### Utility Files
- `run_camera_calibration.bat` - One-click execution
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## License

This project is open source and available under the MIT License.

## Support

For questions or issues, please check:
1. The troubleshooting section above
2. Ensure all dependencies are installed
3. Verify chessboard images are properly formatted
4. Check that images are placed in the correct folder

## Notes

- The project uses OpenCV's built-in chessboard detection
- All visualizations use Matplotlib for consistency
- Results are saved automatically to organized folders
- The batch file provides error handling and user feedback