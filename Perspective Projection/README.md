# Real-World Object Dimension Estimation Tool

A Python-based tool that estimates real-world 2D dimensions of objects from single images using perspective projection and homography transformations.

## Overview

This tool implements advanced computer vision techniques to measure real-world dimensions of objects in photographs. It uses the pinhole camera model, perspective projection geometry, and homography transformations to convert pixel measurements to physical dimensions.

## Methodology

### Mathematical Foundation

The tool implements two key mathematical approaches:

1. **Perspective Projection**: Uses camera intrinsic parameters to relate pixel coordinates to real-world measurements
2. **Homography Transformation**: Computes planar perspective transformations to map image points to real-world coordinates

### Key Equations

**Homography Matrix Computation:**
```
H = findHomography(src_points, dst_points)
```

**Real-World Coordinate Transformation:**
```
dst_coords = perspectiveTransform(src_coords, H)
```

**Scale Factor Calculation:**
```
scale_factor = reference_length_mm / reference_pixel_length
```

### Camera Parameters (Samsung M34)

- **Focal Length**: 27mm equivalent
- **Sensor Size**: 1/2.76" (approx. 6.3mm × 4.7mm)
- **Default Resolution**: 4080 × 3060 pixels
- **EXIF Support**: Automatically extracts camera parameters from image metadata

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Quick Setup

1. **Run the batch file (Windows):**
   ```cmd
   run.bat
   ```

2. **Manual setup:**
   ```cmd
   # Create virtual environment
   python -m venv venv
   
   # Activate environment
   venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the tool
   python measure_object.py
   ```

## Usage

### Command Line Interface

```bash
# Basic usage
python measure_object.py --image photo.jpg --reference 100

# Create sample image for testing
python measure_object.py --create-sample

# Specify reference length
python measure_object.py --image photo.jpg --reference 150
```

### Interactive Mode

When no command-line arguments are provided, the tool runs in interactive mode:

1. **Automatic Image Detection**: Scans `Input_images/` directory for images
2. **EXIF Data Extraction**: Automatically reads camera parameters from image metadata
3. **Interactive Point Selection**: User-friendly interface for selecting measurement points

### Measurement Process

1. **Load Image**: Tool opens the selected image
2. **Select Object Corners**: Click 4 corners of the object (top-left, top-right, bottom-right, bottom-left)
3. **Select Reference Points**: Click 2 points representing a known reference length
4. **Compute Dimensions**: Tool calculates real-world width and height
5. **View Results**: Display measurements with visual overlay

## Input Requirements

### Image Quality

- **Resolution**: Minimum 800×600 pixels recommended
- **Focus**: Object should be in sharp focus
- **Lighting**: Even illumination, minimal shadows
- **Angle**: Camera should be as perpendicular to object as possible

### Reference Measurement

- **Known Length**: Must have at least one known dimension in the image
- **Accuracy**: Reference measurement should be precise
- **Placement**: Reference should be on the same plane as the object
- **Visibility**: Reference points should be clearly identifiable

### Camera Position

- **Distance**: 30cm to 2m from object (avoid extreme close-ups)
- **Angle**: Camera optical axis should be normal to object plane
- **Stability**: Use tripod if possible to avoid motion blur

## Output

### Console Output

```
Real-World Object Dimensions
============================================================
Estimated Width:  152.34 mm (15.23 cm)
Estimated Height: 98.67 mm (9.87 cm)
Reference Used:   100.00 mm
Scale Factor:     0.254 mm/px
============================================================
```

### Output Image

The tool saves a measurement result image to `Results/measurement_output.png` containing:

- **Object Outline**: Green rectangle around measured object
- **Reference Line**: Red line showing reference measurement
- **Dimension Labels**: Width and height annotations
- **Camera Info**: Focal length and resolution details

## Camera Parameter Configuration

### Automatic EXIF Extraction

The tool automatically extracts camera parameters from image EXIF data:

```python
# Extract focal length
if 'FocalLength' in exif_data:
    focal_length = exif_data['FocalLength']
    if isinstance(focal_length, tuple):
        self.focal_length_mm = focal_length[0] / focal_length[1]

# Extract resolution
if 'ImageWidth' in exif_data and 'ImageHeight' in exif_data:
    self.image_width_px = exif_data['ImageWidth']
    self.image_height_px = exif_data['ImageHeight']
```

### Manual Configuration

For cameras without EXIF data or custom parameters:

```python
# Modify in measure_object.py
self.focal_length_mm = 26.0      # Focal length in mm
self.sensor_width_mm = 6.3       # Sensor width in mm
self.sensor_height_mm = 4.7      # Sensor height in mm
self.image_width_px = 4000       # Image width in pixels
self.image_height_px = 3000      # Image height in pixels
```

## Technical Implementation

### Core Classes

- **ObjectDimensionEstimator**: Main class handling all measurement logic
- **Point Selection**: Interactive mouse callback system for point selection
- **Homography Computation**: Perspective transformation calculations
- **EXIF Processing**: Metadata extraction and parameter parsing

### Key Functions

```python
# Point selection interface
def select_points_interactively(self, image, mode)

# Homography matrix computation
def compute_homography(self, object_corners, reference_length_mm)

# Dimension estimation
def estimate_dimensions(self, object_corners, reference_length_mm)

# EXIF data extraction
def extract_exif_data(self, image_path)

# Measurement overlay drawing
def draw_measurement_overlay(self, image, object_corners, width_mm, height_mm, reference_length_mm)
```

### Algorithm Flow

1. **Image Loading**: Load and validate input image
2. **Parameter Extraction**: Extract or set camera parameters
3. **Point Selection**: Interactive selection of object corners and reference points
4. **Homography Computation**: Calculate perspective transformation matrix
5. **Dimension Calculation**: Transform coordinates and compute real-world dimensions
6. **Result Visualization**: Draw measurement overlay and save results

## Troubleshooting

### Common Issues

**Image Not Found:**
- Ensure image is in `Input_images/` directory or use `--image` parameter
- Check file path and permissions

**Invalid Points:**
- Ensure all 4 corners are selected for the object
- Make sure reference points are clearly separated
- Points should be within image boundaries

**Poor Accuracy:**
- Check camera angle (should be perpendicular to object)
- Ensure reference measurement is accurate
- Verify object and reference are on the same plane
- Use higher resolution images

**EXIF Data Not Found:**
- Tool will use default Samsung M34 parameters
- Consider manually setting camera parameters
- Check if image has been processed/edited (may remove EXIF)

### Error Messages

- **"Need exactly 4 corner points"**: Select all 4 corners of the object
- **"Need exactly 2 reference points"**: Select 2 points for reference measurement
- **"Could not compute dimensions"**: Check point selection and reference length
- **"Image file not found"**: Verify image path and file existence

## Best Practices

### Image Capture

1. **Use Good Lighting**: Avoid shadows and reflections
2. **Keep Camera Steady**: Use tripod or stable surface
3. **Fill Frame**: Make object occupy significant portion of image
4. **Avoid Distortion**: Keep object centered, avoid wide-angle effects

### Measurement Accuracy

1. **Precise Reference**: Use accurate reference measurements
2. **Clear Points**: Select easily identifiable corners
3. **Multiple Measurements**: Take several measurements and average results
4. **Validation**: Measure known objects to verify accuracy

### Camera Settings

1. **Auto Focus**: Ensure object is in sharp focus
2. **Flash**: Avoid using flash if possible (causes reflections)
3. **Resolution**: Use highest available resolution
4. **Stability**: Minimize camera movement during capture

## Advanced Features

### Batch Processing

For processing multiple images, modify the script to loop through image directories:

```python
# Process all images in directory
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for image_file in image_files:
    process_image(os.path.join(input_dir, image_file))
```

### Custom Camera Profiles

Add support for different camera models by creating camera parameter profiles:

```python
CAMERA_PROFILES = {
    'samsung_m34': {'focal_length': 26.0, 'sensor_width': 6.3, 'sensor_height': 4.7},
    'iphone_15': {'focal_length': 4.25, 'sensor_width': 5.8, 'sensor_height': 4.3},
    # Add more profiles
}
```

### Integration with Other Tools

The measurement results can be exported for use in other applications:

```python
# Export to JSON
import json
results = {
    'width_mm': width_mm,
    'height_mm': height_mm,
    'reference_mm': reference_length_mm,
    'scale_factor': scale_factor
}
with open('measurement_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Dependencies

### Required Packages

- **opencv-python**: Image processing and computer vision operations
- **numpy**: Numerical computations and matrix operations
- **pillow**: Image handling and EXIF data extraction
- **matplotlib**: Optional plotting and visualization
- **exifread**: Alternative EXIF data reading (optional)

### Installation Notes

All dependencies are included in `requirements.txt` and installed automatically by `run.bat`.

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For questions, issues, or feature requests:

1. Check the troubleshooting section
2. Review the technical implementation
3. Verify your camera parameters and image quality
4. Create an issue with detailed information

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Perspective Projection](https://en.wikipedia.org/wiki/Perspective_(graphical))
- [Homography Transformations](https://en.wikipedia.org/wiki/Homography)

- [Camera Calibration](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html)
