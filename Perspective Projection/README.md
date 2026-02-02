# Real-World Object Dimension Estimation Tool

A Python-based tool that estimates real-world 2D dimensions of objects from single images using perspective projection and homography transformations.

## 🚀 **New Features**

### ✨ **Multiple Image Support**
- Automatically detects and lists all available images in `Input_images/` directory
- Interactive selection menu for choosing which image to process
- Unique output files for each processed image

### 📏 **Smart Reference Lengths**
- **image1**: Automatically uses 215.9mm reference length
- **image2**: Automatically uses 102.0mm reference length  
- Other images: Uses configurable default reference length
- All dimensions displayed in **millimeters, centimeters, AND inches**

### 🎯 **Enhanced Output**
- Dimensions displayed in inches on the resulting image overlay
- Console output shows all three units (mm, cm, in)
- Clear, professional measurement labels

## Overview

This tool implements advanced computer vision techniques to measure real-world dimensions of objects in photographs. It uses the pinhole camera model, perspective projection geometry, and homography transformations to convert pixel measurements to physical dimensions.

## 📋 **Quick Start**

### **1. Installation**
```bash
# Windows (recommended)
run.bat

# Or manually
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### **2. Usage**
```bash
# Interactive mode (recommended)
python measure_object.py

# Process specific image
python measure_object.py --image Input_images/image1.jpeg

# Create sample image for testing
python measure_object.py --create-sample
```

### **3. Measurement Process**
1. **Select Image**: Choose from available images in Input_images/
2. **Select Object Corners**: Click 4 corners (top-left, top-right, bottom-right, bottom-left)
3. **Select Reference Points**: Click 2 points representing known reference length
4. **View Results**: Get dimensions in mm, cm, and inches

## 📊 **Example Output**

### **Console Display**
```
============================================================
REAL-WORLD OBJECT DIMENSIONS
============================================================
Estimated Width:  779.18 mm (77.92 cm) (30.68 in)
Estimated Height: 1154.35 mm (115.43 cm) (45.45 in)
Reference Used:   215.9 mm (8.50 in)
Scale Factor:     1.943075 mm/px
============================================================
```

### **Image Overlay**
- **Object Outline**: Green rectangle around measured object
- **Reference Line**: Red line showing reference measurement in inches
- **Dimension Labels**: Width and height in inches (e.g., "W: 30.68in", "H: 45.45in")
- **Camera Info**: Focal length and resolution details

## 🎯 **Image-Specific Reference Lengths**

The tool automatically detects which image you're processing and applies the correct reference length:

| Image Pattern | Reference Length | Example Usage |
|---------------|------------------|---------------|
| `image1*` | **215.9mm** (8.50in) | Standard reference objects |
| `image2*` | **102.0mm** (4.02in) | Smaller reference objects |
| Other images | **Configurable default** | Custom reference objects |

## 📸 **Input Requirements**

### **Image Quality**
- **Resolution**: Minimum 800×600 pixels recommended
- **Focus**: Object should be in sharp focus
- **Lighting**: Even illumination, minimal shadows
- **Angle**: Camera should be as perpendicular to object as possible

### **Reference Measurement**
- **Known Length**: Must have at least one known dimension in the image
- **Accuracy**: Reference measurement should be precise
- **Placement**: Reference should be on the same plane as the object
- **Visibility**: Reference points should be clearly identifiable

### **Camera Position**
- **Distance**: 30cm to 2m from object (avoid extreme close-ups)
- **Angle**: Camera optical axis should be normal to object plane
- **Stability**: Use tripod if possible to avoid motion blur

## ⚙️ **Configuration**

### **Reference Length (Editable Variable)**
Located in `measure_object.py` line 34-36:
```python
# Reference length for calibration (editable variable)
# Change this value to match your reference object's real-world length in millimeters
self.reference_length_mm = 102  # References: 102mm/ 215.9mm 
```

### **Camera Parameters**
The tool automatically extracts camera parameters from EXIF data, or uses these defaults:
- **Focal Length**: 27mm equivalent
- **Sensor Size**: 1/2.76" (approx. 6.3mm × 4.7mm)
- **Default Resolution**: 4080 × 3060 pixels

## 📁 **File Structure**

```
Perspective Projection/
├── measure_object.py          # Main script
├── requirements.txt           # Python dependencies
├── run.bat                  # Windows setup script
├── README.md                # This documentation
├── Input_images/            # Directory for input images
│   ├── image1.jpeg         # Processed with 215.9mm reference
│   ├── image2.jpeg         # Processed with 102.0mm reference
│   └── sample_object.jpg   # Test image
└── Results/                 # Output directory
    ├── image1_measurement_output.png
    ├── image2_measurement_output.png
    └── sample_object_measurement_output.png
```

## 🔧 **Technical Implementation**

### **Core Features**
- **Interactive Point Selection**: Mouse-based interface for precise point selection
- **Homography Computation**: Perspective transformation calculations
- **EXIF Processing**: Automatic camera parameter extraction
- **Multi-Unit Display**: Dimensions in mm, cm, and inches
- **Smart Image Detection**: Automatic filename-based reference length selection

### **Key Functions**
```python
# Point selection interface
def select_points_interactively(self, image, mode)

# Homography matrix computation
def compute_homography(self, object_corners, reference_length_mm)

# Dimension estimation with multi-unit output
def estimate_dimensions(self, object_corners, reference_length_mm)

# EXIF data extraction
def extract_exif_data(self, image_path)

# Measurement overlay with inch labels
def draw_measurement_overlay(self, image, object_corners, width_mm, height_mm, reference_length_mm)
```

## 🚨 **Troubleshooting**

### **Common Issues**

**"No images found in Input_images directory"**
- Ensure images are placed in the `Input_images/` folder
- Check file extensions (.jpg, .jpeg, .png)

**"Error: Could not select 4 corners"**
- Select all 4 corners of the object in order
- Ensure points are within image boundaries
- Click precisely on corner intersections

**"Error: Could not select reference points"**
- Select 2 distinct points for reference measurement
- Ensure reference points are clearly separated
- Reference should be on the same plane as object

**Poor Accuracy**
- Check camera angle (should be perpendicular to object)
- Ensure reference measurement is accurate
- Verify object and reference are on the same plane
- Use higher resolution images

### **Error Messages**
- **"Need exactly 4 corner points"**: Select all 4 corners of the object
- **"Need exactly 2 reference points"**: Select 2 points for reference measurement
- **"Could not compute dimensions"**: Check point selection and reference length
- **"Image file not found"**: Verify image path and file existence

## 💡 **Best Practices**

### **Image Capture**
1. **Use Good Lighting**: Avoid shadows and reflections
2. **Keep Camera Steady**: Use tripod or stable surface
3. **Fill Frame**: Make object occupy significant portion of image
4. **Avoid Distortion**: Keep object centered, avoid wide-angle effects

### **Measurement Accuracy**
1. **Precise Reference**: Use accurate reference measurements
2. **Clear Points**: Select easily identifiable corners
3. **Multiple Measurements**: Take several measurements and average results
4. **Validation**: Measure known objects to verify accuracy

### **For Best Results**
- Use the appropriate image naming convention (image1, image2) for automatic reference length detection
- Ensure reference objects are clearly visible and measurable
- Process images in good lighting conditions
- Use the interactive mode for optimal user experience

## 📦 **Dependencies**

### **Required Packages**
- **opencv-python**: Image processing and computer vision operations
- **numpy**: Numerical computations and matrix operations
- **pillow**: Image handling and EXIF data extraction
- **argparse**: Command-line argument parsing

### **Installation**
All dependencies are included in `requirements.txt` and installed automatically by `run.bat`.

## 🤝 **Contributing**

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 **License**

This project is open source and available under the MIT License.

## 🆘 **Support**

For questions, issues, or feature requests:

1. Check the troubleshooting section
2. Review the technical implementation
3. Verify your camera parameters and image quality
4. Create an issue with detailed information

## 🔗 **References**

- [OpenCV Documentation](https://docs.opencv.org/)
- [Perspective Projection](https://en.wikipedia.org/wiki/Perspective_(graphical))
- [Homography Transformations](https://en.wikipedia.org/wiki/Homography)
- [Camera Calibration](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html)

---


**Note**: This tool provides accurate measurements when used correctly. Always verify critical measurements with physical tools when precision is essential.
