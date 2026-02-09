# Image Blurring and Convolution Theorem Demonstration

This Python project demonstrates the **Convolution Theorem** by implementing image blurring in both spatial and frequency domains, then mathematically proving they produce equivalent results.

## Theory Overview

The **Convolution Theorem** states that convolution in the spatial domain is equivalent to multiplication in the frequency domain:

```
f(x,y) * h(x,y)  ↔  F(u,v) · H(u,v)
```

Where:
- `*` represents convolution in spatial domain
- `·` represents element-wise multiplication in frequency domain
- `F(u,v)` and `H(u,v)` are the Fourier Transforms of `f(x,y)` and `h(x,y)`

## Project Structure

```
Image Blurring/
├── Input Images/          # Contains input images to process
│   ├── image (1).jpg
│   ├── image (2).jpg
│   └── ...
├── Results/              # Output directory for processed images
│   ├── [filename]_spatial_blur.png
│   ├── [filename]_fourier_blur.png
│   └── [filename]_difference_map.png
├── src/
│   └── main.py          # Main implementation script
├── requirements.txt     # Python dependencies
├── run_project.bat      # Windows batch file for setup and execution
└── README.md           # This documentation
```

## Features

### Spatial Domain Filtering
- Creates a 2D Gaussian kernel based on user-defined sigma
- Applies convolution using OpenCV's `filter2D` function
- Demonstrates traditional image smoothing techniques

### Frequency Domain Filtering
- Computes Discrete Fourier Transform (DFT) of the image
- Pads and transforms the Gaussian kernel to match image size
- Performs element-wise multiplication in frequency domain
- Applies Inverse DFT to recover the blurred image

### Validation & Verification
- Calculates Mean Squared Error (MSE) between spatial and frequency domain results
- Generates visual difference maps showing pixel-by-pixel differences
- Mathematically proves the Convolution Theorem equivalence

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Windows operating system (for batch file support)

### Quick Start

1. **Clone or download** this project to your local machine

2. **Run the batch file** (Windows):
   ```cmd
   run_project.bat
   ```

   This will:
   - Create a virtual environment (`venv/`)
   - Install required dependencies
   - Execute the main script

3. **Alternative manual setup**:
   ```cmd
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the script
   python src/main.py
   ```

## Usage

### Basic Usage
```cmd
python src/main.py
```

### Command Line Options
```cmd
python src/main.py --sigma 2.0 --kernel-size 15
```

**Available Options:**
- `--sigma`: Standard deviation for Gaussian kernel (default: 2.0)
- `--kernel-size`: Size of Gaussian kernel, must be odd (default: 15)

### Examples

```cmd
# Use a smaller kernel for subtle blurring
python src/main.py --sigma 1.0 --kernel-size 9

# Use a larger kernel for strong blurring
python src/main.py --sigma 3.0 --kernel-size 21

# Process with custom parameters
python src/main.py --sigma 2.5 --kernel-size 17
```

## Output Files

For each input image, the script generates three output files in the `Results/` directory:

1. **`[filename]_spatial_blur.png`**: Result from spatial domain convolution
2. **`[filename]_fourier_blur.png`**: Result from frequency domain filtering
3. **`[filename]_difference_map.png`**: Visual representation of differences between the two methods

## Understanding the Results

### Console Output
The script provides detailed console output including:
- Image processing progress
- MSE values between spatial and frequency domain results
- Verification status of the Convolution Theorem

### MSE Interpretation
- **MSE < 1.0**: Results are nearly identical (Convolution Theorem verified)
- **MSE ≥ 1.0**: Significant differences detected (Check implementation)

### Difference Maps
- **Black areas**: Perfect match between methods
- **Bright areas**: Differences between spatial and frequency domain results
- **Expected**: Very dark difference maps with MSE < 1.0

## Dependencies

The project uses standard scientific computing libraries:

- **numpy**: Numerical computations and array operations
- **opencv-python**: Image processing and computer vision functions
- **matplotlib**: (Optional) For additional visualization capabilities

Install with:
```cmd
pip install numpy opencv-python matplotlib
```

## Technical Implementation

### Gaussian Kernel Generation
The script creates a 2D Gaussian kernel using the formula:
```
G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))
```

Where σ (sigma) controls the blur intensity.

### Frequency Domain Processing
1. **Padding**: Kernel is padded to match image dimensions
2. **DFT**: Both image and kernel are transformed to frequency domain
3. **Multiplication**: Element-wise multiplication of frequency representations
4. **Inverse DFT**: Transform back to spatial domain

### Validation Method
- **MSE Calculation**: `MSE = mean((Image1 - Image2)²)`
- **Threshold**: Results with MSE < 1.0 are considered equivalent
- **Visual Verification**: Difference maps provide pixel-level comparison

## Troubleshooting

### Common Issues

1. **Python not found**:
   - Ensure Python is installed and added to PATH
   - Check with: `python --version`

2. **Virtual environment creation fails**:
   - Try running Command Prompt as Administrator
   - Check Python installation integrity

3. **Import errors**:
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

4. **No images found**:
   - Ensure `Input Images/` directory exists
   - Check that images are in supported formats (JPG, PNG, BMP, TIFF)

### Image Format Support
The script supports common image formats:
- JPEG/JPG
- PNG
- BMP
- TIFF/TIF

## Educational Value

This project demonstrates several important concepts:

1. **Convolution Theorem**: Fundamental principle in signal processing
2. **Fourier Analysis**: Frequency domain representation of signals
3. **Image Processing**: Practical application of mathematical concepts
4. **Numerical Methods**: Handling floating-point precision and numerical stability

## License

This project is provided as an educational demonstration. Feel free to use, modify, and distribute for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or report issues.

## Contact

For questions or feedback about this project, please open an issue or contact the maintainer.