#!/usr/bin/env python3
"""
Real-World Object Dimension Estimation using Perspective Projection

This script estimates real-world 2D dimensions of objects from single images using
perspective projection and homography transformations. It implements a complete
solution for measuring objects with known reference dimensions.

Author: AI Assistant
Date: 2025
"""

import cv2
import numpy as np
import argparse
import sys
import os
from typing import Tuple, List, Optional
from PIL import Image
from PIL.ExifTags import TAGS


class ObjectDimensionEstimator:
    """Class for estimating real-world object dimensions using perspective projection."""
    
    def __init__(self):
        """Initialize the dimension estimator with Samsung M34 camera parameters."""
        # Samsung M34 Camera Parameters (approximate)
        self.focal_length_mm = 27.0  # 26mm equivalent focal length
        self.sensor_width_mm = 6.3  # Approximate for 1/2.76" sensor
        self.sensor_height_mm = 4.7  # Approximate aspect ratio
        self.image_width_px = 4080  # Default resolution
        self.image_height_px = 3060
        
        # Reference length for calibration (editable variable)
        # Change this value to match your reference object's real-world length in millimeters
        self.reference_length_mm = 102  # References: 102mm/ 215.9mm 
        
        # Mouse callback variables
        self.points = []
        self.reference_points = []
        self.drawing_mode = "corners"  # "corners" or "reference"
        
        # Calibration data
        self.homography_matrix = None
        self.scale_factor = 1.0
        
    def extract_exif_data(self, image_path: str) -> dict:
        """
        Extract EXIF metadata from image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing EXIF data
        """
        exif_data = {}
        try:
            image = Image.open(image_path)
            info = image._getexif()
            
            if info is not None:
                for tag, value in info.items():
                    decoded = TAGS.get(tag, tag)
                    exif_data[decoded] = value
                    
                # Extract specific camera parameters if available
                if 'FocalLength' in exif_data:
                    focal_length = exif_data['FocalLength']
                    if isinstance(focal_length, tuple):
                        self.focal_length_mm = focal_length[0] / focal_length[1]
                    else:
                        self.focal_length_mm = float(focal_length)
                        
                if 'ImageWidth' in exif_data and 'ImageHeight' in exif_data:
                    self.image_width_px = exif_data['ImageWidth']
                    self.image_height_px = exif_data['ImageHeight']
                    
        except Exception as e:
            print(f"Warning: Could not extract EXIF data: {e}")
            
        return exif_data
    
    def calculate_pixel_size(self) -> Tuple[float, float]:
        """
        Calculate pixel size on sensor in both dimensions.
        
        Returns:
            Tuple of (pixel_width_mm, pixel_height_mm)
        """
        pixel_width_mm = self.sensor_width_mm / self.image_width_px
        pixel_height_mm = self.sensor_height_mm / self.image_height_px
        return pixel_width_mm, pixel_height_mm
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback function for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing_mode == "corners":
                if len(self.points) < 4:
                    self.points.append((x, y))
                    print(f"Selected corner {len(self.points)}: ({x}, {y})")
                else:
                    print("All 4 corners selected. Switch to reference mode.")
                    
            elif self.drawing_mode == "reference":
                if len(self.reference_points) < 2:
                    self.reference_points.append((x, y))
                    print(f"Selected reference point {len(self.reference_points)}: ({x}, {y})")
                else:
                    print("Reference points selected.")
    
    def select_points_interactively(self, image: np.ndarray, mode: str = "corners") -> List[Tuple[int, int]]:
        """
        Interactive point selection interface.
        
        Args:
            image: Input image
            mode: Selection mode ("corners" or "reference")
            
        Returns:
            List of selected points
        """
        self.drawing_mode = mode
        self.points = [] if mode == "corners" else self.points
        self.reference_points = [] if mode == "reference" else self.reference_points
        
        window_name = "Point Selection - " + ("Corners" if mode == "corners" else "Reference")
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        instructions = {
            "corners": "Click 4 corners of the object (top-left, top-right, bottom-right, bottom-left)",
            "reference": "Click 2 points representing known reference length"
        }
        
        print(f"\n{instructions[mode]}")
        print("Controls:")
        print("  - Left Click: Select points")
        print("  - 'c': Clear points")
        print("  - 'q': Quit")
        print("  - 'm': Confirm selection")
        
        while True:
            temp_image = image.copy()
            
            # Draw selected points
            current_points = self.points if mode == "corners" else self.reference_points
            
            for i, point in enumerate(current_points):
                cv2.circle(temp_image, point, 5, (0, 255, 0), -1)
                cv2.putText(temp_image, f"{i+1}", (point[0]+5, point[1]+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw lines between points if multiple selected
            if len(current_points) > 1:
                for i in range(len(current_points) - 1):
                    cv2.line(temp_image, current_points[i], current_points[i+1], (255, 0, 0), 2)
            
            cv2.imshow(window_name, temp_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return []
            elif key == ord('c'):
                current_points.clear()
            elif key == ord('m'):
                if len(current_points) >= 2:
                    cv2.destroyAllWindows()
                    return current_points.copy()
                else:
                    print("Please select at least 2 points")
    
    def compute_homography(self, object_corners: List[Tuple[int, int]], 
                          reference_length_mm: float) -> Optional[np.ndarray]:
        """
        Compute homography matrix for perspective transformation.
        
        Args:
            object_corners: 4 corner points of the object
            reference_length_mm: Known reference length in millimeters
            
        Returns:
            Homography matrix or None if computation fails
        """
        if len(object_corners) != 4:
            print("Error: Need exactly 4 corner points")
            return None
            
        # Define target rectangle dimensions based on reference
        # We'll use the reference to establish scale
        ref_points = np.array(self.reference_points, dtype=np.float32)
        
        if len(ref_points) != 2:
            print("Error: Need exactly 2 reference points")
            return None
            
        # Calculate reference length in pixels
        ref_pixel_length = np.linalg.norm(ref_points[0] - ref_points[1])
        
        # Calculate scale factor (mm per pixel)
        self.scale_factor = reference_length_mm / ref_pixel_length
        
        # Define target coordinates for homography
        # We'll create a rectangle with the correct aspect ratio
        src_points = np.array(object_corners, dtype=np.float32)
        
        # Calculate object dimensions in pixels
        width_px = np.linalg.norm(np.array(src_points[1]) - np.array(src_points[0]))
        height_px = np.linalg.norm(np.array(src_points[3]) - np.array(src_points[0]))
        
        # Create target rectangle with correct aspect ratio
        target_width = width_px * self.scale_factor
        target_height = height_px * self.scale_factor
        
        dst_points = np.array([
            [0, 0],
            [target_width, 0],
            [target_width, target_height],
            [0, target_height]
        ], dtype=np.float32)
        
        # Compute homography
        try:
            self.homography_matrix, mask = cv2.findHomography(src_points, dst_points)
            return self.homography_matrix
        except Exception as e:
            print(f"Error computing homography: {e}")
            return None
    
    def estimate_dimensions(self, object_corners: List[Tuple[int, int]], 
                           reference_length_mm: float) -> Tuple[float, float]:
        """
        Estimate real-world dimensions of the object.
        
        Args:
            object_corners: 4 corner points of the object
            reference_length_mm: Known reference length in millimeters
            
        Returns:
            Tuple of (width_mm, height_mm)
        """
        homography = self.compute_homography(object_corners, reference_length_mm)
        
        if homography is None:
            return 0.0, 0.0
        
        # Transform corner points to get real-world coordinates
        src_points = np.array(object_corners, dtype=np.float32).reshape(-1, 1, 2)
        dst_points = cv2.perspectiveTransform(src_points, homography)
        
        # Extract transformed coordinates
        dst_points = dst_points.reshape(-1, 2)
        
        # Calculate width and height
        width_mm = np.linalg.norm(dst_points[1] - dst_points[0])
        height_mm = np.linalg.norm(dst_points[3] - dst_points[0])
        
        return width_mm, height_mm
    
    def draw_measurement_overlay(self, image: np.ndarray, object_corners: List[Tuple[int, int]], 
                                width_mm: float, height_mm: float, reference_length_mm: float) -> np.ndarray:
        """
        Draw measurement overlay on the image.
        
        Args:
            image: Original image
            object_corners: Object corner points
            width_mm: Estimated width
            height_mm: Estimated height
            reference_length_mm: Reference length used
            
        Returns:
            Image with measurement overlay
        """
        overlay_image = image.copy()
        
        # Convert dimensions to inches (1 inch = 25.4 mm)
        width_in = width_mm / 25.4
        height_in = height_mm / 25.4
        reference_in = reference_length_mm / 25.4
        
        # Draw object rectangle
        points_array = np.array(object_corners, np.int32)
        points_array = points_array.reshape((-1, 1, 2))
        cv2.polylines(overlay_image, [points_array], True, (0, 255, 0), 2)
        
        # Draw reference line
        if len(self.reference_points) == 2:
            cv2.line(overlay_image, self.reference_points[0], self.reference_points[1], (255, 0, 0), 2)
            mid_point = ((self.reference_points[0][0] + self.reference_points[1][0]) // 2,
                        (self.reference_points[0][1] + self.reference_points[1][1]) // 2)
            cv2.putText(overlay_image, f"Ref: {reference_in:.2f}in", mid_point,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add dimension labels in inches
        # Width label (top side)
        top_mid = ((object_corners[0][0] + object_corners[1][0]) // 2,
                  (object_corners[0][1] + object_corners[1][1]) // 2)
        cv2.putText(overlay_image, f"W: {width_in:.2f}in", (top_mid[0], top_mid[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Height label (left side)
        left_mid = ((object_corners[0][0] + object_corners[3][0]) // 2,
                   (object_corners[0][1] + object_corners[3][1]) // 2)
        cv2.putText(overlay_image, f"H: {height_in:.2f}in", (left_mid[0] + 10, left_mid[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add camera info
        camera_info = f"Samsung M34 | FL: {self.focal_length_mm}mm | Res: {self.image_width_px}x{self.image_height_px}"
        cv2.putText(overlay_image, camera_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay_image
    
    def validate_points(self, points: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> bool:
        """Validate that points are within image bounds and form a reasonable shape."""
        height, width = image_shape[:2]
        
        # Check bounds
        for x, y in points:
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
        
        # Check for reasonable spacing (not all points too close)
        if len(points) >= 3:
            distances = []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                    distances.append(dist)
            
            min_dist = min(distances)
            if min_dist < 10:  # Points too close together
                return False
        
        return True


def create_sample_image():
    """Create a sample image for testing."""
    # Create a checkerboard-like image with a rectangle
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw a rectangle (simulating an object)
    cv2.rectangle(image, (200, 150), (600, 450), (0, 0, 0), 2)
    
    # Add grid for perspective effect
    for i in range(0, 800, 50):
        cv2.line(image, (i, 0), (i, 600), (200, 200, 200), 1)
    for i in range(0, 600, 50):
        cv2.line(image, (0, i), (800, i), (200, 200, 200), 1)
    
    # Add text
    cv2.putText(image, "Sample Object for Dimension Estimation", (100, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(image, "Click corners of black rectangle", (100, 550), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save sample image
    sample_path = "Input_images/sample_object.jpg"
    cv2.imwrite(sample_path, image)
    return sample_path


def main():
    """Main function to run the dimension estimation tool."""
    print("Real-World Object Dimension Estimation Tool")
    print("=" * 50)
    print("Using Perspective Projection and Homography")
    print()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Estimate real-world object dimensions')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--reference', type=float, help='Known reference length (mm)')
    parser.add_argument('--create-sample', action='store_true', help='Create sample image')
    
    args = parser.parse_args()
    
    # Handle sample creation
    if args.create_sample:
        sample_path = create_sample_image()
        print(f"Sample image created: {sample_path}")
        print("Run: python measure_object.py --image Input_images/sample_object.jpg --reference 200")
        return
    
    # Determine image path
    if args.image:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)
    else:
        # Look for images in Input_images directory
        input_dir = "Input_images"
        if os.path.exists(input_dir):
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if image_files:
                if len(image_files) == 1:
                    # Only one image available, use it directly
                    image_path = os.path.join(input_dir, image_files[0])
                    print(f"Using image: {image_path}")
                else:
                    # Multiple images available, let user choose
                    print(f"Found {len(image_files)} images in Input_images directory:")
                    for i, filename in enumerate(image_files, 1):
                        print(f"  {i}. {filename}")
                    
                    while True:
                        try:
                            choice = input(f"\nSelect an image (1-{len(image_files)}) or press Enter for first image: ").strip()
                            if choice == "":
                                choice = 1
                            else:
                                choice = int(choice)
                            
                            if 1 <= choice <= len(image_files):
                                image_path = os.path.join(input_dir, image_files[choice - 1])
                                print(f"Selected image: {image_path}")
                                break
                            else:
                                print(f"Please enter a number between 1 and {len(image_files)}")
                        except ValueError:
                            print("Please enter a valid number")
            else:
                print("No images found in Input_images directory.")
                print("Please place an image file there or use --image parameter.")
                sys.exit(1)
        else:
            print("Input_images directory not found.")
            print("Please create the directory and place an image file there.")
            sys.exit(1)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    # Extract EXIF data if available
    estimator = ObjectDimensionEstimator()
    exif_data = estimator.extract_exif_data(image_path)
    
    if exif_data:
        print("EXIF data extracted successfully")
        print(f"Camera parameters: FL={estimator.focal_length_mm}mm, Res={estimator.image_width_px}x{estimator.image_height_px}")
    else:
        print("Using default Samsung M34 parameters")
        print(f"Focal Length: {estimator.focal_length_mm}mm")
        print(f"Resolution: {estimator.image_width_px}x{estimator.image_height_px}")
    
    # Get reference length based on the image being processed
    image_filename = os.path.basename(image_path).lower()
    
    if 'image1' in image_filename:
        reference_length_mm = 215.9  # Reference length for image1
        print(f"Using reference length for image1: {reference_length_mm} mm")
    elif 'image2' in image_filename:
        reference_length_mm = 102.0  # Reference length for image2
        print(f"Using reference length for image2: {reference_length_mm} mm")
    else:
        # Use the class variable for other images
        reference_length_mm = estimator.reference_length_mm
        print(f"Using default reference length: {reference_length_mm} mm")
    
    # Select object corners
    print("\nStep 1: Select 4 corners of the object")
    print("Click in order: top-left, top-right, bottom-right, bottom-left")
    
    object_corners = estimator.select_points_interactively(image, "corners")
    
    if len(object_corners) != 4:
        print("Error: Could not select 4 corners")
        sys.exit(1)
    
    # Validate corner points
    if not estimator.validate_points(object_corners, image.shape):
        print("Warning: Selected points may be invalid. Proceeding anyway...")
    
    # Select reference points
    print("\nStep 2: Select 2 points representing the reference length")
    reference_points = estimator.select_points_interactively(image, "reference")
    
    if len(reference_points) != 2:
        print("Error: Could not select reference points")
        sys.exit(1)
    
    # Estimate dimensions
    print("\nComputing dimensions...")
    width_mm, height_mm = estimator.estimate_dimensions(object_corners, reference_length_mm)
    
    if width_mm == 0 or height_mm == 0:
        print("Error: Could not compute dimensions")
        sys.exit(1)
    
    # Convert to inches (1 inch = 25.4 mm)
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4
    reference_in = reference_length_mm / 25.4
    
    # Display results
    print("\n" + "=" * 60)
    print("REAL-WORLD OBJECT DIMENSIONS")
    print("=" * 60)
    print(f"Estimated Width:  {width_mm:.2f} mm ({width_mm/10:.2f} cm) ({width_in:.2f} in)")
    print(f"Estimated Height: {height_mm:.2f} mm ({height_mm/10:.2f} cm) ({height_in:.2f} in)")
    print(f"Reference Used:   {reference_length_mm} mm ({reference_in:.2f} in)")
    print(f"Scale Factor:     {estimator.scale_factor:.6f} mm/px")
    print("=" * 60)
    
    # Create output directory
    output_dir = "Results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique output filename based on input image name
    image_filename = os.path.basename(image_path)
    image_name = os.path.splitext(image_filename)[0]
    
    # Draw measurement overlay
    result_image = estimator.draw_measurement_overlay(image, object_corners, width_mm, height_mm, reference_length_mm)
    
    # Save result with unique filename
    output_filename = f"{image_name}_measurement_output.png"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, result_image)
    print(f"\nResult saved to: {output_path}")
    
    # Display result
    cv2.imshow("Measurement Result", result_image)
    print("\nPress any key to close the result window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()