import cv2
import numpy as np
import glob
import os

def run_camera_calibration():
    """
    Run camera calibration and save all necessary data including extrinsic parameters
    """
    print("Starting camera calibration process...")
    
    # Termination criteria for corner subpixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Chessboard dimensions (change if needed)
    chessboard_size = (9, 6)  # internal corners

    # Real world square size in meters (measure your printed chessboard)
    square_size = 0.025  # 2.5 cm per square

    # Prepare object points like (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints = []
    imgpoints = []

    images = glob.glob('Callibration Images/*.jpeg')

    print(f"Found {len(images)} calibration images")
    print("Detecting chessboard corners...")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Corners', img)
            cv2.waitKey(200)
        else:
            print(f"Warning: Could not find chessboard corners in {fname}")

    cv2.destroyAllWindows()

    # Camera Calibration
    if len(objpoints) > 0 and len(imgpoints) > 0:
        # Get image dimensions from the first image
        first_img = cv2.imread(images[0])
        gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        print("\nCamera Matrix:")
        print(camera_matrix)

        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        
        print(f"\nNumber of valid images: {len(objpoints)}")
        print(f"Rotation vectors: {len(rvecs)}")
        print(f"Translation vectors: {len(tvecs)}")
        
        # Create organized directory structure
        os.makedirs('Results', exist_ok=True)
        os.makedirs('Results/npy_files', exist_ok=True)
        os.makedirs('Results/Undistorted_Images', exist_ok=True)
        
        # Save intrinsic parameters
        np.save("Results/npy_files/camera_matrix.npy", camera_matrix)
        np.save("Results/npy_files/dist_coeffs.npy", dist_coeffs)
        
        # Save extrinsic parameters
        np.save("Results/npy_files/rotation_vectors.npy", rvecs)
        np.save("Results/npy_files/translation_vectors.npy", tvecs)
        
        # Save object points (world coordinates of chessboard corners)
        np.save("Results/npy_files/object_points.npy", objpoints)
        
        # Save image points (detected corner coordinates in images)
        np.save("Results/npy_files/image_points.npy", imgpoints)
        
        # Also save to root directory for backward compatibility
        np.save("camera_matrix.npy", camera_matrix)
        np.save("dist_coeffs.npy", dist_coeffs)
        
        print("\nAll calibration data saved to Results folder.")
        print("Intrinsic parameters also saved to root directory.")
        
        return camera_matrix, dist_coeffs, rvecs, tvecs, objpoints, imgpoints
    else:
        print("No valid chessboard corners found in any images!")
        return None, None, None, None, None, None

def undistort_calibration_images(camera_matrix, dist_coeffs):
    """
    Undistort and save calibration images to Results folder
    """
    print("\nUndistorting calibration images and saving to Results folder...")
    
    # Chessboard dimensions (same as in calibration)
    chessboard_size = (9, 6)  # internal corners
    
    # Termination criteria for corner subpixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    images = glob.glob('Callibration Images/*.jpeg')
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find corners for this image
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            # Refine corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw corners on the original image
            img_with_corners = img.copy()
            cv2.drawChessboardCorners(img_with_corners, chessboard_size, corners2, ret)
            
            h, w = img_with_corners.shape[:2]
            
            # Undistort the image with corners
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
            undistorted_img = cv2.undistort(img_with_corners, camera_matrix, dist_coeffs, None, new_camera_matrix)
            
            # Crop the image if needed
            x, y, w, h = roi
            undistorted_img = undistorted_img[y:y+h, x:x+w]
            
            # Generate output filename
            filename = os.path.basename(fname)
            output_path = os.path.join("Results/Undistorted_Images", f"undistorted_{filename}")
            
            # Save the undistorted image with corners
            cv2.imwrite(output_path, undistorted_img)
            print(f"Saved: {output_path}")
        else:
            print(f"Warning: Could not find chessboard corners in {fname}, saving undistorted image without corners")
            # Fallback: save undistorted image without corners
            h, w = img.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
            undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
            x, y, w, h = roi
            undistorted_img = undistorted_img[y:y+h, x:x+w]
            
            filename = os.path.basename(fname)
            output_path = os.path.join("Results/Undistorted_Images", f"undistorted_{filename}")
            cv2.imwrite(output_path, undistorted_img)
            print(f"Saved: {output_path}")

def main():
    """
    Main function to run the complete camera calibration process
    """
    # Run camera calibration
    camera_matrix, dist_coeffs, rvecs, tvecs, objpoints, imgpoints = run_camera_calibration()
    
    if camera_matrix is not None:
        # Undistort images
        undistort_calibration_images(camera_matrix, dist_coeffs)
        
        print("\nCamera calibration and image processing complete!")
        print("All data saved to Results/ directory:")
        print("  - camera_matrix.npy (intrinsic parameters)")
        print("  - dist_coeffs.npy (distortion coefficients)")
        print("  - rotation_vectors.npy (extrinsic parameters)")
        print("  - translation_vectors.npy (extrinsic parameters)")
        print("  - object_points.npy (world coordinates)")
        print("  - image_points.npy (detected coordinates)")
        print("  - undistorted_*.jpeg (processed images)")
    else:
        print("Camera calibration failed!")

if __name__ == "__main__":
    main()