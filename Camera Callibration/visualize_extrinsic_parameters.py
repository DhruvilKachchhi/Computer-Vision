import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

def load_calibration_data():
    """
    Load calibration data including intrinsic and extrinsic parameters
    """
    print("Loading calibration data...")
    
    # Load intrinsic parameters from npy_files directory
    camera_matrix = np.load("Results/npy_files/camera_matrix.npy")
    dist_coeffs = np.load("Results/npy_files/dist_coeffs.npy")
    
    # Load extrinsic parameters from npy_files directory
    rvecs = np.load("Results/npy_files/rotation_vectors.npy", allow_pickle=True)
    tvecs = np.load("Results/npy_files/translation_vectors.npy", allow_pickle=True)
    
    # Load object and image points from npy_files directory
    objpoints = np.load("Results/npy_files/object_points.npy", allow_pickle=True)
    imgpoints = np.load("Results/npy_files/image_points.npy", allow_pickle=True)
    
    print(f"Loaded {len(rvecs)} rotation vectors and {len(tvecs)} translation vectors")
    print(f"Camera matrix shape: {camera_matrix.shape}")
    print(f"Distortion coefficients shape: {dist_coeffs.shape}")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs, objpoints, imgpoints

def reconstruct_chessboard_planes(objpoints, rvecs, tvecs):
    """
    Reconstruct chessboard planes in 3D space using extrinsic parameters
    Each plane is oriented and positioned exactly according to its extrinsic parameters
    """
    print("Reconstructing chessboard planes with proper 3D orientation...")
    
    planes_data = []
    
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Chessboard corners in world coordinates (from objpoints)
        corners_3d = objpoints[i]
        
        # Transform corners to camera coordinate system
        # P_camera = R * P_world + t
        corners_3d_camera = (R @ corners_3d.T).T + tvec.flatten()
        
        # Get the four corner points of the chessboard rectangle
        # Chessboard is 9x6, so corners are at indices 0, 8, 45, 53
        corner_indices = [0, 8, 45, 53]
        rectangle_corners = corners_3d_camera[corner_indices]
        
        # Create a detailed grid of points for the plane surface
        # This ensures the plane appears with proper 3D orientation
        x_coords = corners_3d[:, 0]
        y_coords = corners_3d[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Create a fine grid for smooth surface rendering
        grid_resolution = 30
        grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, grid_resolution), 
                                   np.linspace(y_min, y_max, grid_resolution))
        grid_z = np.zeros_like(grid_x)
        
        # Stack coordinates for transformation
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
        
        # Transform grid to camera coordinates using the same transformation
        # This ensures the plane is oriented exactly as it was in the original photo
        grid_3d_camera = (R @ grid_points.T).T + tvec.flatten()
        
        # Reshape for plotting
        grid_x_camera = grid_3d_camera[:, 0].reshape(grid_x.shape)
        grid_y_camera = grid_3d_camera[:, 1].reshape(grid_y.shape)
        grid_z_camera = grid_3d_camera[:, 2].reshape(grid_z.shape)
        
        # Calculate normal vector of the plane for verification
        # This helps ensure proper 3D orientation
        v1 = rectangle_corners[1] - rectangle_corners[0]
        v2 = rectangle_corners[3] - rectangle_corners[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        
        planes_data.append({
            'index': i + 1,
            'corners': rectangle_corners,
            'grid_x': grid_x_camera,
            'grid_y': grid_y_camera,
            'grid_z': grid_z_camera,
            'rotation_matrix': R,
            'translation_vector': tvec.flatten(),
            'normal_vector': normal,
            'distance_from_camera': np.linalg.norm(tvec.flatten())
        })
    
    print(f"Reconstructed {len(planes_data)} chessboard planes with proper 3D orientation")
    return planes_data

def plot_camera_coordinate_frame(ax, scale=0.1):
    """
    Plot the camera coordinate frame at the origin
    """
    # Camera coordinate frame
    origin = np.array([0, 0, 0])
    x_axis = np.array([scale, 0, 0])
    y_axis = np.array([0, scale, 0])
    z_axis = np.array([0, 0, scale])
    
    # Plot axes
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], 
             color='r', label='X-axis (Right)', arrow_length_ratio=0.1)
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], 
             color='g', label='Y-axis (Down)', arrow_length_ratio=0.1)
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], 
             color='b', label='Z-axis (Forward)', arrow_length_ratio=0.1)
    
    # Add labels
    ax.text(x_axis[0]*1.2, x_axis[1]*1.2, x_axis[2]*1.2, 'X', color='r', fontsize=12)
    ax.text(y_axis[0]*1.2, y_axis[1]*1.2, y_axis[2]*1.2, 'Y', color='g', fontsize=12)
    ax.text(z_axis[0]*1.2, z_axis[1]*1.2, z_axis[2]*1.2, 'Z', color='b', fontsize=12)

def create_3d_visualization(planes_data):
    """
    Create 3D visualization of all chessboard planes and camera coordinate frame
    """
    print("Creating 3D visualization...")
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for different planes
    colors = plt.cm.tab20(np.linspace(0, 1, len(planes_data)))
    
    # Plot each chessboard plane
    for i, plane in enumerate(planes_data):
        # Plot the surface
        ax.plot_surface(plane['grid_x'], plane['grid_y'], plane['grid_z'], 
                       color=colors[i], alpha=0.6, linewidth=0, antialiased=True)
        
        # Plot the border
        corners = plane['corners']
        # Close the rectangle
        border_corners = np.vstack([corners, corners[0]])
        ax.plot(border_corners[:, 0], border_corners[:, 1], border_corners[:, 2], 
               color='black', linewidth=2, alpha=0.8)
        
        # Add label
        center = np.mean(corners, axis=0)
        ax.text(center[0], center[1], center[2] + 0.01, f'Image {plane["index"]}', 
               fontsize=8, ha='center', va='bottom')
    
    # Plot camera coordinate frame
    plot_camera_coordinate_frame(ax, scale=0.05)
    
    # Set labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('3D Visualization of Camera Calibration Extrinsic Parameters\n'
                'Chessboard Planes in Camera Coordinate System')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add legend
    ax.legend()
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    return fig, ax

def create_multiple_views(planes_data):
    """
    Create multiple views of the 3D visualization
    """
    print("Creating multiple view visualizations...")
    
    views = [
        {'elev': 90, 'azim': 0, 'name': 'Top View (X-Y)'},
        {'elev': 0, 'azim': 0, 'name': 'Front View (Y-Z)'},
        {'elev': 0, 'azim': 90, 'name': 'Side View (X-Z)'}
    ]
    
    figs = []
    
    for view in views:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot planes
        colors = plt.cm.tab20(np.linspace(0, 1, len(planes_data)))
        for i, plane in enumerate(planes_data):
            ax.plot_surface(plane['grid_x'], plane['grid_y'], plane['grid_z'], 
                           color=colors[i], alpha=0.6, linewidth=0, antialiased=True)
            corners = plane['corners']
            border_corners = np.vstack([corners, corners[0]])
            ax.plot(border_corners[:, 0], border_corners[:, 1], border_corners[:, 2], 
                   color='black', linewidth=2, alpha=0.8)
        
        # Plot camera coordinate frame
        plot_camera_coordinate_frame(ax, scale=0.05)
        
        # Set view
        ax.view_init(elev=view['elev'], azim=view['azim'])
        
        # Set labels
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title(f'Extrinsic Parameters Visualization - {view["name"]}')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        figs.append((fig, view['name']))
    
    return figs

def save_visualizations(fig, figs, planes_data):
    """
    Save all visualizations to the Results/3D_Plot directory
    """
    print("Saving visualizations...")
    
    # Create directory if it doesn't exist
    output_dir = 'Results/3D_Plot'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main visualization
    main_output_path = os.path.join(output_dir, 'extrinsic_visualization.png')
    fig.savefig(main_output_path, dpi=300, bbox_inches='tight')
    print(f"Main visualization saved to: {main_output_path}")
    
    # Save multiple views
    for fig_view, view_name in figs:
        filename = f"extrinsic_{view_name.replace(' ', '_').lower()}.png"
        output_path = os.path.join(output_dir, filename)
        fig_view.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"{view_name} saved to: {output_path}")
    
    # Save additional information
    info_output_path = os.path.join(output_dir, 'extrinsic_parameters_info.txt')
    with open(info_output_path, 'w') as f:
        f.write("Camera Calibration Extrinsic Parameters Information\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of calibration images: {len(planes_data)}\n")
        f.write(f"Chessboard size: 9x6 corners\n")
        f.write(f"Square size: 0.025 meters (2.5 cm)\n\n")
        
        for i, plane in enumerate(planes_data):
            f.write(f"Image {plane['index']}:\n")
            f.write(f"  Translation vector: {plane['translation_vector']}\n")
            f.write(f"  Distance from camera: {np.linalg.norm(plane['translation_vector']):.4f} meters\n")
            f.write("\n")
    
    print(f"Additional information saved to: {info_output_path}")

def main():
    """
    Main function to run the extrinsic parameter visualization
    """
    print("Starting camera calibration extrinsic parameter visualization...")
    
    try:
        # Load calibration data
        camera_matrix, dist_coeffs, rvecs, tvecs, objpoints, imgpoints = load_calibration_data()
        
        # Reconstruct chessboard planes
        planes_data = reconstruct_chessboard_planes(objpoints, rvecs, tvecs)
        
        # Create main 3D visualization
        fig, ax = create_3d_visualization(planes_data)
        
        # Create multiple views
        figs = create_multiple_views(planes_data)
        
        # Save all visualizations
        save_visualizations(fig, figs, planes_data)
        
        # Show the main plot
        plt.show()
        
        print("Visualization complete!")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find calibration data files. Please run save_calibration_data.py first.")
        print(f"Missing file: {e}")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()