import numpy as np
import pyvista as pv
import pickle
import random
import sqlite3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to load normalization parameters from the database
def load_normalization_params(db_path="point_clouds_pressure_stratified.db"):
    """
    Load the normalization parameters from the original database
    to enable denormalization of point clouds for visualization.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM point_clouds LIMIT 1")
            sample_data = cursor.fetchone()[0]
            
            # Get a sample to determine the original data structure
            sample_points = np.frombuffer(sample_data, dtype=np.float32).reshape(8192, 7)
            
            # Calculate global min/max for the database
            cursor.execute("SELECT data FROM point_clouds")
            
            xyz_min = np.array([float('inf')] * 3)
            xyz_max = np.array([float('-inf')] * 3)
            pressure_min = float('inf')
            pressure_max = float('-inf')
            
            # Use a limited number of samples for efficiency
            for i, row in enumerate(cursor.fetchall()):
                if i > 100:  # Limit to 100 samples for speed
                    break
                    
                points = np.frombuffer(row[0], dtype=np.float32).reshape(8192, 7)
                xyz_min = np.minimum(xyz_min, points[:, :3].min(axis=0))
                xyz_max = np.maximum(xyz_max, points[:, :3].max(axis=0))
                pressure_min = min(pressure_min, points[:, 3].min())
                pressure_max = max(pressure_max, points[:, 3].max())
            
            xyz_range = xyz_max - xyz_min
            pressure_range = pressure_max - pressure_min
            
            print("Normalization parameters:")
            print(f"XYZ min: {xyz_min}")
            print(f"XYZ max: {xyz_max}")
            print(f"XYZ range: {xyz_range}")
            print(f"Pressure min: {pressure_min}")
            print(f"Pressure max: {pressure_max}")
            
            return {
                'xyz_min': xyz_min,
                'xyz_range': xyz_range,
                'pressure_min': pressure_min,
                'pressure_range': pressure_range
            }
    except Exception as e:
        print(f"Could not load normalization parameters: {e}")
        # Provide default normalization parameters
        return {
            'xyz_min': np.array([-1.0, -0.5, -0.5]),
            'xyz_range': np.array([2.0, 1.0, 1.0]),
            'pressure_min': -1.0,
            'pressure_range': 2.0
        }

# Function to denormalize point cloud data
def denormalize_points(points, pressure, norm_params):
    """
    Denormalize the normalized point cloud data back to original scale.
    
    Args:
        points: Normalized XYZ coordinates (N, 3)
        pressure: Normalized pressure values (N, 1)
        norm_params: Dictionary with normalization parameters
        
    Returns:
        Denormalized points and pressure values
    """
    if norm_params is None:
        return points, pressure
    
    # Make a copy to avoid modifying the original data
    denorm_xyz = points.copy()
    denorm_pressure = pressure.copy()
    
    # Denormalize XYZ coordinates - apply to each dimension separately
    for i in range(3):
        denorm_xyz[:, i] = points[:, i] * norm_params['xyz_range'][i] + norm_params['xyz_min'][i]
    
    # Denormalize pressure
    denorm_pressure = pressure * norm_params['pressure_range'] + norm_params['pressure_min']
    
    return denorm_xyz, denorm_pressure

# Function to load data
def load_test_data():
    # Load from pickle file
    try:
        with open('point_transformer_predictions_4.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

def main(denormalize=True):
    # Load the data
    print("Loading test data...")
    test_data = load_test_data()
    
    if test_data is None:
        print("Failed to load data. Please make sure your pickle file is in the same directory.")
        return
    
    # Get the dimensions of the data
    num_batches = len(test_data['inputs'])
    samples_per_batch = len(test_data['inputs'][0]) if num_batches > 0 else 0
    
    print(f"Data loaded successfully! Found {num_batches} batches with {samples_per_batch} samples each.")
    
    # Load normalization parameters if denormalization is requested
    norm_params = None
    if denormalize:
        norm_params = load_normalization_params()
        if norm_params:
            print("Normalization parameters loaded successfully for denormalization.")
        else:
            print("Failed to load normalization parameters. Using normalized data.")
    else:
        print("Using normalized data for visualization (denormalization disabled).")
    
    # Select a random sample
    batch_idx = 1
    sample_idx = 0
    print(f"Selected random sample: Batch {batch_idx}, Sample {sample_idx}")
    
    # Get the selected data
    points = test_data['inputs'][batch_idx][sample_idx]  # Shape: (4096, 3)
    gt_pressure = test_data['ground_truth'][batch_idx][sample_idx]  # Shape: (4096, 1)
    pred_pressure = test_data['predictions'][batch_idx][sample_idx]  # Shape: (4096, 1)
    
    # Print some statistics to verify the data
    print(f"Ground Truth - min: {gt_pressure.min():.6f}, max: {gt_pressure.max():.6f}, mean: {gt_pressure.mean():.6f}")
    print(f"Prediction - min: {pred_pressure.min():.6f}, max: {pred_pressure.max():.6f}, mean: {pred_pressure.mean():.6f}")
    
    # Denormalize the data if requested
    if denormalize and norm_params:
        # Only denormalize the XYZ coordinates, keep pressure normalized
        points_viz = points.copy()
        for i in range(3):
            points_viz[:, i] = points[:, i] * norm_params['xyz_range'][i] + norm_params['xyz_min'][i]
        
        # Keep pressure values normalized
        gt_pressure_viz = gt_pressure.copy()
        pred_pressure_viz = pred_pressure.copy()
        
        print(f"Denormalized XYZ coordinates only. Pressure values remain normalized.")
        print(f"GT Pressure - min: {gt_pressure_viz.min():.6f}, max: {gt_pressure_viz.max():.6f}")
        print(f"Pred Pressure - min: {pred_pressure_viz.min():.6f}, max: {pred_pressure_viz.max():.6f}")
    else:
        points_viz, gt_pressure_viz = points.copy(), gt_pressure.copy()
        pred_pressure_viz = pred_pressure.copy()
        print("Using normalized values for visualization")
    
    # Calculate pressure difference
    pressure_diff = gt_pressure_viz - pred_pressure_viz
    
    # assert that the gt_pressure_viz and pred_pressure_viz are not the same
    assert not np.allclose(gt_pressure_viz, pred_pressure_viz)
    
    # Calculate error metrics
    mse = np.mean((gt_pressure - pred_pressure)**2)
    mae = np.mean(np.abs(gt_pressure - pred_pressure))
    
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    
    # Create PyVista visualization
    # Create point clouds for each visualization
    gt_cloud = pv.PolyData(points_viz)
    gt_cloud.point_data["pressure"] = gt_pressure_viz.flatten()
    
    pred_cloud = pv.PolyData(points_viz)
    pred_cloud.point_data["pressure"] = pred_pressure_viz.flatten()
    
    diff_cloud = pv.PolyData(points_viz)
    diff_cloud.point_data["pressure_diff"] = pressure_diff.flatten()
    
    # Create a plotter with three subplots
    plotter = pv.Plotter(shape=(1, 3))
    
    # Add ground truth to first subplot
    plotter.subplot(0, 0)
    plotter.add_title("Ground Truth Pressure")
    plotter.add_mesh(gt_cloud, scalars="pressure", point_size=20, render_points_as_spheres=True, 
                    cmap="jet", show_scalar_bar=True, scalar_bar_args={"title": "Pressure"})
    
    # Add prediction to second subplot
    plotter.subplot(0, 1)
    plotter.add_title("Predicted Pressure")
    plotter.add_mesh(pred_cloud, scalars="pressure", point_size=20, render_points_as_spheres=True, 
                    cmap="jet", show_scalar_bar=True, scalar_bar_args={"title": "Pressure"})
    
    # Add difference to third subplot
    plotter.subplot(0, 2)
    plotter.add_title("Pressure Difference (GT - Pred)")
    # Set symmetric colormap for difference
    diff_abs_max = max(abs(pressure_diff.min()), abs(pressure_diff.max()))
    plotter.add_mesh(diff_cloud, scalars="pressure_diff", point_size=20, render_points_as_spheres=True, 
                    cmap="jet", show_scalar_bar=True, scalar_bar_args={"title": "Difference"},
                    clim=[-diff_abs_max, diff_abs_max])
    
    # Add text with error metrics
    plotter.add_text(f"MSE: {mse:.6f}\nMAE: {mae:.6f}", position="upper_left", font_size=10)
    
    # Set camera position for all subplots
    for i in range(3):
        plotter.subplot(0, i)
        plotter.camera_position = 'iso'
        plotter.camera.zoom(1.5)
    
    # Show the plot
    print("Displaying interactive visualization. Close the window to exit.")
    plotter.link_views()  # Link camera movement between subplots
    plotter.show(window_size=[1800, 600], title=f"Pressure Visualization - Batch {batch_idx}, Sample {sample_idx}")

if __name__ == "__main__":
    main(denormalize=True) 