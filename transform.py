import torch
import numpy as np
import pyvista as pv

def extract_point_cloud_data(vtk_file_path, n_points=100000):
    """
    Extract point cloud data with pressure values and surface normals from a VTK file,
    using pressure-stratified sampling to ensure even representation across pressure ranges.

    Parameters:
    vtk_file_path (str): Path to the VTK file
    n_points (int): Number of points to sample

    Returns:
    torch.Tensor: Point cloud tensor of shape [n, 7] containing:
        - xyz coordinates (3)
        - pressure value (1)
        - normal vectors (3)
    """
    import torch
    import numpy as np
    import pyvista as pv

    # Load the mesh
    mesh = pv.read(vtk_file_path)
    
    # Compute surface normals if they don't exist
    if 'Normals' not in mesh.point_data:
        mesh.compute_normals(inplace=True)
    
    # Get all pressure values
    all_pressures = mesh.point_data['p']
    min_pressure = np.min(all_pressures)
    max_pressure = np.max(all_pressures)
    
    # Create pressure bins for stratified sampling
    n_bins = min(100, n_points)  # Number of bins to use
    pressure_bins = np.linspace(min_pressure, max_pressure, n_bins + 1)
    
    # Determine which bin each point belongs to
    bin_indices = np.digitize(all_pressures, pressure_bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Ensure valid indices
    
    # Calculate how many points to sample from each bin
    points_per_bin = n_points // n_bins
    remainder = n_points % n_bins
    
    # Sample points from each bin
    selected_indices = []
    
    for bin_idx in range(n_bins):
        # Find points in this pressure bin
        bin_points = np.where(bin_indices == bin_idx)[0]
        
        # Determine number of points to sample from this bin
        n_to_sample = points_per_bin + (1 if bin_idx < remainder else 0)
        
        if len(bin_points) > 0:
            # If we have points in this bin, sample with replacement if necessary
            bin_samples = np.random.choice(
                bin_points, 
                size=min(n_to_sample, len(bin_points)),
                replace=False
            )
            selected_indices.extend(bin_samples)
        
    # If we couldn't fill our quota from some bins, sample more from others
    if len(selected_indices) < n_points:
        remaining = n_points - len(selected_indices)
        # Sample from all points, excluding already selected ones
        all_indices = set(range(len(all_pressures)))
        available = list(all_indices - set(selected_indices))
        
        if len(available) >= remaining:
            additional = np.random.choice(available, size=remaining, replace=False)
        else:
            # If we need more points than available, sample with replacement from all points
            additional = np.random.choice(range(len(all_pressures)), size=remaining, replace=True)
            
        selected_indices.extend(additional)
    
    # Ensure we don't exceed n_points
    selected_indices = selected_indices[:n_points]
    
    # Extract coordinates, pressure, and normals for selected points
    points = mesh.points[selected_indices]
    pressures = all_pressures[selected_indices]
    normals = mesh.point_data['Normals'][selected_indices]

    # Combine data and convert to torch tensor
    combined_data = np.hstack([
        points,                    # xyz coordinates
        pressures.reshape(-1, 1),  # pressure values
        normals                    # normal vectors
    ])
    
    return torch.from_numpy(combined_data).float()

# Example usage:
if __name__ == "__main__":
    vtk_file_path = 'F_D_WM_WW_0001.vtk'
    
    # Define fixed camera positions for consistent benchmarking
    # Each position is (camera_location, focal_point, view_up_direction)
    camera_positions = [
        [(0, 0, 10), (0, 0, 0), (0, 1, 0)],  # Front view
        [(10, 0, 0), (0, 0, 0), (0, 1, 0)],  # Right side view
        [(-10, 0, 0), (0, 0, 0), (0, 1, 0)], # Left side view
        [(0, 10, 0), (0, 0, 0), (0, 0, 1)],  # Top view
        [(5, 5, 5), (0, 0, 0), (0, 1, 0)]    # Isometric view
    ]
    
    # Extract point clouds with different resolutions for benchmarking
    point_cloud_resolutions = [8192, 100000, 500000, 1000000]
    
    # Track timing information for benchmarking
    import time
    timing_data = {}
    
    # Process each resolution
    for n_points in point_cloud_resolutions:
        start_time = time.time()
        point_cloud = extract_point_cloud_data(vtk_file_path, n_points=n_points)
        extraction_time = time.time() - start_time
        
        print(f"Point cloud with {n_points} points extracted in {extraction_time:.2f} seconds")
        print(f"Shape: {point_cloud.shape}")
        
        timing_data[n_points] = {
            'extraction_time': extraction_time,
            'rendering_times': []
        }
        
        # For each camera position, create and save images
        for i, camera_position in enumerate(camera_positions):
            angle_num = i + 1
            
            # Visualize and save point cloud
            start_time = time.time()
            plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
            plotter.background_color = 'white'  # Clean white background
            points = point_cloud[:, :3].numpy()
            pressure = point_cloud[:, 3].numpy()
            cloud = pv.PolyData(points)
            
            # Add points with improved styling
            plotter.add_points(cloud, scalars=pressure, cmap='plasma', point_size=20)
            
            # Add a color bar with better formatting
            plotter.add_scalar_bar(title='Pressure', n_labels=5, italic=False, bold=False, 
                                  font_family='arial', shadow=False, fmt='%.2e')
            
            # Add a more subtle, professional title
            plotter.add_text(f"Pressure Distribution ({n_points} points) - View {angle_num}", 
                           font_size=14, position='upper_left')
            
            # Use predefined camera position
            plotter.camera_position = camera_position
            
            # Save with resolution in filename
            plotter.screenshot(f"point_cloud_{n_points}_angle{angle_num}.png")
            render_time = time.time() - start_time
            timing_data[n_points]['rendering_times'].append(render_time)
            
            print(f"  Angle {angle_num} rendered in {render_time:.2f} seconds")
            plotter.close()
    
    # Save benchmark results
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(timing_data, f, indent=2)
    
    print("Benchmark completed and results saved!")