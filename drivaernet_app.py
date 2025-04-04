import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import random
from plotly.subplots import make_subplots
import sqlite3
import open3d as o3d
from stpyvista import stpyvista

st.set_page_config(layout="wide", page_title="DrivAerNet Visualization")

st.title("DrivAerNet Model Visualization")

# Function to load normalization parameters from the database
@st.cache_data
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
            
            st.write("Normalization parameters:")
            st.write(f"XYZ min: {xyz_min}")
            st.write(f"XYZ max: {xyz_max}")
            st.write(f"XYZ range: {xyz_range}")
            st.write(f"Pressure min: {pressure_min}")
            st.write(f"Pressure max: {pressure_max}")
            
            return {
                'xyz_min': xyz_min,
                'xyz_range': xyz_range,
                'pressure_min': pressure_min,
                'pressure_range': pressure_range
            }
    except Exception as e:
        st.warning(f"Could not load normalization parameters: {e}")
        # Provide default normalization parameters based on preprocess_database.py
        # These are approximate values that might help if the database isn't available
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

# Function to create Open3D point cloud with colors
def create_o3d_point_cloud(points, values, cmap_name='viridis', vmin=None, vmax=None):
    """
    Create an Open3D point cloud with colors based on scalar values.
    
    Args:
        points: XYZ coordinates (N, 3)
        values: Scalar values for coloring (N,)
        cmap_name: Name of the colormap to use
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        
    Returns:
        Open3D point cloud with colors
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Normalize values for colormap
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    
    # Get colormap and normalize values
    cmap = plt.get_cmap(cmap_name)
    normalized_values = (values - vmin) / (vmax - vmin)
    colors = cmap(normalized_values)[:, :3]  # Get RGB values
    
    # Set colors
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

# Function to load data - replace this with your actual data loading logic
@st.cache_data
def load_test_data():
    # This is a placeholder - replace with your actual data loading code
    # For example, you might load from a pickle file or other source
    import pickle
    with open('point_transformer_predictions_4.pkl', 'rb') as f:
        return pickle.load(f)
    
    # If you don't have a pickle file, you can use this placeholder
    # which you should replace with your actual data loading logic
    # return {
    #     'inputs': [np.random.rand(10, 4096, 3)],  # Example shape
    #     'ground_truth': [np.random.rand(10, 4096, 1)],
    #     'predictions': [np.random.rand(10, 4096, 1)]
    # }

# Try to load the data
try:
    test_data = load_test_data()
    data_loaded = True
    
    # Get the dimensions of the data
    num_batches = len(test_data['inputs'])
    samples_per_batch = len(test_data['inputs'][0]) if num_batches > 0 else 0
    
    st.success(f"Data loaded successfully! Found {num_batches} batches with {samples_per_batch} samples each.")
    
    # Load normalization parameters
    norm_params = load_normalization_params()
    if norm_params:
        st.success("Normalization parameters loaded successfully for denormalization.")
    else:
        st.warning("Using normalized data for visualization (denormalization parameters not available).")
    
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.info("Please make sure your test_data.pkl file is in the same directory as this app.")
    data_loaded = False

if data_loaded:
    # Sidebar controls
    st.sidebar.header("Sample Selection")
    
    # Option to randomly select or manually choose indices
    selection_mode = st.sidebar.radio(
        "Selection Mode",
        ["Random", "Manual"]
    )
    
    # Option to use denormalized data
    use_denormalized = st.sidebar.checkbox("Use denormalized data", value=True)
    
    # Add point size control slider
    point_size = st.sidebar.slider("Point Size", min_value=1, max_value=10, value=3, step=1, 
                                  help="Control the size of points in the 3D visualization")
    
    # Visualization settings
    st.sidebar.subheader("Visualization Settings")
    
    # Color map selection
    cmap_options = ["viridis", "plasma", "inferno", "magma", "cividis", "jet", "rainbow", "turbo"]
    selected_cmap = st.sidebar.selectbox("Color Map", cmap_options, index=0)
    
    # Difference colormap (for diverging data)
    diff_cmap_options = ["coolwarm", "bwr", "seismic", "spectral", "RdBu", "RdYlBu"]
    diff_cmap = st.sidebar.selectbox("Difference Color Map", diff_cmap_options, index=0)
    
    # Background color options
    bg_color_options = ["white", "black", "gray", "lightblue"]
    selected_bg_color = st.sidebar.selectbox("Background Color", bg_color_options, index=0)
    
    # Advanced visualization options
    st.sidebar.subheader("Advanced Options")
    
    # Point cloud filtering/downsampling (optional)
    enable_downsampling = st.sidebar.checkbox("Enable Point Cloud Downsampling", value=False,
                                            help="Reduce the number of points for better performance")
    
    if enable_downsampling:
        downsample_factor = st.sidebar.slider("Downsample Factor", min_value=1, max_value=10, value=2, step=1,
                                           help="Higher values show fewer points")
    else:
        downsample_factor = 1
    
    # Show axes indicator (orientation widget)
    show_axes = st.sidebar.checkbox("Show Orientation Axes", value=True,
                                  help="Display orientation axes in the corner")
    
    # Add camera position controls
    st.sidebar.subheader("Camera Settings")
    use_fixed_camera = st.sidebar.checkbox("Use fixed camera position", value=True)
    
    if use_fixed_camera:
        camera_x = st.sidebar.slider("Camera X", -2.0, 2.0, 1.5, 0.1)
        camera_y = st.sidebar.slider("Camera Y", -2.0, 2.0, 1.5, 0.1)
        camera_z = st.sidebar.slider("Camera Z", -2.0, 2.0, 1.5, 0.1)
        camera_eye = [camera_x, camera_y, camera_z]
    else:
        camera_eye = None
    
    if selection_mode == "Random":
        if st.sidebar.button("Generate Random Sample"):
            batch_idx = random.randint(0, num_batches-1)
            sample_idx = random.randint(0, samples_per_batch-1)
            st.session_state['batch_idx'] = batch_idx
            st.session_state['sample_idx'] = sample_idx
    else:
        batch_idx = st.sidebar.slider("Batch Index", 0, num_batches-1, 
                                     st.session_state.get('batch_idx', 0))
        sample_idx = st.sidebar.slider("Sample Index", 0, samples_per_batch-1, 
                                      st.session_state.get('sample_idx', 0))
        st.session_state['batch_idx'] = batch_idx
        st.session_state['sample_idx'] = sample_idx
    
    # Display current selection
    st.sidebar.info(f"Current selection: Batch {st.session_state.get('batch_idx', 0)}, Sample {st.session_state.get('sample_idx', 0)}")
    
    # Get the selected data
    batch_idx = st.session_state.get('batch_idx', 0)
    sample_idx = st.session_state.get('sample_idx', 0)
    
    points = test_data['inputs'][batch_idx][sample_idx]  # Shape: (4096, 3)
    gt_pressure = test_data['ground_truth'][batch_idx][sample_idx]  # Shape: (4096, 1)
    pred_pressure = test_data['predictions'][batch_idx][sample_idx]  # Shape: (4096, 1)
    
    # Denormalize if requested and parameters are available
    if use_denormalized and norm_params:
        points_viz, gt_pressure_viz = denormalize_points(points, gt_pressure, norm_params)
        _, pred_pressure_viz = denormalize_points(points, pred_pressure, norm_params)
        data_label = "Denormalized"
    else:
        points_viz, gt_pressure_viz, pred_pressure_viz = points, gt_pressure, pred_pressure
        data_label = "Normalized"
    
    # Calculate pressure difference
    pressure_diff = gt_pressure_viz - pred_pressure_viz
    
    # Calculate error metrics
    mse = np.mean((gt_pressure - pred_pressure)**2)
    mae = np.mean(np.abs(gt_pressure - pred_pressure))
    
    # Display error metrics
    st.header("Error Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error", f"{mse:.6f}")
    col2.metric("Mean Absolute Error", f"{mae:.6f}")
    
    # Create 3D visualizations
    st.header(f"3D Point Cloud Visualization ({data_label} Data)")
    
    # Set color scale limits for consistent visualization
    vmin = min(gt_pressure_viz.min(), pred_pressure_viz.min())
    vmax = max(gt_pressure_viz.max(), pred_pressure_viz.max())
    diff_abs_max = max(abs(pressure_diff.min()), abs(pressure_diff.max()))
    
    # Apply downsampling if enabled
    if enable_downsampling and downsample_factor > 1:
        # Get indices for downsampled points
        num_points = points_viz.shape[0]
        indices = np.arange(0, num_points, downsample_factor)
        
        # Downsample data
        points_viz_ds = points_viz[indices]
        gt_pressure_viz_ds = gt_pressure_viz[indices]
        pred_pressure_viz_ds = pred_pressure_viz[indices]
        pressure_diff_ds = pressure_diff[indices]
        
        st.info(f"Displaying {len(indices)} points (downsampled from {num_points})")
    else:
        # Use full data
        points_viz_ds = points_viz
        gt_pressure_viz_ds = gt_pressure_viz
        pred_pressure_viz_ds = pred_pressure_viz
        pressure_diff_ds = pressure_diff
    
    # Create Open3D point clouds
    gt_pcd = create_o3d_point_cloud(points_viz_ds, gt_pressure_viz_ds.flatten(), 
                                   cmap_name=selected_cmap, vmin=vmin, vmax=vmax)
    pred_pcd = create_o3d_point_cloud(points_viz_ds, pred_pressure_viz_ds.flatten(), 
                                     cmap_name=selected_cmap, vmin=vmin, vmax=vmax)
    diff_pcd = create_o3d_point_cloud(points_viz_ds, pressure_diff_ds.flatten(), 
                                     cmap_name=diff_cmap, vmin=-diff_abs_max, vmax=diff_abs_max)
    
    # Create three columns for side-by-side visualization
    col1, col2, col3 = st.columns(3)
    
    # Visualization settings
    vis_settings = {
        "point_size": point_size,
        "background_color": selected_bg_color,
        "show_axes": show_axes,
        "camera_eye": camera_eye
    }
    
    # Ground Truth visualization
    with col1:
        st.subheader("Ground Truth Pressure")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(gt_pcd)
        
        # Set visualization options
        opt = vis.get_render_option()
        opt.point_size = vis_settings["point_size"]
        opt.background_color = np.asarray(plt.get_cmap(vis_settings["background_color"])(0))[:3]
        
        if vis_settings["show_axes"]:
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        
        if vis_settings["camera_eye"] is not None:
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, 1, 0])
            ctr.set_zoom(0.5)
            ctr.set_lookat([0, 0, 0])
            ctr.set_front(vis_settings["camera_eye"])
        
        # Capture the image
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        
        # Convert to numpy array and display
        img = np.asarray(img)
        st.image(img, use_column_width=True)
    
    # Prediction visualization
    with col2:
        st.subheader("Predicted Pressure")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pred_pcd)
        
        # Set visualization options
        opt = vis.get_render_option()
        opt.point_size = vis_settings["point_size"]
        opt.background_color = np.asarray(plt.get_cmap(vis_settings["background_color"])(0))[:3]
        
        if vis_settings["show_axes"]:
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        
        if vis_settings["camera_eye"] is not None:
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, 1, 0])
            ctr.set_zoom(0.5)
            ctr.set_lookat([0, 0, 0])
            ctr.set_front(vis_settings["camera_eye"])
        
        # Capture the image
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        
        # Convert to numpy array and display
        img = np.asarray(img)
        st.image(img, use_column_width=True)
    
    # Difference visualization
    with col3:
        st.subheader("Pressure Difference (GT - Pred)")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(diff_pcd)
        
        # Set visualization options
        opt = vis.get_render_option()
        opt.point_size = vis_settings["point_size"]
        opt.background_color = np.asarray(plt.get_cmap(vis_settings["background_color"])(0))[:3]
        
        if vis_settings["show_axes"]:
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        
        if vis_settings["camera_eye"] is not None:
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, 1, 0])
            ctr.set_zoom(0.5)
            ctr.set_lookat([0, 0, 0])
            ctr.set_front(vis_settings["camera_eye"])
        
        # Capture the image
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        
        # Convert to numpy array and display
        img = np.asarray(img)
        st.image(img, use_column_width=True)
    
    # Create 2D visualizations
    st.header("2D Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 2D histogram of predicted vs ground truth
        fig_hist2d = go.Figure()
        fig_hist2d.add_trace(go.Histogram2d(
            x=gt_pressure_viz_ds.flatten(),
            y=pred_pressure_viz_ds.flatten(),
            colorscale='Blues',
            nbinsx=50,
            nbinsy=50,
            colorbar=dict(title="Count")
        ))
        
        # Add identity line
        gt_min, gt_max = gt_pressure_viz_ds.min(), gt_pressure_viz_ds.max()
        fig_hist2d.add_trace(go.Scatter(
            x=[gt_min, gt_max],
            y=[gt_min, gt_max],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Identity Line'
        ))
        
        fig_hist2d.update_layout(
            title="Predicted vs Ground Truth Pressure Values",
            xaxis_title="Ground Truth Pressure",
            yaxis_title="Predicted Pressure",
            height=500
        )
        
        st.plotly_chart(fig_hist2d, use_container_width=True)
    
    with col2:
        # Histogram of differences
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=pressure_diff_ds.flatten(),
            nbinsx=50,
            marker_color='skyblue',
            marker_line_color='black',
            marker_line_width=1
        ))
        
        # Add vertical line at x=0
        fig_hist.add_vline(
            x=0, 
            line_color='red',
            line_dash='dash'
        )
        
        fig_hist.update_layout(
            title="Distribution of Prediction Errors",
            xaxis_title="Pressure Difference (GT - Pred)",
            yaxis_title="Frequency",
            height=500
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.warning("Please upload your test data to use this application.")
    
    # Placeholder for data upload (if you want to implement this feature)
    uploaded_file = st.file_uploader("Upload test data (pickle format)", type="pkl")
    if uploaded_file is not None:
        # Process the uploaded file
        st.info("Processing uploaded file...")
        # Add your processing logic here 