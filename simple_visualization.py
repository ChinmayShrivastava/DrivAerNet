import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pickle
import sqlite3

st.set_page_config(layout="wide", page_title="Simple DrivAerNet Visualization")

st.title("Simple DrivAerNet Visualization")

# Function to load data
@st.cache_data
def load_test_data():
    try:
        with open('point_transformer_predictions_4.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# Function to load normalization parameters
@st.cache_data
def load_normalization_params(db_path="point_clouds_pressure_stratified.db"):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM point_clouds LIMIT 1")
            sample_data = cursor.fetchone()[0]
            sample_points = np.frombuffer(sample_data, dtype=np.float32).reshape(8192, 7)
            
            cursor.execute("SELECT data FROM point_clouds")
            xyz_min = np.array([float('inf')] * 3)
            xyz_max = np.array([float('-inf')] * 3)
            pressure_min = float('inf')
            pressure_max = float('-inf')
            
            for i, row in enumerate(cursor.fetchall()):
                if i > 100:  # Limit to 100 samples for speed
                    break
                points = np.frombuffer(row[0], dtype=np.float32).reshape(8192, 7)
                xyz_min = np.minimum(xyz_min, points[:, :3].min(axis=0))
                xyz_max = np.maximum(xyz_max, points[:, :3].max(axis=0))
                pressure_min = min(pressure_min, points[:, 3].min())
                pressure_max = max(pressure_max, points[:, 3].max())
            
            return {
                'xyz_min': xyz_min,
                'xyz_range': xyz_max - xyz_min,
                'pressure_min': pressure_min,
                'pressure_range': pressure_max - pressure_min
            }
    except Exception as e:
        st.warning(f"Could not load normalization parameters: {e}")
        return None

# Function to denormalize points
def denormalize_points(points, pressure, norm_params):
    if norm_params is None:
        return points, pressure
    
    denorm_xyz = points.copy()
    denorm_pressure = pressure.copy()
    
    for i in range(3):
        denorm_xyz[:, i] = points[:, i] * norm_params['xyz_range'][i] + norm_params['xyz_min'][i]
    
    denorm_pressure = pressure * norm_params['pressure_range'] + norm_params['pressure_min']
    
    return denorm_xyz, denorm_pressure

# Load data
test_data = load_test_data()
if test_data is None:
    st.error("Please make sure your test_data.pkl file is in the same directory as this app.")
    st.stop()

# Load normalization parameters
norm_params = load_normalization_params()

# Sidebar controls
st.sidebar.header("Controls")

# Sample selection
batch_idx = st.sidebar.slider("Batch Index", 0, len(test_data['inputs'])-1, 0)
sample_idx = st.sidebar.slider("Sample Index", 0, len(test_data['inputs'][0])-1, 0)

# Visualization settings
point_size = st.sidebar.slider("Point Size", 1, 10, 3)
use_denormalized = st.sidebar.checkbox("Use denormalized data", value=True)

# Get the selected data
points = test_data['inputs'][batch_idx][sample_idx]
gt_pressure = test_data['ground_truth'][batch_idx][sample_idx]
pred_pressure = test_data['predictions'][batch_idx][sample_idx]

# Denormalize if requested
if use_denormalized and norm_params:
    points_viz, gt_pressure_viz = denormalize_points(points, gt_pressure, norm_params)
    _, pred_pressure_viz = denormalize_points(points, pred_pressure, norm_params)
else:
    points_viz, gt_pressure_viz, pred_pressure_viz = points, gt_pressure, pred_pressure

# Calculate error metrics
mse = np.mean((gt_pressure - pred_pressure)**2)
mae = np.mean(np.abs(gt_pressure - pred_pressure))

# Display error metrics
st.header("Error Metrics")
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", f"{mse:.6f}")
col2.metric("Mean Absolute Error", f"{mae:.6f}")

# Create side-by-side visualization
st.header("3D Point Cloud Visualization")
col1, col2 = st.columns(2)

# Set color scale limits
vmin = min(gt_pressure_viz.min(), pred_pressure_viz.min())
vmax = max(gt_pressure_viz.max(), pred_pressure_viz.max())

# Ground Truth plot
with col1:
    fig_gt = go.Figure()
    fig_gt.add_trace(
        go.Scatter3d(
            x=points_viz[:, 0],
            y=points_viz[:, 1],
            z=points_viz[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=gt_pressure_viz.flatten(),
                colorscale='viridis',
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title="Pressure")
            ),
            name="Ground Truth"
        )
    )
    fig_gt.update_layout(
        title="Ground Truth Pressure",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        height=600
    )
    st.plotly_chart(fig_gt, use_container_width=True)

# Prediction plot
with col2:
    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter3d(
            x=points_viz[:, 0],
            y=points_viz[:, 1],
            z=points_viz[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=pred_pressure_viz.flatten(),
                colorscale='viridis',
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title="Pressure")
            ),
            name="Prediction"
        )
    )
    fig_pred.update_layout(
        title="Predicted Pressure",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        height=600
    )
    st.plotly_chart(fig_pred, use_container_width=True) 