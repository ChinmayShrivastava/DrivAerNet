import sqlite3
import numpy as np
import torch
from tqdm import tqdm

def normalize_and_save_database(input_db_path: str, output_db_path: str):
    """
    Create a new database with normalized point cloud data, keeping only x, y, z, and pressure.
    
    Args:
        input_db_path (str): Path to the original SQLite database
        output_db_path (str): Path where the new normalized database will be saved
    """
    # First pass: compute global normalization parameters
    print("Computing normalization parameters...")
    with sqlite3.connect(input_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM point_clouds")
        
        xyz_min = np.array([float('inf')] * 3)
        xyz_max = np.array([float('-inf')] * 3)
        pressure_min = float('inf')
        pressure_max = float('-inf')
        
        for row in tqdm(cursor.fetchall()):
            points = np.frombuffer(row[0], dtype=np.float32).reshape(8192, 7)
            xyz_min = np.minimum(xyz_min, points[:, :3].min(axis=0))
            xyz_max = np.maximum(xyz_max, points[:, :3].max(axis=0))
            pressure_min = min(pressure_min, points[:, 3].min())
            pressure_max = max(pressure_max, points[:, 3].max())
        
        xyz_range = xyz_max - xyz_min
        pressure_range = pressure_max - pressure_min

    # Create new database and copy normalized data
    print("Creating new database with normalized values...")
    with sqlite3.connect(input_db_path) as in_conn, \
         sqlite3.connect(output_db_path) as out_conn:
        
        in_cursor = in_conn.cursor()
        out_cursor = out_conn.cursor()
        
        # Create the new table with TEXT id
        out_cursor.execute("""
            CREATE TABLE point_clouds (
                id TEXT PRIMARY KEY,
                data BLOB NOT NULL
            )
        """)
        
        # Get all records
        in_cursor.execute("SELECT id, data FROM point_clouds")
        
        # Process and save each point cloud
        for id_, binary_data in tqdm(in_cursor.fetchall()):
            # Load and reshape the original data
            points = np.frombuffer(binary_data, dtype=np.float32).reshape(8192, 7)
            
            # Normalize xyz coordinates
            xyz = (points[:, :3] - xyz_min) / xyz_range
            
            # Normalize pressure
            pressure = (points[:, 3:4] - pressure_min) / pressure_range
            
            # Combine normalized xyz and pressure (4 columns total)
            normalized_points = np.concatenate([xyz, pressure], axis=1)
            
            # Convert to float32 and save
            normalized_binary = normalized_points.astype(np.float32).tobytes()
            
            try:
                out_cursor.execute("INSERT INTO point_clouds (id, data) VALUES (?, ?)",
                                 (id_, normalized_binary))
            except sqlite3.Error as e:
                print(f"Error inserting record {id_}: {e}")
                continue
        
        out_conn.commit()
    
    print(f"Normalized database saved to: {output_db_path}")
    print(f"Original data shape per point cloud: (8192, 7)")
    print(f"New data shape per point cloud: (8192, 4)")

if __name__ == "__main__":
    input_db_path = "point_clouds_pressure_stratified.db"
    output_db_path = "point_clouds_pressure_stratified_normalized.db"
    normalize_and_save_database(input_db_path, output_db_path) 