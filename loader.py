import os
import torch
import sqlite3
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset

class DrivAerNetSQLDataset(Dataset):
    """Dataset class for loading point clouds from SQLite database"""
    def __init__(
        self, 
        db_path: str
    ):
        """
        Args:
            db_path (str): Path to SQLite database containing point clouds
            csv_file (str): Path to CSV file with metadata
        """
        self.db_path = db_path
        
        # Verify all designs exist and compute global normalization parameters
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM point_clouds")
            self.available_ids = set(row[0] for row in cursor.fetchall())
            
            # Compute global min and max for xyz coordinates and pressure
            cursor.execute("SELECT data FROM point_clouds")
            xyz_min = np.array([float('inf')] * 3)
            xyz_max = np.array([float('-inf')] * 3)
            pressure_min = float('inf')
            pressure_max = float('-inf')
            
            for row in cursor.fetchall():
                points = np.frombuffer(row[0], dtype=np.float32).reshape(8192, 4)
                xyz_min = np.minimum(xyz_min, points[:, :3].min(axis=0))
                xyz_max = np.maximum(xyz_max, points[:, :3].max(axis=0))
                pressure_min = min(pressure_min, points[:, 3].min())
                pressure_max = max(pressure_max, points[:, 3].max())
            
            # Store normalization parameters as tensors
            self.xyz_min = torch.from_numpy(xyz_min)
            self.xyz_max = torch.from_numpy(xyz_max)
            self.xyz_range = self.xyz_max - self.xyz_min
            self.pressure_min = torch.tensor(pressure_min)
            self.pressure_max = torch.tensor(pressure_max)
            self.pressure_range = self.pressure_max - self.pressure_min

    def __len__(self):
        return len(self.available_ids)

    def __getitem__(self, idx):
        # Load point cloud from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, data FROM point_clouds WHERE id=?", (idx,))
            result = cursor.fetchone()
            
            if result is None:
                # Fallback: get the first available point cloud instead of raising an error
                print(f"Warning: No data found for index {idx}, using fallback data")
                cursor.execute("SELECT id, data FROM point_clouds LIMIT 1")
                result = cursor.fetchone()
                if result is None:
                    raise RuntimeError("Database appears to be empty")
                    
            point_id, binary_data = result
            
        # Convert binary data to tensor (already normalized)
        point_cloud_array = np.frombuffer(binary_data, dtype=np.float32).reshape(8192, 4)
        # Make array writable and convert to correct shape
        point_cloud_array = point_cloud_array.copy()  # Make writable
        
        # Sample num_points if less than total available points
        # if self.num_points < 8192:
        #     # Randomly sample indices without replacement
        #     sample_indices = np.random.choice(8192, self.num_points, replace=False)
        #     point_cloud_array = point_cloud_array[sample_indices]
        
        return {
            'id': point_id,
            'points': point_cloud_array  # Shape: (num_points, 4)
        }

def get_dataloaders(
    db_path: str,
    batch_size: int,
    num_workers: int
) -> tuple:
    """
    Prepare and return the training, validation, and test DataLoader objects.

    Args:
        db_path (str): Path to SQLite database containing point clouds
        batch_size (int): The number of samples per batch to load
        num_workers (int): Number of worker processes for data loading

    Returns:
        tuple: A tuple containing the training DataLoader, validation DataLoader, and test DataLoader
    """
    full_dataset = DrivAerNetSQLDataset(db_path=db_path)
    
    train_ids = pd.read_csv('train_val_test_splits/train_design_ids.txt', header=None).values.flatten()
    val_ids = pd.read_csv('train_val_test_splits/val_design_ids.txt', header=None).values.flatten()
    test_ids = pd.read_csv('train_val_test_splits/test_design_ids.txt', header=None).values.flatten()
    
    train_dataset = Subset(full_dataset, train_ids)
    val_dataset = Subset(full_dataset, val_ids)
    test_dataset = Subset(full_dataset, test_ids)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
    
    return train_dataloader, val_dataloader, test_dataloader