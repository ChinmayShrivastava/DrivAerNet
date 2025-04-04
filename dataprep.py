import os
import zipfile
import sqlite3
import torch
from pathlib import Path
from transform import extract_point_cloud_data
from tqdm import tqdm

def process_vtk_files(base_folder, n_points=8192, db_path='point_clouds_pressure_stratified.db', resume=True):
    """
    Process all VTK files from zip archives in the given folder and store point cloud data in SQLite.
    
    Parameters:
    base_folder (str): Path to folder containing zip files
    n_points (int): Number of points to sample from each mesh
    db_path (str): Path to SQLite database
    resume (bool): Whether to resume from last processed zip file
    """
    # Initialize SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS point_clouds
        (id TEXT PRIMARY KEY, data BLOB)
    ''')
    
    # Create a progress tracking table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_progress
        (zip_file TEXT PRIMARY KEY, processed INTEGER)
    ''')
    conn.commit()
    
    # Process each zip file in the base folder
    zip_files = list(Path(base_folder).glob('*.zip'))
    
    # Get already processed zip files if resuming
    processed_zips = set()
    if resume:
        cursor.execute('SELECT zip_file FROM processing_progress WHERE processed = 1')
        processed_zips = {row[0] for row in cursor.fetchall()}
        print(f"Resuming: {len(processed_zips)} zip files already processed")
    
    # Filter out already processed zip files
    zip_files_to_process = [z for z in zip_files if z.name not in processed_zips]
    
    for zip_path in tqdm(zip_files_to_process, desc="Processing zip files"):
        # Skip macOS system files
        if zip_path.name.startswith('._'):
            print(f"Skipping system file: {zip_path.name}")
            continue
            
        print(f"\nProcessing {zip_path.name}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Create temporary directory for extraction
                temp_dir = Path('temp_extract')
                temp_dir.mkdir(exist_ok=True)
                
                try:
                    # Extract zip file
                    zip_ref.extractall(temp_dir)
                    
                    # Navigate to PressureVTK folder and its only subfolder
                    point_dir = temp_dir / 'PressureVTK'
                    if not point_dir.exists():
                        print(f"No PressureVTK folder in {zip_path.name}, skipping...")
                        continue
                    
                    # Get the only subfolder in PressureVTK
                    subfolders = list(point_dir.iterdir())
                    if not subfolders or not subfolders[0].is_dir():
                        print(f"No valid subfolder in PressureVTK directory of {zip_path.name}, skipping...")
                        continue
                    
                    vtk_folder = subfolders[0]
                    
                    # Process each VTK file
                    vtk_files = list(vtk_folder.glob('*.vtk'))
                    for vtk_file in tqdm(vtk_files, desc=f"Processing VTK files in {zip_path.name}", leave=False):
                        try:
                            # Extract point cloud data
                            point_cloud = extract_point_cloud_data(str(vtk_file), n_points=n_points)
                            
                            # Convert tensor to binary for storage
                            binary_data = sqlite3.Binary(point_cloud.numpy().tobytes())
                            
                            # Store in database
                            cursor.execute(
                                'INSERT OR REPLACE INTO point_clouds (id, data) VALUES (?, ?)',
                                (vtk_file.stem, binary_data)
                            )
                            
                            print(f"Processed {vtk_file.name}")
                            
                        except Exception as e:
                            print(f"Error processing {vtk_file.name}: {str(e)}")
                    
                    # Mark this zip file as processed
                    cursor.execute(
                        'INSERT OR REPLACE INTO processing_progress (zip_file, processed) VALUES (?, ?)',
                        (zip_path.name, 1)
                    )
                    conn.commit()
                    
                finally:
                    # Cleanup: remove temporary directory
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Error processing zip file {zip_path.name}: {str(e)}")
            # Don't mark as processed if there was an error
    
    conn.close()

if __name__ == "__main__":
    # Example usage
    base_folder = os.path.expanduser("/Volumes/Backup Plus/temp/")
    process_vtk_files(
        base_folder, 
        n_points=20_000, 
        db_path='point_clouds_pressure_stratified_20_000.db',
        resume=True
    )