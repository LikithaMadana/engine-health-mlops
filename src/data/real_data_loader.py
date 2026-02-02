"""
Real Engine Health Data Loader

Loads actual engine sensor data from CSV file.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path


def load_real_engine_data(file_path: str = 'data/raw/engine_data.csv') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real engine health data from CSV file.
    
    Your dataset has 6 features:
    1. Engine rpm
    2. Lub oil pressure
    3. Fuel pressure
    4. Coolant pressure
    5. lub oil temp
    6. Coolant temp
    
    And 1 label:
    - Engine Condition (0 = healthy, 1 = unhealthy)
    
    Args:
        file_path: Path to CSV file containing engine data
        
    Returns:
        Tuple containing:
        - X: Feature array of shape (n_samples, 6)
        - y: Label array of shape (n_samples,)
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is incorrect
    """
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"Please ensure your dataset is in the correct location."
        )
    
    # Load CSV
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Clean column names (remove spaces, lowercase, replace spaces with underscores)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Expected label column
    label_col = 'engine_condition'
    
    # Verify label column exists
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in dataset.\n"
            f"Available columns: {list(df.columns)}"
        )
    
    # Get feature columns (all except label)
    feature_cols = [col for col in df.columns if col != label_col]
    
    # Extract features and labels
    X = df[feature_cols].values
    y = df[label_col].values
    
    # Validate data
    if X.shape[1] != 6:
        print(f"Warning: Expected 6 features, got {X.shape[1]}")
        print(f"Features: {feature_cols}")
    
    # Check for missing values
    if np.isnan(X).any():
        raise ValueError("Features contain NaN values. Please clean your data.")
    
    if np.isnan(y).any():
        raise ValueError("Labels contain NaN values. Please clean your data.")
    
    # Verify label format
    unique_labels = np.unique(y)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(
            f"Labels must be binary (0 or 1). Found: {unique_labels}"
        )
    
    # Print summary
    print(f"✓ Successfully loaded {len(X):,} samples")
    print(f"✓ Features: {X.shape[1]} columns - {feature_cols}")
    print(f"✓ Class distribution:")
    print(f"    Class 0 (Healthy):   {(y==0).sum():,} samples ({(y==0).sum()/len(y)*100:.2f}%)")
    print(f"    Class 1 (Unhealthy): {(y==1).sum():,} samples ({(y==1).sum()/len(y)*100:.2f}%)")
    print(f"✓ Feature value ranges:")
    for i, col in enumerate(feature_cols):
        print(f"    {col:20s}: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")
    
    return X, y


def get_feature_names() -> list:
    """
    Get the feature names from the real dataset.
    
    Returns:
        List of feature names
    """
    return [
        'engine_rpm',
        'lub_oil_pressure',
        'fuel_pressure',
        'coolant_pressure',
        'lub_oil_temp',
        'coolant_temp'
    ]


if __name__ == "__main__":
    """Test the data loader"""
    print("="*80)
    print("TESTING REAL DATA LOADER")
    print("="*80)
    
    try:
        # Load data
        X, y = load_real_engine_data()
        
        print(f"\nData shapes:")
        print(f"  X: {X.shape}")
        print(f"  y: {y.shape}")
        
        print(f"\nData types:")
        print(f"  X: {X.dtype}")
        print(f"  y: {y.dtype}")
        
        print(f"\nSample data (first 3 rows):")
        print(f"  X[0]: {X[0]}")
        print(f"  y[0]: {y[0]}")
        print(f"  X[1]: {X[1]}")
        print(f"  y[1]: {y[1]}")
        print(f"  X[2]: {X[2]}")
        print(f"  y[2]: {y[2]}")
        
        print("\n" + "="*80)
        print("✓ DATA LOADER TEST PASSED!")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ DATA LOADER TEST FAILED!")
        print("="*80)
        print(f"Error: {str(e)}")
        raise
