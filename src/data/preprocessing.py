"""
Data preprocessing module for engine health prediction.
Aligned with baseline conventions: simple functions and Dataset class.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


# --- Baseline Dataset Class (from user's baseline script) ---
class EngineDataset(Dataset):
    """
    Dataset with sliding window logic for engine health data.
    Follows baseline architecture.
    """
    def __init__(self, features, targets, window_size: int = 10):
        """
        Initialize dataset with windowing.
        
        Args:
            features: Feature array (numpy array)
            targets: Target array (numpy array)
            window_size: Size of sliding window (default: 10, from baseline)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.window_size = window_size

    def __len__(self):
        """Number of available windows"""
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        """
        Get window at index.
        
        Returns:
            x: Window of features (window_size, n_features)
            y: Target at end of window (scalar)
        """
        x = self.features[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size - 1]
        return x, y


# --- Baseline Data Preparation Function (from user's baseline script) ---
def prepare_data(filepath: str, config: dict) -> Tuple:
    """
    Load and prepare data from CSV file.
    Follows baseline architecture.
    
    Args:
        filepath: Path to CSV file
        config: Configuration dict with 'test_size' key
        
    Returns:
        Tuple of (train_X, train_y, test_X, test_y, scaler)
    """
    df = pd.read_csv(filepath)
    
    target_col = 'Engine Condition'
    feature_cols = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 
                    'Coolant pressure', 'lub oil temp', 'Coolant temp']
    
    # 1. Standardization
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 2. Split before windowing
    train_df, test_df = train_test_split(
        df, test_size=config["test_size"], random_state=42, shuffle=False
    )
    
    return (train_df[feature_cols].values, train_df[target_col].values,
            test_df[feature_cols].values, test_df[target_col].values, scaler)


# --- Legacy Support: Keep old class for backward compatibility ---
class EngineDataPreprocessor:
    """
    Legacy preprocessor class for backward compatibility.
    New code should use prepare_data() and EngineDataset instead.
    """
    
    def __init__(self, window_size: int = 10, test_size: float = 0.2, random_state: int = 42):
        """Initialize with baseline defaults (window_size=10)."""
        self.window_size = window_size
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self, n_samples: int = 1000, n_features: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic engine health data.
        Note: Baseline uses 6 features (not 14).
        """
        np.random.seed(self.random_state)
        
        # Generate features
        healthy_data = np.random.randn(n_samples // 2, n_features) * 0.5 + 0.3
        unhealthy_data = np.random.randn(n_samples // 2, n_features) * 1.2 + 1.5
        
        X = np.vstack([healthy_data, unhealthy_data])
        y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
        
        # Shuffle
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]
        
        return X, y
    
    def create_sliding_windows(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from time series data.
        
        Args:
            X: Input features (2D array: samples x features)
            y: Labels (1D array)
            
        Returns:
            Tuple of (windowed features, windowed labels)
        """
        n_windows = len(X) - self.window_size + 1
        X_windows = np.array([X[i:i + self.window_size] for i in range(n_windows)])
        y_windows = y[self.window_size - 1:]
        
        return X_windows, y_windows
    
    def prepare_data(self, 
                    X: Optional[np.ndarray] = None, 
                    y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete data preparation pipeline.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Generate if not provided
        if X is None or y is None:
            X, y = self.generate_synthetic_data()
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Standardize
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data_for_cnn(self,
                            X: Optional[np.ndarray] = None,
                            y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare windowed data for CNN.
        
        Returns:
            Tuple of (X_train_windowed, X_test_windowed, y_train, y_test)
        """
        # Get preprocessed data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Create sliding windows
        X_train_windowed, y_train_windowed = self.create_sliding_windows(X_train, y_train)
        X_test_windowed, y_test_windowed = self.create_sliding_windows(X_test, y_test)
        
        return X_train_windowed, X_test_windowed, y_train_windowed, y_test_windowed
