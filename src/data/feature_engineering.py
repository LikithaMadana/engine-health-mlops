"""
Feature engineering module for XGBoost model.
Extracts statistical, temporal, and domain-specific features from time-series windows.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from scipy import stats


class EngineFeatureEngineer:
    """
    Feature engineering for engine health prediction (XGBoost).
    Extracts meaningful features from raw sensor data windows.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize feature engineer.
        
        Args:
            window_size: Size of sliding window for feature extraction
        """
        self.window_size = window_size
        self.feature_names = []
        
    def extract_statistical_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from a single window.
        
        Args:
            window: Array of shape (window_size, n_sensors)
            
        Returns:
            1D array of statistical features
        """
        features = []
        
        # Ensure window is 2D
        if len(window.shape) == 1:
            window = window.reshape(1, -1)
        
        # For each sensor
        for sensor_idx in range(window.shape[1]):
            sensor_data = window[:, sensor_idx]
            
            # Basic statistics
            features.append(np.mean(sensor_data))      # Mean
            features.append(np.std(sensor_data))       # Standard deviation
            features.append(np.min(sensor_data))       # Minimum
            features.append(np.max(sensor_data))       # Maximum
            features.append(np.median(sensor_data))    # Median
            
            # Range and variation
            features.append(np.max(sensor_data) - np.min(sensor_data))  # Range
            
            # Coefficient of variation (std/mean) - handles division by zero
            mean_val = np.mean(sensor_data)
            cv = np.std(sensor_data) / mean_val if mean_val != 0 else 0
            features.append(cv)
            
            # Percentiles
            features.append(np.percentile(sensor_data, 25))   # 25th percentile
            features.append(np.percentile(sensor_data, 75))   # 75th percentile
            
            # Interquartile range
            iqr = np.percentile(sensor_data, 75) - np.percentile(sensor_data, 25)
            features.append(iqr)
            
        return np.array(features)
    
    def extract_temporal_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract temporal features from a single window.
        
        Args:
            window: Array of shape (window_size, n_sensors)
            
        Returns:
            1D array of temporal features
        """
        features = []
        
        # Ensure window is 2D
        if len(window.shape) == 1:
            window = window.reshape(1, -1)
        
        # For each sensor
        for sensor_idx in range(window.shape[1]):
            sensor_data = window[:, sensor_idx]
            
            # Rate of change (first derivative)
            diff = np.diff(sensor_data)
            features.append(np.mean(diff))      # Average rate of change
            features.append(np.std(diff))       # Variability in rate of change
            
            # Trend (linear slope)
            time_steps = np.arange(len(sensor_data))
            if len(sensor_data) > 1:
                slope, _ = np.polyfit(time_steps, sensor_data, 1)
                features.append(slope)
            else:
                features.append(0)
            
            # Acceleration (second derivative)
            if len(diff) > 1:
                acc = np.diff(diff)
                features.append(np.mean(acc))   # Average acceleration
            else:
                features.append(0)
            
            # Change magnitude (first to last)
            features.append(sensor_data[-1] - sensor_data[0])
            
        return np.array(features)
    
    def extract_domain_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract domain-specific features for engine health.
        
        Assumes sensor order:
        0: Engine rpm
        1: Lub oil pressure
        2: Fuel pressure
        3: Coolant pressure
        4: Lub oil temp
        5: Coolant temp
        
        Args:
            window: Array of shape (window_size, n_sensors)
            
        Returns:
            1D array of domain features
        """
        features = []
        
        # Ensure window is 2D
        if len(window.shape) == 1:
            window = window.reshape(1, -1)
        
        # Average values for ratios
        avg_rpm = np.mean(window[:, 0])
        avg_oil_press = np.mean(window[:, 1])
        avg_fuel_press = np.mean(window[:, 2])
        avg_coolant_press = np.mean(window[:, 3])
        avg_oil_temp = np.mean(window[:, 4])
        avg_coolant_temp = np.mean(window[:, 5])
        
        # Pressure ratios (important for engine health)
        features.append(avg_oil_press / avg_fuel_press if avg_fuel_press != 0 else 0)
        features.append(avg_coolant_press / avg_fuel_press if avg_fuel_press != 0 else 0)
        features.append(avg_oil_press / avg_coolant_press if avg_coolant_press != 0 else 0)
        
        # Temperature difference (cooling efficiency)
        features.append(avg_oil_temp - avg_coolant_temp)
        
        # Temperature-to-RPM ratio (thermal efficiency indicator)
        features.append(avg_oil_temp / avg_rpm if avg_rpm != 0 else 0)
        features.append(avg_coolant_temp / avg_rpm if avg_rpm != 0 else 0)
        
        # Pressure-to-RPM ratio (load indicator)
        features.append(avg_oil_press / avg_rpm if avg_rpm != 0 else 0)
        features.append(avg_fuel_press / avg_rpm if avg_rpm != 0 else 0)
        
        # Temperature stability (std of temperature)
        features.append(np.std(window[:, 4]))  # Oil temp stability
        features.append(np.std(window[:, 5]))  # Coolant temp stability
        
        # Pressure stability
        features.append(np.std(window[:, 1]))  # Oil pressure stability
        features.append(np.std(window[:, 2]))  # Fuel pressure stability
        
        # Combined efficiency score (custom metric)
        # Higher pressure + lower temp variation = healthier
        pressure_score = (avg_oil_press + avg_fuel_press + avg_coolant_press) / 3
        temp_variation = (np.std(window[:, 4]) + np.std(window[:, 5])) / 2
        efficiency = pressure_score / (temp_variation + 1e-6)  # Avoid division by zero
        features.append(efficiency)
        
        return np.array(features)
    
    def extract_all_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract all features from a single window.
        
        Args:
            window: Array of shape (window_size, n_sensors)
            
        Returns:
            1D array of all engineered features
        """
        stat_features = self.extract_statistical_features(window)
        temp_features = self.extract_temporal_features(window)
        domain_features = self.extract_domain_features(window)
        
        return np.concatenate([stat_features, temp_features, domain_features])
    
    def transform(self, X_windows: np.ndarray) -> np.ndarray:
        """
        Transform windowed data into engineered features.
        
        Args:
            X_windows: Array of shape (n_samples, window_size, n_sensors)
            
        Returns:
            Array of shape (n_samples, n_features) with engineered features
        """
        n_samples = X_windows.shape[0]
        
        # Extract features for first window to get dimensionality
        first_features = self.extract_all_features(X_windows[0])
        n_features = len(first_features)
        
        # Initialize output array
        X_engineered = np.zeros((n_samples, n_features))
        
        # Extract features for all windows
        for i in range(n_samples):
            X_engineered[i] = self.extract_all_features(X_windows[i])
        
        return X_engineered
    
    def get_feature_names(self, sensor_names: List[str] = None) -> List[str]:
        """
        Get names of all engineered features.
        
        Args:
            sensor_names: Optional list of sensor names
            
        Returns:
            List of feature names
        """
        if sensor_names is None:
            sensor_names = ['rpm', 'oil_press', 'fuel_press', 
                          'coolant_press', 'oil_temp', 'coolant_temp']
        
        feature_names = []
        
        # Statistical features (10 per sensor)
        stat_features = ['mean', 'std', 'min', 'max', 'median', 
                        'range', 'cv', 'p25', 'p75', 'iqr']
        for sensor in sensor_names:
            for stat in stat_features:
                feature_names.append(f'{sensor}_{stat}')
        
        # Temporal features (5 per sensor)
        temp_features = ['rate_mean', 'rate_std', 'trend', 'accel', 'change']
        for sensor in sensor_names:
            for temp in temp_features:
                feature_names.append(f'{sensor}_{temp}')
        
        # Domain features (13 total)
        domain_features = [
            'oil_fuel_press_ratio', 'coolant_fuel_press_ratio', 
            'oil_coolant_press_ratio', 'temp_diff',
            'oil_temp_rpm_ratio', 'coolant_temp_rpm_ratio',
            'oil_press_rpm_ratio', 'fuel_press_rpm_ratio',
            'oil_temp_stability', 'coolant_temp_stability',
            'oil_press_stability', 'fuel_press_stability',
            'efficiency_score'
        ]
        feature_names.extend(domain_features)
        
        return feature_names


def engineer_features_for_xgboost(X_windows: np.ndarray, 
                                   window_size: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to engineer features for XGBoost.
    
    Args:
        X_windows: Windowed data of shape (n_samples, window_size, n_sensors)
        window_size: Size of sliding window
        
    Returns:
        Tuple of (engineered features, feature names)
    """
    engineer = EngineFeatureEngineer(window_size=window_size)
    X_engineered = engineer.transform(X_windows)
    feature_names = engineer.get_feature_names()
    
    print(f"Feature engineering complete:")
    print(f"  Input shape: {X_windows.shape}")
    print(f"  Output shape: {X_engineered.shape}")
    print(f"  Total features: {len(feature_names)}")
    print(f"    Statistical: 60 (10 per sensor × 6 sensors)")
    print(f"    Temporal: 30 (5 per sensor × 6 sensors)")
    print(f"    Domain-specific: 13")
    
    return X_engineered, feature_names


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering...")
    
    # Create sample window (10 timesteps, 6 sensors)
    sample_window = np.random.randn(10, 6)
    
    engineer = EngineFeatureEngineer(window_size=10)
    features = engineer.extract_all_features(sample_window)
    
    print(f"Sample window shape: {sample_window.shape}")
    print(f"Extracted features: {len(features)}")
    
    # Test batch transformation
    sample_windows = np.random.randn(100, 10, 6)
    X_engineered, feature_names = engineer_features_for_xgboost(sample_windows)
    
    print(f"\nFeature names ({len(feature_names)}):")
    for i, name in enumerate(feature_names[:10]):
        print(f"  {i+1}. {name}")
    print(f"  ...")
    print(f"  {len(feature_names)}. {feature_names[-1]}")
