"""
Unit tests for data preprocessing module.
"""

import pytest
import numpy as np
from src.data.preprocessing import EngineDataPreprocessor


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = EngineDataPreprocessor(window_size=50, test_size=0.2)
    assert preprocessor.window_size == 50
    assert preprocessor.test_size == 0.2
    assert preprocessor.random_state == 42


def test_generate_synthetic_data():
    """Test synthetic data generation."""
    preprocessor = EngineDataPreprocessor()
    X, y = preprocessor.generate_synthetic_data(n_samples=100, n_features=14)
    
    assert X.shape == (100, 14)
    assert y.shape == (100,)
    assert set(y) == {0, 1}  # Binary classification


def test_standardize():
    """Test data standardization."""
    preprocessor = EngineDataPreprocessor()
    X_train = np.random.randn(100, 14)
    X_test = np.random.randn(20, 14)
    
    X_train_scaled, X_test_scaled = preprocessor.standardize(X_train, X_test)
    
    # Check shapes
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    
    # Check that training data is approximately standardized
    assert np.abs(X_train_scaled.mean()) < 0.1
    assert np.abs(X_train_scaled.std() - 1.0) < 0.1


def test_create_sliding_windows():
    """Test sliding window creation."""
    preprocessor = EngineDataPreprocessor(window_size=10)
    X = np.random.randn(50, 14)
    y = np.random.randint(0, 2, 50)
    
    X_windows, y_windows = preprocessor.create_sliding_windows(X, y)
    
    # Check shapes
    expected_windows = 50 - 10 + 1  # 41 windows
    assert X_windows.shape == (expected_windows, 10, 14)
    assert y_windows.shape == (expected_windows,)


def test_prepare_data():
    """Test complete data preparation."""
    preprocessor = EngineDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(use_synthetic=True)
    
    # Check that data is returned
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    
    # Check feature dimensions
    assert X_train.shape[1] == 14
    assert X_test.shape[1] == 14


def test_prepare_data_for_cnn():
    """Test CNN data preparation with windows."""
    preprocessor = EngineDataPreprocessor(window_size=50)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data_for_cnn(use_synthetic=True)
    
    # Check shapes
    assert len(X_train.shape) == 3  # (samples, window_size, features)
    assert len(X_test.shape) == 3
    assert X_train.shape[1] == 50  # window size
    assert X_train.shape[2] == 14  # features
    
    # Check labels
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
