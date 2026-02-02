"""
Unit tests for model modules.
"""

import pytest
import torch
import numpy as np
from src.models.cnn_model import EngineCNN
from src.models.xgboost_model import EngineXGBoost


def test_cnn_initialization():
    """Test CNN model initialization."""
    model = EngineCNN(input_channels=14, window_size=50, hidden_channels=64)
    
    assert model.input_channels == 14
    assert model.window_size == 50
    assert model.hidden_channels == 64
    assert model.num_classes == 2


def test_cnn_forward_pass():
    """Test CNN forward pass."""
    model = EngineCNN(input_channels=14, window_size=50, hidden_channels=64)
    
    # Create dummy input (batch_size=2, window_size=50, features=14)
    x = torch.randn(2, 50, 14)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (2, 2)  # (batch_size, num_classes)


def test_cnn_predict_proba():
    """Test CNN probability prediction."""
    model = EngineCNN(input_channels=14, window_size=50, hidden_channels=64)
    
    # Create dummy input
    x = torch.randn(2, 50, 14)
    
    # Get probabilities
    probs = model.predict_proba(x)
    
    # Check shape and values
    assert probs.shape == (2, 2)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2))  # Probabilities sum to 1


def test_xgboost_initialization():
    """Test XGBoost model initialization."""
    model = EngineXGBoost(max_depth=6, n_estimators=100, learning_rate=0.1)
    
    assert model.max_depth == 6
    assert model.n_estimators == 100
    assert model.learning_rate == 0.1


def test_xgboost_fit_predict():
    """Test XGBoost training and prediction."""
    model = EngineXGBoost(max_depth=3, n_estimators=10, learning_rate=0.1)
    
    # Create dummy data
    X_train = np.random.randn(100, 14)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(20, 14)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    # Check outputs
    assert predictions.shape == (20,)
    assert probs.shape == (20, 2)
    assert set(predictions).issubset({0, 1})
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_xgboost_get_params():
    """Test getting XGBoost parameters."""
    model = EngineXGBoost(max_depth=5, n_estimators=50, learning_rate=0.05)
    params = model.get_params()
    
    assert params['max_depth'] == 5
    assert params['n_estimators'] == 50
    assert params['learning_rate'] == 0.05


class TestCNNInputValidation:
    """Test CNN model input validation."""
    
    def test_cnn_invalid_shape_2d(self):
        """Test CNN rejects 2D input (needs 3D)."""
        model = EngineCNN(input_channels=14, window_size=50, hidden_channels=64)
        
        # 2D input should fail
        x = torch.randn(2, 14)
        
        with pytest.raises(RuntimeError):
            model(x)
    
    def test_cnn_invalid_shape_4d(self):
        """Test CNN rejects 4D input (needs 3D)."""
        model = EngineCNN(input_channels=14, window_size=50, hidden_channels=64)
        
        # 4D input should fail
        x = torch.randn(2, 50, 14, 1)
        
        with pytest.raises(RuntimeError):
            model(x)
    
    def test_cnn_wrong_feature_count(self):
        """Test CNN with wrong number of features."""
        model = EngineCNN(input_channels=14, window_size=50, hidden_channels=64)
        
        # Wrong feature count (10 instead of 14)
        x = torch.randn(2, 50, 10)
        
        with pytest.raises(RuntimeError):
            model(x)
    
    def test_cnn_wrong_window_size(self):
        """Test CNN with wrong window size."""
        model = EngineCNN(input_channels=14, window_size=50, hidden_channels=64)
        
        # Wrong window size (30 instead of 50)
        x = torch.randn(2, 30, 14)
        
        # Should work but may give unexpected results
        # PyTorch is flexible with sequence length in Conv1d
        output = model(x)
        assert output.shape[0] == 2  # batch size preserved
    
    def test_cnn_empty_batch(self):
        """Test CNN with empty batch."""
        model = EngineCNN(input_channels=14, window_size=50, hidden_channels=64)
        
        # Empty batch
        x = torch.randn(0, 50, 14)
        
        output = model(x)
        assert output.shape == (0, 2)
    
    def test_cnn_single_sample(self):
        """Test CNN with single sample."""
        model = EngineCNN(input_channels=14, window_size=50, hidden_channels=64)
        
        # Single sample
        x = torch.randn(1, 50, 14)
        
        output = model(x)
        assert output.shape == (1, 2)
    
    def test_cnn_nan_input(self):
        """Test CNN behavior with NaN input."""
        model = EngineCNN(input_channels=14, window_size=50, hidden_channels=64)
        
        # Input with NaN
        x = torch.randn(2, 50, 14)
        x[0, 0, 0] = float('nan')
        
        output = model(x)
        # NaN propagates through network
        assert torch.isnan(output).any()
    
    def test_cnn_inf_input(self):
        """Test CNN behavior with Inf input."""
        model = EngineCNN(input_channels=14, window_size=50, hidden_channels=64)
        
        # Input with Inf
        x = torch.randn(2, 50, 14)
        x[0, 0, 0] = float('inf')
        
        output = model(x)
        # Inf may propagate or saturate
        assert output.shape == (2, 2)


class TestXGBoostInputValidation:
    """Test XGBoost model input validation."""
    
    def test_xgboost_valid_input(self):
        """Test XGBoost with valid 2D input."""
        model = EngineXGBoost(max_depth=3, n_estimators=10)
        
        X_train = np.random.randn(100, 14)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 14)
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (20,)
    
    def test_xgboost_wrong_features_predict(self):
        """Test XGBoost with wrong feature count at prediction."""
        model = EngineXGBoost(max_depth=3, n_estimators=10)
        
        X_train = np.random.randn(100, 14)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)
        
        # Wrong feature count at prediction
        X_test = np.random.randn(20, 10)
        
        with pytest.raises(ValueError):
            model.predict(X_test)
    
    def test_xgboost_1d_input(self):
        """Test XGBoost with 1D input (should fail)."""
        model = EngineXGBoost(max_depth=3, n_estimators=10)
        
        X_train = np.random.randn(100, 14)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)
        
        # 1D input
        X_test = np.random.randn(20)
        
        with pytest.raises((ValueError, IndexError)):
            model.predict(X_test)
    
    def test_xgboost_3d_input(self):
        """Test XGBoost with 3D input (should fail)."""
        model = EngineXGBoost(max_depth=3, n_estimators=10)
        
        X_train = np.random.randn(100, 14)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)
        
        # 3D input
        X_test = np.random.randn(20, 14, 1)
        
        with pytest.raises(ValueError):
            model.predict(X_test)
    
    def test_xgboost_empty_input(self):
        """Test XGBoost with empty input."""
        model = EngineXGBoost(max_depth=3, n_estimators=10)
        
        X_train = np.random.randn(100, 14)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)
        
        # Empty input
        X_test = np.random.randn(0, 14)
        
        predictions = model.predict(X_test)
        assert predictions.shape == (0,)
    
    def test_xgboost_single_sample(self):
        """Test XGBoost with single sample."""
        model = EngineXGBoost(max_depth=3, n_estimators=10)
        
        X_train = np.random.randn(100, 14)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)
        
        # Single sample (needs to be 2D)
        X_test = np.random.randn(1, 14)
        
        predictions = model.predict(X_test)
        assert predictions.shape == (1,)
    
    def test_xgboost_nan_input(self):
        """Test XGBoost handles NaN in input."""
        model = EngineXGBoost(max_depth=3, n_estimators=10)
        
        X_train = np.random.randn(100, 14)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)
        
        # Input with NaN
        X_test = np.random.randn(20, 14)
        X_test[0, 0] = np.nan
        
        # XGBoost can handle NaN values
        predictions = model.predict(X_test)
        assert predictions.shape == (20,)
    
    def test_xgboost_inf_input(self):
        """Test XGBoost behavior with Inf input."""
        model = EngineXGBoost(max_depth=3, n_estimators=10)
        
        X_train = np.random.randn(100, 14)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)
        
        # Input with Inf
        X_test = np.random.randn(20, 14)
        X_test[0, 0] = np.inf
        
        # XGBoost may handle or reject Inf
        # This test documents current behavior
        try:
            predictions = model.predict(X_test)
            assert predictions.shape == (20,)
        except (ValueError, FloatingPointError):
            # It's okay to reject Inf values
            pass


class TestModelOutputValidation:
    """Test model output validation."""
    
    def test_cnn_output_probabilities_sum_to_one(self):
        """Test CNN probabilities sum to 1."""
        model = EngineCNN(input_channels=14, window_size=50)
        
        x = torch.randn(10, 50, 14)
        probs = model.predict_proba(x)
        
        # Check probabilities sum to 1 for each sample
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(10), atol=1e-5)
    
    def test_cnn_output_in_valid_range(self):
        """Test CNN outputs are in [0, 1]."""
        model = EngineCNN(input_channels=14, window_size=50)
        
        x = torch.randn(10, 50, 14)
        probs = model.predict_proba(x)
        
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)
    
    def test_xgboost_output_probabilities_sum_to_one(self):
        """Test XGBoost probabilities sum to 1."""
        model = EngineXGBoost(max_depth=3, n_estimators=10)
        
        X_train = np.random.randn(100, 14)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)
        
        X_test = np.random.randn(10, 14)
        probs = model.predict_proba(X_test)
        
        # Check probabilities sum to 1 for each sample
        prob_sums = probs.sum(axis=1)
        np.testing.assert_allclose(prob_sums, np.ones(10), atol=1e-5)
    
    def test_xgboost_output_in_valid_range(self):
        """Test XGBoost outputs are in [0, 1]."""
        model = EngineXGBoost(max_depth=3, n_estimators=10)
        
        X_train = np.random.randn(100, 14)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)
        
        X_test = np.random.randn(10, 14)
        probs = model.predict_proba(X_test)
        
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
    
    def test_xgboost_predictions_are_binary(self):
        """Test XGBoost predictions are 0 or 1."""
        model = EngineXGBoost(max_depth=3, n_estimators=10)
        
        X_train = np.random.randn(100, 14)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)
        
        X_test = np.random.randn(10, 14)
        predictions = model.predict(X_test)
        
        assert set(predictions).issubset({0, 1})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
