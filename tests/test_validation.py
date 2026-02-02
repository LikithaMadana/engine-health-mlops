"""
Unit tests for validation utilities.

Tests all validation functions to ensure proper data integrity checks.
"""

import pytest
import numpy as np
from src.utils.validation import (
    validate_array_shape,
    validate_numeric_range,
    validate_positive_int,
    validate_feature_count,
    validate_labels,
    validate_probability
)
from src.utils.exceptions import DataValidationError


class TestValidateArrayShape:
    """Test array shape validation."""
    
    def test_valid_array_shape(self):
        """Test validation passes for valid array."""
        data = np.random.randn(100, 14)
        validate_array_shape(data, expected_shape=(100, 14))
        # Should not raise exception
    
    def test_flexible_dimension(self):
        """Test validation with flexible dimensions."""
        data = np.random.randn(100, 14)
        validate_array_shape(data, expected_shape=(None, 14))
        # Should not raise exception
    
    def test_min_dims(self):
        """Test minimum dimensions validation."""
        data = np.random.randn(100, 14)
        validate_array_shape(data, min_dims=2)
        
        with pytest.raises(DataValidationError, match="must have at least"):
            data_1d = np.random.randn(100)
            validate_array_shape(data_1d, min_dims=2)
    
    def test_max_dims(self):
        """Test maximum dimensions validation."""
        data = np.random.randn(10, 20)
        validate_array_shape(data, max_dims=2)
        
        with pytest.raises(DataValidationError, match="must have at most"):
            data_3d = np.random.randn(10, 20, 30)
            validate_array_shape(data_3d, max_dims=2)
    
    def test_wrong_type(self):
        """Test validation fails for non-array."""
        with pytest.raises(DataValidationError, match="must be a numpy array"):
            validate_array_shape([1, 2, 3], expected_shape=(3,))
    
    def test_shape_mismatch(self):
        """Test validation fails for shape mismatch."""
        data = np.random.randn(100, 14)
        
        with pytest.raises(DataValidationError, match="shape mismatch"):
            validate_array_shape(data, expected_shape=(100, 20))


class TestValidateNumericRange:
    """Test numeric range validation."""
    
    def test_valid_range(self):
        """Test validation passes for value in range."""
        validate_numeric_range(0.5, min_val=0.0, max_val=1.0)
        validate_numeric_range(50, min_val=1, max_val=100)
    
    def test_min_value_violation(self):
        """Test validation fails when below minimum."""
        with pytest.raises(DataValidationError, match="must be >="):
            validate_numeric_range(-1, min_val=0)
    
    def test_max_value_violation(self):
        """Test validation fails when above maximum."""
        with pytest.raises(DataValidationError, match="must be <="):
            validate_numeric_range(101, max_val=100)
    
    def test_non_numeric(self):
        """Test validation fails for non-numeric value."""
        with pytest.raises(DataValidationError, match="must be numeric"):
            validate_numeric_range("50", min_val=0, max_val=100)


class TestValidatePositiveInt:
    """Test positive integer validation."""
    
    def test_valid_positive_int(self):
        """Test validation passes for positive integer."""
        validate_positive_int(50)
        validate_positive_int(1)
    
    def test_zero(self):
        """Test validation fails for zero."""
        with pytest.raises(DataValidationError, match="must be positive"):
            validate_positive_int(0)
    
    def test_negative(self):
        """Test validation fails for negative."""
        with pytest.raises(DataValidationError, match="must be positive"):
            validate_positive_int(-5)
    
    def test_float(self):
        """Test validation fails for float."""
        with pytest.raises(DataValidationError, match="must be an integer"):
            validate_positive_int(50.5)
    
    def test_string(self):
        """Test validation fails for string."""
        with pytest.raises(DataValidationError, match="must be an integer"):
            validate_positive_int("50")


class TestValidateFeatureCount:
    """Test feature count validation."""
    
    def test_valid_feature_count(self):
        """Test validation passes for correct features."""
        data = np.random.randn(100, 14)
        validate_feature_count(data, expected_features=14)
    
    def test_feature_mismatch(self):
        """Test validation fails for wrong feature count."""
        data = np.random.randn(100, 10)
        
        with pytest.raises(DataValidationError, match="feature count mismatch"):
            validate_feature_count(data, expected_features=14)
    
    def test_1d_array(self):
        """Test validation fails for 1D array."""
        data = np.random.randn(100)
        
        with pytest.raises(DataValidationError, match="must be at least 2D"):
            validate_feature_count(data, expected_features=14)
    
    def test_3d_array(self):
        """Test validation works for 3D array (uses last dimension)."""
        data = np.random.randn(10, 50, 14)
        validate_feature_count(data, expected_features=14)


class TestValidateLabels:
    """Test label validation."""
    
    def test_valid_binary_labels(self):
        """Test validation passes for valid binary labels."""
        labels = np.array([0, 1, 0, 1, 1, 0])
        validate_labels(labels, num_classes=2)
    
    def test_valid_multiclass_labels(self):
        """Test validation passes for valid multiclass labels."""
        labels = np.array([0, 1, 2, 1, 2, 0])
        validate_labels(labels, num_classes=3)
    
    def test_too_many_classes(self):
        """Test validation fails when too many unique classes."""
        labels = np.array([0, 1, 2, 3])
        
        with pytest.raises(DataValidationError, match="unique values"):
            validate_labels(labels, num_classes=2)
    
    def test_out_of_range_labels(self):
        """Test validation fails for labels outside valid range."""
        labels = np.array([0, 1, 2])
        
        with pytest.raises(DataValidationError, match="must be in range"):
            validate_labels(labels, num_classes=2)
    
    def test_negative_labels(self):
        """Test validation fails for negative labels."""
        labels = np.array([-1, 0, 1])
        
        with pytest.raises(DataValidationError, match="must be in range"):
            validate_labels(labels, num_classes=2)
    
    def test_non_1d_labels(self):
        """Test validation fails for non-1D labels."""
        labels = np.array([[0, 1], [1, 0]])
        
        with pytest.raises(DataValidationError, match="must be 1D"):
            validate_labels(labels, num_classes=2)
    
    def test_non_array_labels(self):
        """Test validation fails for non-array labels."""
        with pytest.raises(DataValidationError, match="must be a numpy array"):
            validate_labels([0, 1, 0], num_classes=2)


class TestValidateProbability:
    """Test probability validation."""
    
    def test_valid_scalar_probability(self):
        """Test validation passes for valid scalar probability."""
        validate_probability(0.5)
        validate_probability(0.0)
        validate_probability(1.0)
    
    def test_valid_array_probability(self):
        """Test validation passes for valid probability array."""
        probs = np.array([0.1, 0.5, 0.9, 0.0, 1.0])
        validate_probability(probs)
    
    def test_invalid_scalar_below_zero(self):
        """Test validation fails for probability < 0."""
        with pytest.raises(DataValidationError, match="must be in range"):
            validate_probability(-0.1)
    
    def test_invalid_scalar_above_one(self):
        """Test validation fails for probability > 1."""
        with pytest.raises(DataValidationError, match="must be in range"):
            validate_probability(1.5)
    
    def test_invalid_array_below_zero(self):
        """Test validation fails for array with values < 0."""
        probs = np.array([0.5, -0.1, 0.9])
        
        with pytest.raises(DataValidationError, match="must be in range"):
            validate_probability(probs)
    
    def test_invalid_array_above_one(self):
        """Test validation fails for array with values > 1."""
        probs = np.array([0.5, 0.9, 1.5])
        
        with pytest.raises(DataValidationError, match="must be in range"):
            validate_probability(probs)


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_array(self):
        """Test validation handles empty arrays."""
        data = np.array([])
        
        # Should work with flexible shape
        validate_array_shape(data, min_dims=1)
    
    def test_very_large_array(self):
        """Test validation works with large arrays."""
        data = np.random.randn(10000, 14)
        validate_array_shape(data, expected_shape=(None, 14))
        validate_feature_count(data, expected_features=14)
    
    def test_single_sample(self):
        """Test validation works with single sample."""
        data = np.random.randn(1, 14)
        validate_array_shape(data, expected_shape=(None, 14))
        validate_feature_count(data, expected_features=14)
    
    def test_custom_error_messages(self):
        """Test that custom names appear in error messages."""
        data = np.random.randn(100, 10)
        
        try:
            validate_feature_count(data, expected_features=14, name="sensor_data")
        except DataValidationError as e:
            assert "sensor_data" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
