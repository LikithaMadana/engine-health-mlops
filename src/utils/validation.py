"""
Input validation utilities for the Engine Health MLOps pipeline.

Provides functions to validate data shapes, types, and ranges
to ensure data integrity throughout the pipeline.
"""

import numpy as np
from typing import Any, Optional, Union, List, Tuple
from .exceptions import DataValidationError
from .logger import setup_logger

logger = setup_logger(__name__)


def validate_array_shape(
    data: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    name: str = "data"
) -> None:
    """
    Validate numpy array shape.
    
    Args:
        data: Input array to validate
        expected_shape: Expected shape (use None for flexible dimensions)
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        name: Name of the data for error messages
        
    Raises:
        DataValidationError: If validation fails
        
    Example:
        >>> validate_array_shape(X, expected_shape=(None, 14), name="features")
        >>> validate_array_shape(y, min_dims=1, max_dims=2, name="labels")
    """
    if not isinstance(data, np.ndarray):
        raise DataValidationError(
            f"{name} must be a numpy array, got {type(data).__name__}"
        )
    
    # Check dimensions
    if min_dims is not None and data.ndim < min_dims:
        raise DataValidationError(
            f"{name} must have at least {min_dims} dimensions, got {data.ndim}"
        )
    
    if max_dims is not None and data.ndim > max_dims:
        raise DataValidationError(
            f"{name} must have at most {max_dims} dimensions, got {data.ndim}"
        )
    
    # Check specific shape
    if expected_shape is not None:
        if len(expected_shape) != data.ndim:
            raise DataValidationError(
                f"{name} shape mismatch: expected {len(expected_shape)} dimensions, "
                f"got {data.ndim}"
            )
        
        for i, (expected, actual) in enumerate(zip(expected_shape, data.shape)):
            if expected is not None and expected != actual:
                raise DataValidationError(
                    f"{name} shape mismatch at dimension {i}: "
                    f"expected {expected}, got {actual}"
                )
    
    logger.debug(f"Validated {name} shape: {data.shape}")


def validate_numeric_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    name: str = "value"
) -> None:
    """
    Validate numeric value is within range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the value for error messages
        
    Raises:
        DataValidationError: If validation fails
        
    Example:
        >>> validate_numeric_range(0.8, min_val=0.0, max_val=1.0, name="test_size")
    """
    if not isinstance(value, (int, float)):
        raise DataValidationError(
            f"{name} must be numeric, got {type(value).__name__}"
        )
    
    if min_val is not None and value < min_val:
        raise DataValidationError(
            f"{name} must be >= {min_val}, got {value}"
        )
    
    if max_val is not None and value > max_val:
        raise DataValidationError(
            f"{name} must be <= {max_val}, got {value}"
        )
    
    logger.debug(f"Validated {name}: {value}")


def validate_positive_int(value: Any, name: str = "value") -> None:
    """
    Validate value is a positive integer.
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Raises:
        DataValidationError: If validation fails
        
    Example:
        >>> validate_positive_int(50, name="window_size")
    """
    if not isinstance(value, int):
        raise DataValidationError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    
    if value <= 0:
        raise DataValidationError(
            f"{name} must be positive, got {value}"
        )
    
    logger.debug(f"Validated {name}: {value}")


def validate_feature_count(
    data: np.ndarray,
    expected_features: int,
    name: str = "data"
) -> None:
    """
    Validate number of features in data.
    
    Args:
        data: Input data array
        expected_features: Expected number of features
        name: Name of the data for error messages
        
    Raises:
        DataValidationError: If validation fails
        
    Example:
        >>> validate_feature_count(X, expected_features=14, name="sensor_data")
    """
    if data.ndim < 2:
        raise DataValidationError(
            f"{name} must be at least 2D to check features, got {data.ndim}D"
        )
    
    actual_features = data.shape[-1]
    if actual_features != expected_features:
        raise DataValidationError(
            f"{name} feature count mismatch: expected {expected_features}, "
            f"got {actual_features}"
        )
    
    logger.debug(f"Validated {name} features: {actual_features}")


def validate_labels(
    labels: np.ndarray,
    num_classes: int = 2,
    name: str = "labels"
) -> None:
    """
    Validate classification labels.
    
    Args:
        labels: Label array
        num_classes: Expected number of classes
        name: Name of the labels for error messages
        
    Raises:
        DataValidationError: If validation fails
        
    Example:
        >>> validate_labels(y, num_classes=2, name="health_labels")
    """
    if not isinstance(labels, np.ndarray):
        raise DataValidationError(
            f"{name} must be a numpy array, got {type(labels).__name__}"
        )
    
    if labels.ndim != 1:
        raise DataValidationError(
            f"{name} must be 1D, got {labels.ndim}D"
        )
    
    unique_labels = np.unique(labels)
    if len(unique_labels) > num_classes:
        raise DataValidationError(
            f"{name} has {len(unique_labels)} unique values, "
            f"expected at most {num_classes}"
        )
    
    # Check labels are in valid range
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise DataValidationError(
            f"{name} must be in range [0, {num_classes}), "
            f"got values in range [{labels.min()}, {labels.max()}]"
        )
    
    logger.debug(f"Validated {name}: {len(labels)} samples, {len(unique_labels)} classes")


def validate_probability(
    prob: Union[float, np.ndarray],
    name: str = "probability"
) -> None:
    """
    Validate probability values are in [0, 1].
    
    Args:
        prob: Probability value or array
        name: Name for error messages
        
    Raises:
        DataValidationError: If validation fails
        
    Example:
        >>> validate_probability(0.85, name="confidence")
    """
    if isinstance(prob, np.ndarray):
        if np.any(prob < 0) or np.any(prob > 1):
            raise DataValidationError(
                f"{name} must be in range [0, 1], "
                f"got values in range [{prob.min()}, {prob.max()}]"
            )
    else:
        if prob < 0 or prob > 1:
            raise DataValidationError(
                f"{name} must be in range [0, 1], got {prob}"
            )
    
    logger.debug(f"Validated {name}")
