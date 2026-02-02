"""
Custom exceptions for the Engine Health MLOps pipeline.

This module defines domain-specific exceptions to provide clear,
actionable error messages throughout the application.
"""


class EngineHealthException(Exception):
    """Base exception for all engine health prediction errors."""
    pass


class DataValidationError(EngineHealthException):
    """Raised when data validation fails."""
    pass


class ModelLoadError(EngineHealthException):
    """Raised when model loading fails."""
    pass


class ModelTrainingError(EngineHealthException):
    """Raised when model training encounters an error."""
    pass


class PredictionError(EngineHealthException):
    """Raised when prediction fails."""
    pass


class RegistryError(EngineHealthException):
    """Raised when MLflow registry operations fail."""
    pass


class ConfigurationError(EngineHealthException):
    """Raised when configuration is invalid."""
    pass


class DataPreprocessingError(EngineHealthException):
    """Raised when data preprocessing fails."""
    pass
