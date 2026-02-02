"""
Configuration management for the Engine Health MLOps pipeline.

Handles environment variables, secrets, and configuration validation
following MLOps best practices.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
import json
import logging

from .exceptions import ConfigurationError
from .logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class DataConfig:
    """Data-related configuration."""
    window_size: int = 50
    test_size: float = 0.2
    random_state: int = 42
    n_samples: int = 1000
    n_features: int = 14
    
    def validate(self):
        """Validate data configuration."""
        if self.window_size <= 0:
            raise ConfigurationError(f"window_size must be positive, got {self.window_size}")
        if not 0 < self.test_size < 1:
            raise ConfigurationError(f"test_size must be in (0, 1), got {self.test_size}")
        if self.n_features <= 0:
            raise ConfigurationError(f"n_features must be positive, got {self.n_features}")
        logger.debug(f"DataConfig validated: {self}")


@dataclass
class ModelConfig:
    """Model-related configuration."""
    # CNN config
    cnn_hidden_channels: int = 64
    cnn_learning_rate: float = 0.001
    cnn_batch_size: int = 32
    cnn_num_epochs: int = 50
    
    # XGBoost config
    xgb_max_depth: int = 6
    xgb_n_estimators: int = 100
    xgb_learning_rate: float = 0.1
    
    # Paths
    cnn_model_path: str = "models/cnn_model.pth"
    xgb_model_path: str = "models/xgb_model.pkl"
    
    def validate(self):
        """Validate model configuration."""
        if self.cnn_learning_rate <= 0 or self.cnn_learning_rate >= 1:
            raise ConfigurationError(f"Invalid CNN learning rate: {self.cnn_learning_rate}")
        if self.xgb_learning_rate <= 0 or self.xgb_learning_rate >= 1:
            raise ConfigurationError(f"Invalid XGBoost learning rate: {self.xgb_learning_rate}")
        if self.cnn_batch_size <= 0:
            raise ConfigurationError(f"Batch size must be positive: {self.cnn_batch_size}")
        logger.debug(f"ModelConfig validated: {self}")


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "Engine Health Prediction Pipeline"
    artifact_location: Optional[str] = None
    registry_uri: Optional[str] = None
    
    # Authentication (from env vars)
    username: Optional[str] = None
    password: Optional[str] = None
    
    def validate(self):
        """Validate MLflow configuration."""
        if not self.experiment_name:
            raise ConfigurationError("experiment_name cannot be empty")
        
        # Validate URI format
        valid_schemes = ['file', 'http', 'https', 'postgresql', 'mysql', 'sqlite']
        if not any(self.tracking_uri.startswith(scheme) for scheme in valid_schemes):
            logger.warning(f"Unusual tracking_uri scheme: {self.tracking_uri}")
        
        logger.debug(f"MLflowConfig validated (credentials hidden)")


@dataclass
class APIConfig:
    """API service configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    max_batch_size: int = 1000
    
    def validate(self):
        """Validate API configuration."""
        if not 1 <= self.port <= 65535:
            raise ConfigurationError(f"Port must be in [1, 65535], got {self.port}")
        if self.workers < 1:
            raise ConfigurationError(f"Workers must be >= 1, got {self.workers}")
        if self.max_batch_size < 1:
            raise ConfigurationError(f"max_batch_size must be >= 1, got {self.max_batch_size}")
        
        valid_log_levels = ['debug', 'info', 'warning', 'error', 'critical']
        if self.log_level.lower() not in valid_log_levels:
            raise ConfigurationError(f"Invalid log_level: {self.log_level}")
        
        logger.debug(f"APIConfig validated: {self}")


@dataclass
class Config:
    """Main configuration class combining all configs."""
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # General settings
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'Config':
        """
        Create configuration from environment variables.
        
        Environment variables follow the pattern: PREFIX_SECTION_KEY
        Example: ENGINE_HEALTH_DATA_WINDOW_SIZE
        
        Returns:
            Config instance populated from environment
        """
        config = cls()
        
        # General settings
        config.environment = os.getenv('ENGINE_HEALTH_ENV', 'development')
        config.debug = os.getenv('ENGINE_HEALTH_DEBUG', 'false').lower() == 'true'
        config.log_level = os.getenv('ENGINE_HEALTH_LOG_LEVEL', 'INFO')
        
        # Data config
        config.data.window_size = int(os.getenv('ENGINE_HEALTH_DATA_WINDOW_SIZE', '50'))
        config.data.test_size = float(os.getenv('ENGINE_HEALTH_DATA_TEST_SIZE', '0.2'))
        config.data.random_state = int(os.getenv('ENGINE_HEALTH_DATA_RANDOM_STATE', '42'))
        config.data.n_samples = int(os.getenv('ENGINE_HEALTH_DATA_N_SAMPLES', '1000'))
        config.data.n_features = int(os.getenv('ENGINE_HEALTH_DATA_N_FEATURES', '14'))
        
        # Model config
        config.model.cnn_hidden_channels = int(os.getenv('ENGINE_HEALTH_CNN_HIDDEN_CHANNELS', '64'))
        config.model.cnn_learning_rate = float(os.getenv('ENGINE_HEALTH_CNN_LEARNING_RATE', '0.001'))
        config.model.cnn_batch_size = int(os.getenv('ENGINE_HEALTH_CNN_BATCH_SIZE', '32'))
        config.model.cnn_num_epochs = int(os.getenv('ENGINE_HEALTH_CNN_NUM_EPOCHS', '50'))
        
        config.model.xgb_max_depth = int(os.getenv('ENGINE_HEALTH_XGB_MAX_DEPTH', '6'))
        config.model.xgb_n_estimators = int(os.getenv('ENGINE_HEALTH_XGB_N_ESTIMATORS', '100'))
        config.model.xgb_learning_rate = float(os.getenv('ENGINE_HEALTH_XGB_LEARNING_RATE', '0.1'))
        
        config.model.cnn_model_path = os.getenv('ENGINE_HEALTH_CNN_MODEL_PATH', 'models/cnn_model.pth')
        config.model.xgb_model_path = os.getenv('ENGINE_HEALTH_XGB_MODEL_PATH', 'models/xgb_model.pkl')
        
        # MLflow config
        config.mlflow.tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
        config.mlflow.experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 
                                                   'Engine Health Prediction Pipeline')
        config.mlflow.artifact_location = os.getenv('MLFLOW_ARTIFACT_LOCATION')
        config.mlflow.registry_uri = os.getenv('MLFLOW_REGISTRY_URI')
        config.mlflow.username = os.getenv('MLFLOW_TRACKING_USERNAME')
        config.mlflow.password = os.getenv('MLFLOW_TRACKING_PASSWORD')
        
        # API config
        config.api.host = os.getenv('ENGINE_HEALTH_API_HOST', '0.0.0.0')
        config.api.port = int(os.getenv('ENGINE_HEALTH_API_PORT', '8000'))
        config.api.workers = int(os.getenv('ENGINE_HEALTH_API_WORKERS', '1'))
        config.api.reload = os.getenv('ENGINE_HEALTH_API_RELOAD', 'false').lower() == 'true'
        config.api.log_level = os.getenv('ENGINE_HEALTH_API_LOG_LEVEL', 'info')
        config.api.max_batch_size = int(os.getenv('ENGINE_HEALTH_API_MAX_BATCH_SIZE', '1000'))
        
        logger.info(f"Configuration loaded from environment: {config.environment}")
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Config instance
            
        Raises:
            ConfigurationError: If file cannot be loaded
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise ConfigurationError(f"Config file not found: {config_path}")
            
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            config = cls()
            
            # Load sections
            if 'data' in data:
                for key, value in data['data'].items():
                    setattr(config.data, key, value)
            
            if 'model' in data:
                for key, value in data['model'].items():
                    setattr(config.model, key, value)
            
            if 'mlflow' in data:
                for key, value in data['mlflow'].items():
                    setattr(config.mlflow, key, value)
            
            if 'api' in data:
                for key, value in data['api'].items():
                    setattr(config.api, key, value)
            
            # General settings
            config.environment = data.get('environment', 'development')
            config.debug = data.get('debug', False)
            config.log_level = data.get('log_level', 'INFO')
            
            logger.info(f"Configuration loaded from file: {config_path}")
            return config
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from file: {e}")
    
    def validate(self):
        """
        Validate all configuration sections.
        
        Raises:
            ConfigurationError: If any validation fails
        """
        try:
            self.data.validate()
            self.model.validate()
            self.mlflow.validate()
            self.api.validate()
            
            # Validate environment
            valid_environments = ['development', 'staging', 'production']
            if self.environment not in valid_environments:
                raise ConfigurationError(
                    f"Invalid environment '{self.environment}'. "
                    f"Must be one of {valid_environments}"
                )
            
            logger.info(f"Configuration validated successfully for environment: {self.environment}")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation (safe for logging, no secrets)
        """
        config_dict = {
            'environment': self.environment,
            'debug': self.debug,
            'log_level': self.log_level,
            'data': {
                'window_size': self.data.window_size,
                'test_size': self.data.test_size,
                'random_state': self.data.random_state,
                'n_samples': self.data.n_samples,
                'n_features': self.data.n_features,
            },
            'model': {
                'cnn_hidden_channels': self.model.cnn_hidden_channels,
                'cnn_learning_rate': self.model.cnn_learning_rate,
                'cnn_batch_size': self.model.cnn_batch_size,
                'cnn_num_epochs': self.model.cnn_num_epochs,
                'xgb_max_depth': self.model.xgb_max_depth,
                'xgb_n_estimators': self.model.xgb_n_estimators,
                'xgb_learning_rate': self.model.xgb_learning_rate,
                'cnn_model_path': self.model.cnn_model_path,
                'xgb_model_path': self.model.xgb_model_path,
            },
            'mlflow': {
                'tracking_uri': self.mlflow.tracking_uri,
                'experiment_name': self.mlflow.experiment_name,
                'artifact_location': self.mlflow.artifact_location,
                'registry_uri': self.mlflow.registry_uri,
                # Never include credentials
                'username': '***' if self.mlflow.username else None,
                'password': '***' if self.mlflow.password else None,
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'workers': self.api.workers,
                'reload': self.api.reload,
                'log_level': self.api.log_level,
                'max_batch_size': self.api.max_batch_size,
            }
        }
        return config_dict


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Loads configuration on first call, subsequent calls return cached instance.
    
    Returns:
        Global Config instance
    """
    global _config
    
    if _config is None:
        # Try to load from environment variables first
        _config = Config.from_env()
        _config.validate()
        
        logger.info("Global configuration initialized")
        logger.debug(f"Configuration: {_config.to_dict()}")
    
    return _config


def reset_config():
    """Reset global configuration (mainly for testing)."""
    global _config
    _config = None
    logger.debug("Global configuration reset")
