"""
Artifact versioning and metadata tracking utilities.

Provides comprehensive artifact versioning following MLOps best practices,
including git tracking, configuration snapshots, and model lineage.
"""

import subprocess
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import platform
import sys

from .logger import setup_logger
from .exceptions import ConfigurationError

logger = setup_logger(__name__)


def get_git_info() -> Dict[str, str]:
    """
    Get current git repository information.
    
    Returns:
        Dictionary with git commit hash, branch, and status
    """
    git_info = {
        'commit_hash': 'unknown',
        'commit_short': 'unknown',
        'branch': 'unknown',
        'is_dirty': False,
        'remote_url': 'unknown'
    }
    
    try:
        # Get commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        git_info['commit_hash'] = result.stdout.strip()
        git_info['commit_short'] = git_info['commit_hash'][:7]
        
        # Get branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        git_info['branch'] = result.stdout.strip()
        
        # Check if working directory is dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True
        )
        git_info['is_dirty'] = bool(result.stdout.strip())
        
        # Get remote URL
        result = subprocess.run(
            ['git', 'config', '--get', 'remote.origin.url'],
            capture_output=True,
            text=True,
            check=False  # May not have remote
        )
        if result.returncode == 0:
            git_info['remote_url'] = result.stdout.strip()
        
        logger.debug(f"Git info retrieved: {git_info}")
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to get git info: {e}")
    except FileNotFoundError:
        logger.warning("Git not found in system PATH")
    
    return git_info


def get_system_info() -> Dict[str, str]:
    """
    Get system information for reproducibility.
    
    Returns:
        Dictionary with system details
    """
    system_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'hostname': platform.node(),
    }
    
    logger.debug(f"System info: {system_info}")
    return system_info


def compute_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Compute hash of a file for integrity verification.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha256, etc.)
        
    Returns:
        Hex digest of file hash
    """
    try:
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        
        file_hash = hash_func.hexdigest()
        logger.debug(f"Computed {algorithm} hash for {file_path}: {file_hash[:16]}...")
        return file_hash
        
    except Exception as e:
        logger.error(f"Failed to compute hash for {file_path}: {e}")
        return 'unknown'


def create_artifact_metadata(
    artifact_name: str,
    artifact_type: str,
    config: Optional[Dict[str, Any]] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create comprehensive metadata for an artifact.
    
    Args:
        artifact_name: Name of the artifact
        artifact_type: Type (model, dataset, config, etc.)
        config: Configuration used to create artifact
        additional_info: Any additional metadata
        
    Returns:
        Complete artifact metadata dictionary
    """
    metadata = {
        'artifact_name': artifact_name,
        'artifact_type': artifact_type,
        'created_at': datetime.utcnow().isoformat(),
        'version': {
            'git': get_git_info(),
            'system': get_system_info(),
        }
    }
    
    if config:
        metadata['configuration'] = config
    
    if additional_info:
        metadata['additional_info'] = additional_info
    
    logger.info(f"Created metadata for {artifact_type} artifact: {artifact_name}")
    return metadata


def save_artifact_metadata(
    metadata: Dict[str, Any],
    output_path: str
):
    """
    Save artifact metadata to JSON file.
    
    Args:
        metadata: Metadata dictionary
        output_path: Path to save metadata file
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Artifact metadata saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save artifact metadata: {e}")
        raise ConfigurationError(f"Failed to save metadata: {e}")


def load_artifact_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Load artifact metadata from JSON file.
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        Metadata dictionary
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Artifact metadata loaded from: {metadata_path}")
        return metadata
        
    except FileNotFoundError:
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in metadata file: {e}")
        return {}


def create_model_lineage(
    model_name: str,
    dataset_version: str,
    config_version: str,
    parent_model: Optional[str] = None,
    training_metrics: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Create model lineage information for tracking.
    
    Args:
        model_name: Name of the model
        dataset_version: Version/hash of training dataset
        config_version: Version/hash of configuration
        parent_model: Optional parent model (for fine-tuning)
        training_metrics: Optional training metrics
        
    Returns:
        Model lineage dictionary
    """
    lineage = {
        'model_name': model_name,
        'created_at': datetime.utcnow().isoformat(),
        'dataset': {
            'version': dataset_version,
            'type': 'engine_health_synthetic'
        },
        'configuration': {
            'version': config_version
        },
        'code_version': get_git_info(),
        'system': get_system_info()
    }
    
    if parent_model:
        lineage['parent_model'] = parent_model
        lineage['training_type'] = 'fine_tuning'
    else:
        lineage['training_type'] = 'from_scratch'
    
    if training_metrics:
        lineage['metrics'] = training_metrics
    
    logger.info(f"Model lineage created for: {model_name}")
    return lineage


def version_dataset(data_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Create version identifier for a dataset.
    
    Args:
        data_path: Path to dataset file
        metadata: Optional dataset metadata
        
    Returns:
        Dataset version string (hash-based)
    """
    try:
        # Compute hash of dataset file
        data_hash = compute_file_hash(data_path)
        
        # Create version identifier
        version = f"data_{data_hash[:12]}"
        
        # Save metadata if provided
        if metadata:
            metadata['version'] = version
            metadata['file_hash'] = data_hash
            metadata['file_path'] = data_path
            
            metadata_path = str(Path(data_path).with_suffix('.metadata.json'))
            save_artifact_metadata(metadata, metadata_path)
        
        logger.info(f"Dataset versioned: {version}")
        return version
        
    except Exception as e:
        logger.error(f"Failed to version dataset: {e}")
        return 'unknown'


def version_configuration(config_dict: Dict[str, Any]) -> str:
    """
    Create version identifier for a configuration.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configuration version string (hash-based)
    """
    try:
        # Convert config to stable JSON string
        config_json = json.dumps(config_dict, sort_keys=True)
        
        # Compute hash
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        
        # Create version identifier
        version = f"config_{config_hash[:12]}"
        
        logger.info(f"Configuration versioned: {version}")
        return version
        
    except Exception as e:
        logger.error(f"Failed to version configuration: {e}")
        return 'unknown'


class ArtifactTracker:
    """
    Track artifacts throughout the ML pipeline.
    
    Provides methods to track datasets, models, and configurations
    with full lineage and versioning.
    """
    
    def __init__(self, tracking_dir: str = "artifacts"):
        """
        Initialize artifact tracker.
        
        Args:
            tracking_dir: Directory to store artifact metadata
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ArtifactTracker initialized: {tracking_dir}")
    
    def track_dataset(
        self,
        dataset_path: str,
        description: str = "",
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track a dataset artifact.
        
        Args:
            dataset_path: Path to dataset
            description: Dataset description
            additional_metadata: Additional metadata
            
        Returns:
            Dataset version ID
        """
        metadata = create_artifact_metadata(
            artifact_name=Path(dataset_path).name,
            artifact_type='dataset',
            additional_info={
                'description': description,
                'path': dataset_path,
                **(additional_metadata or {})
            }
        )
        
        version = version_dataset(dataset_path, metadata)
        
        # Save tracking metadata
        tracking_file = self.tracking_dir / f"dataset_{version}.json"
        save_artifact_metadata(metadata, str(tracking_file))
        
        return version
    
    def track_model(
        self,
        model_path: str,
        model_type: str,
        dataset_version: str,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Track a model artifact.
        
        Args:
            model_path: Path to model file
            model_type: Type of model (cnn, xgboost, etc.)
            dataset_version: Version of training dataset
            config: Configuration used for training
            metrics: Training metrics
            
        Returns:
            Model version ID
        """
        config_version = version_configuration(config)
        
        lineage = create_model_lineage(
            model_name=Path(model_path).stem,
            dataset_version=dataset_version,
            config_version=config_version,
            training_metrics=metrics
        )
        
        metadata = create_artifact_metadata(
            artifact_name=Path(model_path).name,
            artifact_type='model',
            config=config,
            additional_info={
                'model_type': model_type,
                'lineage': lineage
            }
        )
        
        # Compute model file hash
        if Path(model_path).exists():
            metadata['file_hash'] = compute_file_hash(model_path)
        
        # Generate version
        version = f"model_{model_type}_{get_git_info()['commit_short']}"
        metadata['version'] = version
        
        # Save tracking metadata
        tracking_file = self.tracking_dir / f"model_{version}.json"
        save_artifact_metadata(metadata, str(tracking_file))
        
        logger.info(f"Model tracked: {version}")
        return version
