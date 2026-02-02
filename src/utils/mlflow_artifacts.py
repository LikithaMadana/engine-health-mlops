"""
MLflow artifact logging utilities.

Comprehensive artifact logging for MLflow including:
- Environment files (conda.yaml, requirements.txt)
- Configuration files (JSON)
- Evaluation results (JSON, images)
- Model metadata
"""

import os
import json
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
from pathlib import Path
import mlflow
from sklearn.metrics import confusion_matrix, roc_curve, auc

from .logger import setup_logger
from .artifacts import get_git_info

logger = setup_logger(__name__)


def log_conda_environment(
    python_version: str = "3.9",
    additional_packages: Optional[List[str]] = None
) -> None:
    """
    Log conda.yaml environment file to MLflow.
    
    Args:
        python_version: Python version to specify
        additional_packages: Additional conda packages to include
    """
    conda_env = {
        'name': 'engine-health-env',
        'channels': ['defaults', 'conda-forge'],
        'dependencies': [
            f'python={python_version}',
            'pip',
            {
                'pip': [
                    'torch>=2.0.0',
                    'numpy>=1.21.0',
                    'scikit-learn>=1.0.0',
                    'xgboost>=1.5.0',
                    'mlflow>=2.0.0',
                    'pandas>=1.3.0',
                    'matplotlib>=3.5.0',
                    'seaborn>=0.11.0',
                ]
            }
        ]
    }
    
    if additional_packages:
        conda_env['dependencies'].extend(additional_packages)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(conda_env, f, default_flow_style=False)
        temp_path = f.name
    
    try:
        mlflow.log_artifact(temp_path, artifact_path="environment")
        logger.info("Logged conda.yaml to MLflow")
    finally:
        os.unlink(temp_path)


def log_requirements_txt() -> None:
    """
    Log requirements.txt file to MLflow.
    """
    requirements = [
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        'mlflow>=2.0.0',
        'pandas>=1.3.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'fastapi>=0.95.0',
        'uvicorn>=0.21.0',
        'pydantic>=1.10.0',
        'optuna>=3.0.0',
        'kfp>=2.0.0',
    ]
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('\n'.join(requirements))
        temp_path = f.name
    
    try:
        mlflow.log_artifact(temp_path, artifact_path="environment")
        logger.info("Logged requirements.txt to MLflow")
    finally:
        os.unlink(temp_path)


def log_model_config(
    model_type: str,
    hyperparams: Dict[str, Any],
    input_shape: tuple,
    output_shape: tuple
) -> None:
    """
    Log model configuration as JSON.
    
    Args:
        model_type: Type of model (CNN, XGBoost)
        hyperparams: Model hyperparameters
        input_shape: Input data shape
        output_shape: Output data shape
    """
    config = {
        'model_type': model_type,
        'hyperparameters': hyperparams,
        'input_shape': list(input_shape) if isinstance(input_shape, tuple) else input_shape,
        'output_shape': list(output_shape) if isinstance(output_shape, tuple) else output_shape,
    }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        temp_path = f.name
    
    try:
        mlflow.log_artifact(temp_path, artifact_path="configs")
        logger.info(f"Logged model_config.json for {model_type}")
    finally:
        os.unlink(temp_path)


def log_training_config(
    training_params: Dict[str, Any],
    data_params: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log training configuration as JSON.
    
    Args:
        training_params: Training configuration (epochs, batch size, etc.)
        data_params: Data configuration (test_size, window_size, etc.)
    """
    config = {
        'training': training_params,
        'data': data_params or {}
    }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        temp_path = f.name
    
    try:
        mlflow.log_artifact(temp_path, artifact_path="configs")
        logger.info("Logged training_config.json")
    finally:
        os.unlink(temp_path)


def log_evaluation_metrics(metrics: Dict[str, float]) -> None:
    """
    Log evaluation metrics as JSON file.
    
    Args:
        metrics: Dictionary of metric names and values
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metrics, f, indent=2)
        temp_path = f.name
    
    try:
        mlflow.log_artifact(temp_path, artifact_path="evaluation")
        logger.info("Logged metrics.json")
    finally:
        os.unlink(temp_path)


def log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Log confusion matrix as image.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes for labels
    """
    if class_names is None:
        class_names = ['Healthy', 'Unhealthy']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        plt.savefig(f.name, dpi=150, bbox_inches='tight')
        temp_path = f.name
    
    plt.close()
    
    try:
        mlflow.log_artifact(temp_path, artifact_path="evaluation")
        logger.info("Logged confusion_matrix.png")
    finally:
        os.unlink(temp_path)


def log_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Log ROC curve as image.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities (2D array)
        class_names: Names of classes for labels
    """
    if class_names is None:
        class_names = ['Healthy', 'Unhealthy']
    
    # For binary classification, use positive class probabilities
    if y_proba.ndim == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        plt.savefig(f.name, dpi=150, bbox_inches='tight')
        temp_path = f.name
    
    plt.close()
    
    try:
        mlflow.log_artifact(temp_path, artifact_path="evaluation")
        logger.info("Logged roc_curve.png")
    finally:
        os.unlink(temp_path)


def log_training_history(
    history: Dict[str, List[float]],
    model_type: str
) -> None:
    """
    Log training history as JSON and plot.
    
    Args:
        history: Training history (loss, accuracy per epoch)
        model_type: Type of model for plot title
    """
    # Log history as JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(history, f, indent=2)
        temp_path_json = f.name
    
    try:
        mlflow.log_artifact(temp_path_json, artifact_path="evaluation")
        logger.info("Logged training_history.json")
    finally:
        os.unlink(temp_path_json)
    
    # Create loss plot if available
    if 'train_loss' in history or 'val_loss' in history:
        plt.figure(figsize=(10, 6))
        
        if 'train_loss' in history:
            plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
        
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_type} Training History')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plt.savefig(f.name, dpi=150, bbox_inches='tight')
            temp_path_plot = f.name
        
        plt.close()
        
        try:
            mlflow.log_artifact(temp_path_plot, artifact_path="evaluation")
            logger.info("Logged training_history.png")
        finally:
            os.unlink(temp_path_plot)


def log_model_info(
    model_type: str,
    num_parameters: int,
    model_architecture: Optional[str] = None
) -> None:
    """
    Log model information as JSON.
    
    Args:
        model_type: Type of model
        num_parameters: Number of trainable parameters
        model_architecture: String representation of architecture
    """
    info = {
        'model_type': model_type,
        'num_parameters': num_parameters,
        'architecture': model_architecture
    }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(info, f, indent=2)
        temp_path = f.name
    
    try:
        mlflow.log_artifact(temp_path, artifact_path="metadata")
        logger.info("Logged model_info.json")
    finally:
        os.unlink(temp_path)


def log_git_info() -> None:
    """
    Log Git information as JSON.
    """
    try:
        git_info = get_git_info()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(git_info, f, indent=2)
            temp_path = f.name
        
        try:
            mlflow.log_artifact(temp_path, artifact_path="metadata")
            logger.info("Logged git_info.json")
        finally:
            os.unlink(temp_path)
    except Exception as e:
        logger.warning(f"Could not log git info: {e}")


def log_run_metadata(
    model_type: str,
    dataset_size: Dict[str, int],
    training_time: float,
    device: str
) -> None:
    """
    Log run metadata as JSON.
    
    Args:
        model_type: Type of model
        dataset_size: Dictionary with train/test sizes
        training_time: Training duration in seconds
        device: Device used (cpu/cuda)
    """
    import platform
    import datetime
    
    metadata = {
        'model_type': model_type,
        'dataset_size': dataset_size,
        'training_time_seconds': training_time,
        'device': device,
        'timestamp': datetime.datetime.now().isoformat(),
        'system': {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
        }
    }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metadata, f, indent=2)
        temp_path = f.name
    
    try:
        mlflow.log_artifact(temp_path, artifact_path="metadata")
        logger.info("Logged run_metadata.json")
    finally:
        os.unlink(temp_path)


def log_comprehensive_artifacts(
    model_type: str,
    hyperparams: Dict[str, Any],
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    input_shape: tuple,
    output_shape: tuple,
    training_history: Optional[Dict[str, List[float]]] = None,
    num_parameters: Optional[int] = None,
    training_time: Optional[float] = None,
    device: str = "cpu"
) -> None:
    """
    Log all comprehensive artifacts to MLflow in one call.
    
    Args:
        model_type: Type of model (CNN, XGBoost)
        hyperparams: Model hyperparameters
        metrics: Evaluation metrics
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        input_shape: Input data shape
        output_shape: Output data shape
        training_history: Training history (optional)
        num_parameters: Number of model parameters (optional)
        training_time: Training duration (optional)
        device: Device used for training
    """
    logger.info(f"Logging comprehensive artifacts for {model_type}...")
    
    try:
        # Environment files
        log_conda_environment()
        log_requirements_txt()
        
        # Configuration files
        log_model_config(model_type, hyperparams, input_shape, output_shape)
        log_training_config(
            training_params=hyperparams,
            data_params={'input_shape': input_shape, 'output_shape': output_shape}
        )
        
        # Evaluation files
        log_evaluation_metrics(metrics)
        log_confusion_matrix(y_true, y_pred)
        log_roc_curve(y_true, y_proba)
        
        # Training history
        if training_history:
            log_training_history(training_history, model_type)
        
        # Metadata
        if num_parameters:
            log_model_info(model_type, num_parameters)
        
        log_git_info()
        
        if training_time:
            log_run_metadata(
                model_type=model_type,
                dataset_size={'train': len(y_true), 'test': len(y_true)},
                training_time=training_time,
                device=device
            )
        
        logger.info(f"Successfully logged all artifacts for {model_type}")
        
    except Exception as e:
        logger.error(f"Error logging artifacts: {e}", exc_info=True)
        raise
