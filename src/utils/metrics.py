"""
Evaluation metrics utilities for model assessment.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from typing import Dict, Any


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC calculation)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }
    
    # Calculate AUC if probabilities are provided
    if y_proba is not None:
        try:
            # For binary classification, use the probability of the positive class
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba
            metrics['auc'] = roc_auc_score(y_true, y_proba_pos)
        except Exception as e:
            print(f"Warning: Could not calculate AUC - {e}")
            metrics['auc'] = 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, Any], model_name: str = "Model"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{model_name} Performance Metrics:")
    print("=" * 50)
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{metric_name.capitalize():20s}: {value:.4f}")
        else:
            print(f"{metric_name.capitalize():20s}: {value}")
    print("=" * 50)
