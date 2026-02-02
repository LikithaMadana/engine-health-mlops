"""
Evaluation Module

This module contains evaluation components for the Engine Health MLOps pipeline.

Components:
- evaluate_cnn: Comprehensive evaluation for CNN models
- evaluate_xgboost: Comprehensive evaluation for XGBoost models

Following baseline conventions:
- Dict-based configurations
- Window size: 10
- Input features: 6
- MLflow integration
"""

from .evaluate_cnn import evaluate_model as evaluate_cnn_model
from .evaluate_cnn import compare_models as compare_cnn_models
from .evaluate_xgboost import evaluate_model as evaluate_xgboost_model

__all__ = [
    'evaluate_cnn_model',
    'compare_cnn_models',
    'evaluate_xgboost_model',
]
