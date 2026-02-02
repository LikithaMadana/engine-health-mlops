"""
XGBoost training script with MLflow tracking and feature engineering.
"""

import numpy as np
import mlflow
import mlflow.xgboost
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.xgboost_model import EngineXGBoost
from utils.metrics import calculate_metrics, print_metrics
from data.feature_engineering import engineer_features_for_xgboost


def train_xgboost_with_mlflow(X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              hyperparams: Dict[str, Any] = None,
                              experiment_name: str = "Engine Health Prediction Pipeline",
                              run_name: str = "XGBoost_Model_Run") -> Dict[str, Any]:
    """
    Train XGBoost model with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        hyperparams: Dictionary of hyperparameters
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        
    Returns:
        Dictionary with training results and metrics
    """
    # Set default hyperparameters
    if hyperparams is None:
        hyperparams = {
            'max_depth': 6,
            'n_estimators': 100,
            'learning_rate': 0.1
        }
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params(hyperparams)
        
        # Initialize model
        model = EngineXGBoost(
            max_depth=hyperparams.get('max_depth', 6),
            n_estimators=hyperparams.get('n_estimators', 100),
            learning_rate=hyperparams.get('learning_rate', 0.1)
        )
        
        print("Training XGBoost with feature engineering...")
        
        # Feature engineering for XGBoost
        # If windowed data (3D), apply feature engineering
        if len(X_train.shape) == 3:
            print(f"Applying feature engineering to windowed data...")
            print(f"  Input shape: {X_train.shape}")
            
            # Extract engineered features
            X_train_flat, feature_names = engineer_features_for_xgboost(X_train)
            X_test_flat, _ = engineer_features_for_xgboost(X_test)
            
            print(f"  Engineered features: {X_train_flat.shape[1]}")
            mlflow.log_param("feature_engineering", "enabled")
            mlflow.log_param("n_engineered_features", X_train_flat.shape[1])
        else:
            # Already flat data
            X_train_flat = X_train
            X_test_flat = X_test
            mlflow.log_param("feature_engineering", "disabled")
        
        # Train model
        evals_result = model.fit(X_train_flat, y_train, X_test_flat, y_test, verbose=False)
        
        # Make predictions
        y_pred = model.predict(X_test_flat)
        y_proba = model.predict_proba(X_test_flat)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        
        # Add training loss
        if 'validation_1' in evals_result:
            final_loss = evals_result['validation_1']['logloss'][-1]
            metrics['train_loss'] = final_loss
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.xgboost.log_model(model.model, "model")
        
        # Get run ID for model registry
        run_id = mlflow.active_run().info.run_id
        
        # Print results
        print_metrics(metrics, "XGBoost")
        
        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_proba,
            'metrics': metrics,
            'evals_result': evals_result,
            'run_id': run_id
        }


if __name__ == "__main__":
    # Example usage
    from data.preprocessing import EngineDataPreprocessor
    
    # Prepare data (without windowing for XGBoost)
    preprocessor = EngineDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train model
    results = train_xgboost_with_mlflow(X_train, y_train, X_test, y_test)
    
    print("\nTraining complete!")
