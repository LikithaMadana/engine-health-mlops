"""
Hyperparameter tuning with Optuna for CNN model.
"""

import optuna
import mlflow
import numpy as np
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_cnn import train_cnn_with_mlflow


def optimize_cnn(X_train: np.ndarray,
                y_train: np.ndarray,
                X_test: np.ndarray,
                y_test: np.ndarray,
                n_trials: int = 20,
                experiment_name: str = "Engine Health Prediction Pipeline") -> Dict[str, Any]:
    """
    Optimize CNN hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_trials: Number of optimization trials
        experiment_name: MLflow experiment name
        
    Returns:
        Dictionary with best parameters and results
    """
    
    def objective(trial):
        """Optuna objective function."""
        # Suggest hyperparameters
        hyperparams = {
            'window_size': X_train.shape[1],  # Fixed from data
            'hidden_channels': trial.suggest_categorical('hidden_channels', [32, 64, 128]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'num_epochs': 30  # Reduced for faster tuning
        }
        
        # Train model
        run_name = f"CNN_Optuna_Trial_{trial.number}"
        results = train_cnn_with_mlflow(
            X_train=X_train, y_train=y_train, 
            X_test=X_test, y_test=y_test,
            hyperparams=hyperparams
        )
        
        # Return F1 score as optimization metric
        return results['metrics']['f1_score']
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='cnn_optimization'
    )
    
    # Optimize
    print("Starting CNN hyperparameter optimization...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Log best trial to MLflow
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="CNN_Best_Trial"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1_score", study.best_value)
        
        # Log optimization history
        for trial in study.trials:
            mlflow.log_metric(f"trial_{trial.number}_f1", trial.value)
    
    print(f"\nBest trial F1 score: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")
    
    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'study': study
    }


if __name__ == "__main__":
    from data.preprocessing import EngineDataPreprocessor
    
    # Prepare data
    preprocessor = EngineDataPreprocessor(window_size=50)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data_for_cnn()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Optimize hyperparameters
    results = optimize_cnn(X_train, y_train, X_test, y_test, n_trials=10)
    
    print("\nOptimization complete!")
