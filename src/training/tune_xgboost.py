"""
Hyperparameter tuning with Optuna for XGBoost model.
"""

import optuna
import mlflow
import numpy as np
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_xgboost import train_xgboost_with_mlflow


def optimize_xgboost(X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    n_trials: int = 20,
                    experiment_name: str = "Engine Health Prediction Pipeline") -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Optuna.
    
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
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
        
        # Train model
        run_name = f"XGBoost_Optuna_Trial_{trial.number}"
        results = train_xgboost_with_mlflow(
            X_train, y_train, X_test, y_test,
            hyperparams=hyperparams,
            experiment_name=experiment_name,
            run_name=run_name
        )
        
        # Return F1 score as optimization metric
        return results['metrics']['f1_score']
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_optimization'
    )
    
    # Optimize
    print("Starting XGBoost hyperparameter optimization...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Log best trial to MLflow
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="XGBoost_Best_Trial"):
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
    preprocessor = EngineDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Optimize hyperparameters
    results = optimize_xgboost(X_train, y_train, X_test, y_test, n_trials=10)
    
    print("\nOptimization complete!")
