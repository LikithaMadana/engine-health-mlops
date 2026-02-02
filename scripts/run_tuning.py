"""
Example script for hyperparameter tuning with Optuna.

This script demonstrates how to optimize hyperparameters for both models.
"""

import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import EngineDataPreprocessor
from src.training.tune_cnn import optimize_cnn
from src.training.tune_xgboost import optimize_xgboost


def main():
    """Run hyperparameter optimization."""
    
    print("=" * 70)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("=" * 70)
    
    # Configuration
    EXPERIMENT_NAME = "Engine Health Prediction Pipeline"
    N_TRIALS = 10  # Increase for better results
    
    # ========================================================================
    # STEP 1: Prepare Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPARATION")
    print("=" * 70)
    
    preprocessor = EngineDataPreprocessor(window_size=50)
    
    # Data for XGBoost
    print("\nPreparing data for XGBoost optimization...")
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = preprocessor.prepare_data()
    
    # Data for CNN
    print("Preparing data for CNN optimization...")
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = preprocessor.prepare_data_for_cnn()
    
    # ========================================================================
    # STEP 2: Optimize XGBoost
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: OPTIMIZE XGBOOST HYPERPARAMETERS")
    print("=" * 70)
    
    print(f"\nRunning {N_TRIALS} trials...")
    print("Tuning: max_depth, n_estimators, learning_rate")
    
    xgb_results = optimize_xgboost(
        X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb,
        n_trials=N_TRIALS,
        experiment_name=EXPERIMENT_NAME
    )
    
    # ========================================================================
    # STEP 3: Optimize CNN
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: OPTIMIZE CNN HYPERPARAMETERS")
    print("=" * 70)
    
    print(f"\nRunning {N_TRIALS} trials...")
    print("Tuning: hidden_channels, learning_rate, batch_size")
    
    cnn_results = optimize_cnn(
        X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn,
        n_trials=N_TRIALS,
        experiment_name=EXPERIMENT_NAME
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)
    
    print("\nBest XGBoost Parameters:")
    for key, value in xgb_results['best_params'].items():
        print(f"  {key}: {value}")
    print(f"  Best F1 Score: {xgb_results['best_value']:.4f}")
    
    print("\nBest CNN Parameters:")
    for key, value in cnn_results['best_params'].items():
        print(f"  {key}: {value}")
    print(f"  Best F1 Score: {cnn_results['best_value']:.4f}")
    
    print(f"\nView results in MLflow UI:")
    print(f"  Run: mlflow ui")
    print(f"  Then navigate to: http://localhost:5000")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
