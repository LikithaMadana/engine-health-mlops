"""
XGBoost Evaluation Component - Baseline Aligned
Separate evaluation component for trained XGBoost models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.xgboost
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import prepare_data

# --- Configuration (Baseline Style) ---
EVALUATION_CONFIG = {
    "threshold": 0.5
}

LOADING_CONFIG = {
    "test_size": 0.2
}


def calculate_comprehensive_metrics(all_targets, all_preds, all_probs):
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1_score': f1_score(all_targets, all_preds, zero_division=0),
        'accuracy': accuracy_score(all_targets, all_preds)
    }
    
    if len(set(all_targets)) > 1:
        metrics['auc_roc'] = roc_auc_score(all_targets, all_probs)
    else:
        metrics['auc_roc'] = 0.0
    
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds,
                                   target_names=['Healthy', 'Unhealthy'],
                                   zero_division=0)
    
    return metrics, cm, report


def plot_confusion_matrix(cm, save_path='confusion_matrix_xgb.png'):
    """Create and save confusion matrix visualization."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Healthy', 'Unhealthy'],
                yticklabels=['Healthy', 'Unhealthy'])
    plt.title('Confusion Matrix - XGBoost')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_feature_importance(model, feature_names, save_path='feature_importance.png'):
    """Plot feature importance from XGBoost model."""
    import xgboost as xgb
    
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, importance_type='weight', max_num_features=10)
    plt.title('Feature Importance - XGBoost')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def evaluate_xgboost_model(model_path: str,
                          data_path: str = "data/raw/engine_data.csv",
                          save_artifacts: bool = True,
                          log_to_mlflow: bool = True):
    """
    Evaluate a trained XGBoost model.
    
    Args:
        model_path: Path to saved model (.pkl file)
        data_path: Path to data CSV file
        save_artifacts: Whether to save evaluation artifacts
        log_to_mlflow: Whether to log to MLflow
        
    Returns:
        Dictionary with evaluation results
    """
    print("=" * 70)
    print("XGBOOST MODEL EVALUATION (Baseline Architecture)")
    print("=" * 70)
    
    # Start MLflow run if requested
    if log_to_mlflow:
        mlflow.set_experiment("Engine Health - XGBoost Evaluation")
        mlflow.start_run(run_name="baseline_xgboost_evaluation")
    
    try:
        # 1. Load data
        print("\n1. Loading and preparing data...")
        train_X, train_y, test_X, test_y, scaler = prepare_data(data_path, LOADING_CONFIG)
        print(f"   Test samples: {len(test_X)}")
        
        # 2. Load model
        print("\n2. Loading trained model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("   Model loaded successfully")
        
        # 3. Run evaluation
        print("\n3. Running evaluation...")
        
        # Get predictions
        all_probs = model.predict_proba(test_X)[:, 1]
        all_preds = (all_probs > EVALUATION_CONFIG["threshold"]).astype(int)
        all_targets = test_y
        
        # 4. Calculate metrics
        print("\n4. Calculating metrics...")
        metrics, cm, report = calculate_comprehensive_metrics(
            all_targets, all_preds, all_probs
        )
        
        # Print results
        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)
        for name, value in metrics.items():
            print(f"{name.upper():15s}: {value:.4f}")
        
        print("\n" + "=" * 70)
        print("CONFUSION MATRIX")
        print("=" * 70)
        print(cm)
        
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(report)
        
        # 5. Save artifacts
        if save_artifacts:
            print("\n5. Saving evaluation artifacts...")
            artifacts_dir = Path("evaluation_artifacts")
            artifacts_dir.mkdir(exist_ok=True)
            
            # Save metrics
            metrics_path = artifacts_dir / "evaluation_metrics_xgb.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"   Metrics saved to: {metrics_path}")
            
            # Save confusion matrix plot
            cm_plot_path = artifacts_dir / "confusion_matrix_xgb.png"
            plot_confusion_matrix(cm, cm_plot_path)
            print(f"   Confusion matrix saved to: {cm_plot_path}")
            
            # Save feature importance
            feature_names = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure',
                           'Coolant pressure', 'lub oil temp', 'Coolant temp']
            fi_plot_path = artifacts_dir / "feature_importance.png"
            plot_feature_importance(model, feature_names, fi_plot_path)
            print(f"   Feature importance saved to: {fi_plot_path}")
            
            # Save detailed results
            results_df = pd.DataFrame({
                'true_label': all_targets,
                'predicted_label': all_preds,
                'probability': all_probs
            })
            results_csv = artifacts_dir / "detailed_predictions_xgb.csv"
            results_df.to_csv(results_csv, index=False)
            print(f"   Detailed results saved to: {results_csv}")
        
        # 6. Log to MLflow
        if log_to_mlflow:
            print("\n6. Logging to MLflow...")
            mlflow.log_metrics(metrics)
            mlflow.log_param("threshold", EVALUATION_CONFIG["threshold"])
            
            if save_artifacts:
                mlflow.log_artifact(str(cm_plot_path))
                mlflow.log_artifact(str(fi_plot_path))
                mlflow.log_artifact(str(metrics_path))
                mlflow.log_artifact(str(results_csv))
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE!")
        print("=" * 70)
        
        return {
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": report,
            "predictions": all_preds,
            "probabilities": all_probs,
            "targets": all_targets
        }
    
    finally:
        if log_to_mlflow:
            mlflow.end_run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model (.pkl file)")
    parser.add_argument("--data-path", type=str, default="data/raw/engine_data.csv",
                       help="Path to data CSV file")
    parser.add_argument("--no-mlflow", action="store_true",
                       help="Disable MLflow logging")
    parser.add_argument("--no-artifacts", action="store_true",
                       help="Disable artifact saving")
    
    args = parser.parse_args()
    
    results = evaluate_xgboost_model(
        model_path=args.model_path,
        data_path=args.data_path,
        save_artifacts=not args.no_artifacts,
        log_to_mlflow=not args.no_mlflow
    )
    
    print(f"\nOverall F1 Score: {results['metrics']['f1_score']:.4f}")
    print(f"Overall Accuracy: {results['metrics']['accuracy']:.4f}")
