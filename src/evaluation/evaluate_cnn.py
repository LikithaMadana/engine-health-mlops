"""
CNN Evaluation Component - Baseline Aligned
Separate evaluation component for trained models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report
)
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import EngineHealthCNN
from data.preprocessing import EngineDataset, prepare_data

# --- Configuration (Baseline Style) ---
EVALUATION_CONFIG = {
    "window_size": 10,
    "batch_size": 32,
    "threshold": 0.5  # Classification threshold
}

LOADING_CONFIG = {
    "test_size": 0.2
}


def calculate_comprehensive_metrics(all_targets, all_preds, all_probs):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        all_targets: True labels
        all_preds: Predicted labels
        all_probs: Prediction probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1_score': f1_score(all_targets, all_preds, zero_division=0),
        'accuracy': accuracy_score(all_targets, all_preds)
    }
    
    # Add AUC if we have both classes
    if len(set(all_targets)) > 1:
        metrics['auc_roc'] = roc_auc_score(all_targets, all_probs)
    else:
        metrics['auc_roc'] = 0.0
    
    # Get confusion matrix and classification report
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, 
                                   target_names=['Healthy', 'Unhealthy'],
                                   zero_division=0)
    
    return metrics, cm, report


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """
    Create and save confusion matrix visualization.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Unhealthy'],
                yticklabels=['Healthy', 'Unhealthy'])
    plt.title('Confusion Matrix - Engine Health CNN')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_prediction_distribution(all_probs, all_targets, save_path='prediction_distribution.png'):
    """
    Plot distribution of prediction probabilities.
    
    Args:
        all_probs: Prediction probabilities
        all_targets: True labels
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(10, 6))
    
    # Separate by true class
    healthy_probs = [p for p, t in zip(all_probs, all_targets) if t == 0]
    unhealthy_probs = [p for p, t in zip(all_probs, all_targets) if t == 1]
    
    plt.hist(healthy_probs, bins=50, alpha=0.5, label='Healthy', color='green')
    plt.hist(unhealthy_probs, bins=50, alpha=0.5, label='Unhealthy', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def evaluate_model(model_path: str, 
                  data_path: str = "data/raw/engine_data.csv",
                  save_artifacts: bool = True,
                  log_to_mlflow: bool = True):
    """
    Evaluate a trained CNN model.
    
    Args:
        model_path: Path to saved model (.pth file)
        data_path: Path to data CSV file
        save_artifacts: Whether to save evaluation artifacts
        log_to_mlflow: Whether to log to MLflow
        
    Returns:
        Dictionary with evaluation results
    """
    print("=" * 70)
    print("CNN MODEL EVALUATION (Baseline Architecture)")
    print("=" * 70)
    
    # Start MLflow run if requested
    if log_to_mlflow:
        mlflow.set_experiment("Engine Health - CNN Evaluation")
        mlflow.start_run(run_name="baseline_cnn_evaluation")
    
    try:
        # 1. Load data
        print("\n1. Loading and preparing data...")
        train_X, train_y, test_X, test_y, scaler = prepare_data(data_path, LOADING_CONFIG)
        print(f"   Test samples: {len(test_X)}")
        
        # 2. Create test dataset
        print("\n2. Creating test data loader...")
        test_ds = EngineDataset(test_X, test_y, EVALUATION_CONFIG["window_size"])
        test_loader = DataLoader(test_ds, batch_size=EVALUATION_CONFIG["batch_size"])
        print(f"   Test windows: {len(test_ds)}")
        
        # 3. Load model
        print("\n3. Loading trained model...")
        model = EngineHealthCNN(
            input_channels=6,
            window_size=EVALUATION_CONFIG["window_size"],
            hidden_channels=16
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Model parameters: {total_params:,}")
        
        # 4. Run evaluation
        print("\n4. Running evaluation...")
        all_preds, all_targets, all_probs = [], [], []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x).squeeze()
                
                # Handle scalar outputs
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                
                probs = outputs.cpu().numpy()
                preds = (outputs > EVALUATION_CONFIG["threshold"]).float().cpu().numpy()
                
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_targets.extend(batch_y.cpu().numpy().tolist())
        
        # 5. Calculate metrics
        print("\n5. Calculating metrics...")
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
        
        # 6. Save artifacts
        if save_artifacts:
            print("\n6. Saving evaluation artifacts...")
            artifacts_dir = Path("evaluation_artifacts")
            artifacts_dir.mkdir(exist_ok=True)
            
            # Save metrics as JSON
            metrics_path = artifacts_dir / "evaluation_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"   Metrics saved to: {metrics_path}")
            
            # Save confusion matrix plot
            cm_plot_path = artifacts_dir / "confusion_matrix.png"
            plot_confusion_matrix(cm, cm_plot_path)
            print(f"   Confusion matrix saved to: {cm_plot_path}")
            
            # Save prediction distribution plot
            dist_plot_path = artifacts_dir / "prediction_distribution.png"
            plot_prediction_distribution(all_probs, all_targets, dist_plot_path)
            print(f"   Distribution plot saved to: {dist_plot_path}")
            
            # Save detailed results
            results_df = pd.DataFrame({
                'true_label': all_targets,
                'predicted_label': all_preds,
                'probability': all_probs
            })
            results_csv = artifacts_dir / "detailed_predictions.csv"
            results_df.to_csv(results_csv, index=False)
            print(f"   Detailed results saved to: {results_csv}")
        
        # 7. Log to MLflow
        if log_to_mlflow:
            print("\n7. Logging to MLflow...")
            mlflow.log_metrics(metrics)
            mlflow.log_param("window_size", EVALUATION_CONFIG["window_size"])
            mlflow.log_param("threshold", EVALUATION_CONFIG["threshold"])
            
            if save_artifacts:
                mlflow.log_artifact(str(cm_plot_path))
                mlflow.log_artifact(str(dist_plot_path))
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


def compare_models(model_paths: dict, data_path: str = "data/raw/engine_data.csv"):
    """
    Compare multiple trained models.
    
    Args:
        model_paths: Dictionary of {model_name: model_path}
        data_path: Path to data CSV file
        
    Returns:
        Comparison results DataFrame
    """
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    comparison_results = []
    
    for model_name, model_path in model_paths.items():
        print(f"\nEvaluating: {model_name}")
        print("-" * 70)
        
        results = evaluate_model(
            model_path=model_path,
            data_path=data_path,
            save_artifacts=False,
            log_to_mlflow=False
        )
        
        metrics = results['metrics']
        metrics['model_name'] = model_name
        comparison_results.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.set_index('model_name')
    
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(comparison_df.to_string())
    
    # Find best model for each metric
    print("\n" + "=" * 70)
    print("BEST MODELS")
    print("=" * 70)
    for metric in ['precision', 'recall', 'f1_score', 'accuracy', 'auc_roc']:
        best_model = comparison_df[metric].idxmax()
        best_value = comparison_df[metric].max()
        print(f"{metric.upper():15s}: {best_model} ({best_value:.4f})")
    
    return comparison_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CNN model (baseline architecture)")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model (.pth file)")
    parser.add_argument("--data-path", type=str, default="data/raw/engine_data.csv",
                       help="Path to data CSV file")
    parser.add_argument("--no-mlflow", action="store_true",
                       help="Disable MLflow logging")
    parser.add_argument("--no-artifacts", action="store_true",
                       help="Disable artifact saving")
    
    args = parser.parse_args()
    
    results = evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        save_artifacts=not args.no_artifacts,
        log_to_mlflow=not args.no_mlflow
    )
    
    print(f"\nOverall F1 Score: {results['metrics']['f1_score']:.4f}")
    print(f"Overall Accuracy: {results['metrics']['accuracy']:.4f}")
