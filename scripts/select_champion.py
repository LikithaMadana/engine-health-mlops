"""
Standalone script to select champion model and register to Model Registry.

This script can be run independently to compare existing MLflow runs
and register the best model to the Model Registry.
"""

import sys
import os
import mlflow
from mlflow.tracking import MlflowClient

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.model_registry import ModelComparator, register_champion_model


def get_experiment_runs(experiment_name: str, model_filter: str = None):
    """
    Get runs from an MLflow experiment.
    
    Args:
        experiment_name: Name of the experiment
        model_filter: Optional filter for run names (e.g., "CNN", "XGBoost")
        
    Returns:
        Dictionary of runs with their metrics
    """
    client = MlflowClient()
    
    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"❌ Experiment '{experiment_name}' not found!")
        return {}
    
    # Get runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["start_time DESC"],
        max_results=100
    )
    
    # Extract relevant information
    results = {}
    for run in runs:
        run_name = run.data.tags.get('mlflow.runName', 'Unknown')
        
        # Apply filter if provided
        if model_filter and model_filter not in run_name:
            continue
        
        # Extract metrics
        metrics = {
            'accuracy': run.data.metrics.get('accuracy', 0.0),
            'precision': run.data.metrics.get('precision', 0.0),
            'recall': run.data.metrics.get('recall', 0.0),
            'f1_score': run.data.metrics.get('f1_score', 0.0),
            'auc': run.data.metrics.get('auc', 0.0)
        }
        
        results[run_name] = {
            'metrics': metrics,
            'run_id': run.info.run_id,
            'start_time': run.info.start_time
        }
    
    return results


def select_and_register_champion(experiment_name: str = "Engine Health Prediction Pipeline",
                                 primary_metric: str = 'f1_score',
                                 stage: str = 'Production'):
    """
    Select champion from experiment runs and register to Model Registry.
    
    Args:
        experiment_name: MLflow experiment name
        primary_metric: Primary metric for champion selection
        stage: Stage to promote champion to
    """
    print("=" * 70)
    print("CHAMPION MODEL SELECTION FROM MLFLOW EXPERIMENT")
    print("=" * 70)
    
    print(f"\nExperiment: {experiment_name}")
    print(f"Primary Metric: {primary_metric}")
    print(f"Target Stage: {stage}")
    
    # Get runs for each model type
    print("\n📊 Fetching runs from MLflow...")
    
    cnn_runs = get_experiment_runs(experiment_name, model_filter="CNN")
    xgb_runs = get_experiment_runs(experiment_name, model_filter="XGBoost")
    
    if not cnn_runs and not xgb_runs:
        print("❌ No runs found! Please train models first.")
        return
    
    print(f"  Found {len(cnn_runs)} CNN runs")
    print(f"  Found {len(xgb_runs)} XGBoost runs")
    
    # Select best run for each model type
    models_results = {}
    
    if cnn_runs:
        # Get best CNN run
        best_cnn = max(cnn_runs.items(), 
                      key=lambda x: x[1]['metrics'].get(primary_metric, 0.0))
        models_results['CNN'] = best_cnn[1]
        print(f"\n  Best CNN: {best_cnn[0]}")
        print(f"    {primary_metric}: {best_cnn[1]['metrics'][primary_metric]:.4f}")
    
    if xgb_runs:
        # Get best XGBoost run
        best_xgb = max(xgb_runs.items(),
                      key=lambda x: x[1]['metrics'].get(primary_metric, 0.0))
        models_results['XGBoost'] = best_xgb[1]
        print(f"\n  Best XGBoost: {best_xgb[0]}")
        print(f"    {primary_metric}: {best_xgb[1]['metrics'][primary_metric]:.4f}")
    
    # Compare and select champion
    comparator = ModelComparator(primary_metric=primary_metric)
    comparison = comparator.compare_models(models_results)
    comparator.print_comparison(comparison)
    
    # Get champion details
    champion_name = comparison['champion']['name']
    champion_results = models_results[champion_name]
    run_id = champion_results['run_id']
    
    # Register champion
    print("\n" + "=" * 70)
    print("REGISTERING CHAMPION TO MODEL REGISTRY")
    print("=" * 70)
    
    model_name = f"Engine{champion_name.replace('_', '')}"
    metrics = comparison['champion']['metrics']
    description = (
        f"Champion model selected based on {primary_metric}. "
        f"Metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    )
    
    registration = register_champion_model(
        model_name=model_name,
        run_id=run_id,
        stage=stage,
        description=description
    )
    
    if registration:
        print("\n" + "=" * 70)
        print("✅ CHAMPION MODEL SUCCESSFULLY REGISTERED!")
        print("=" * 70)
        print(f"\nModel Details:")
        print(f"  Name: {registration['model_name']}")
        print(f"  Version: {registration['version']}")
        print(f"  Stage: {registration['stage']}")
        print(f"  Run ID: {registration['run_id'][:8]}...")
        print(f"\n🚀 Model is ready for {stage} deployment!")
        print(f"\nView in MLflow UI:")
        print(f"  1. Run: mlflow ui")
        print(f"  2. Navigate to: http://localhost:5000")
        print(f"  3. Click 'Models' tab")
        print(f"  4. Find '{registration['model_name']}'")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Select and register champion model from MLflow experiment"
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default="Engine Health Prediction Pipeline",
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='f1_score',
        choices=['accuracy', 'precision', 'recall', 'f1_score', 'auc'],
        help='Primary metric for champion selection'
    )
    parser.add_argument(
        '--stage',
        type=str,
        default='Production',
        choices=['None', 'Staging', 'Production', 'Archived'],
        help='Stage to promote champion to'
    )
    
    args = parser.parse_args()
    
    select_and_register_champion(
        experiment_name=args.experiment,
        primary_metric=args.metric,
        stage=args.stage
    )


if __name__ == "__main__":
    main()
