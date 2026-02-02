"""
Model comparison and champion selection utilities.
"""

import mlflow
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.exceptions import RegistryError, DataValidationError
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelComparator:
    """Compare multiple models and select the champion."""
    
    def __init__(self, primary_metric: str = 'f1_score'):
        """
        Initialize the comparator.
        
        Args:
            primary_metric: Primary metric for champion selection
            
        Raises:
            DataValidationError: If primary metric is invalid
        """
        valid_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        if primary_metric not in valid_metrics:
            raise DataValidationError(
                f"Invalid primary metric '{primary_metric}'. "
                f"Must be one of {valid_metrics}"
            )
        
        self.primary_metric = primary_metric
        logger.info(f"ModelComparator initialized with primary_metric={primary_metric}")
        
    def compare_models(self, 
                      models_results: Dict[str, Dict[str, Any]],
                      metrics_to_compare: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple models based on metrics.
        
        Args:
            models_results: Dictionary of model results
                          {model_name: {'metrics': {...}, 'model': ..., ...}}
            metrics_to_compare: List of metrics to compare
            
        Returns:
            Dictionary with comparison results and champion
            
        Raises:
            DataValidationError: If inputs are invalid
        """
        try:
            if not models_results:
                raise DataValidationError("models_results cannot be empty")
            
            if metrics_to_compare is None:
                metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
            
            logger.info(f"Comparing {len(models_results)} models on metrics: {metrics_to_compare}")
            
            comparison = {}
            
            # Extract metrics for each model
            for model_name, results in models_results.items():
                if 'metrics' not in results:
                    logger.warning(f"Model '{model_name}' has no metrics, skipping")
                    continue
                    
                metrics = results.get('metrics', {})
                comparison[model_name] = {
                    metric: metrics.get(metric, 0.0) 
                    for metric in metrics_to_compare
                }
            
            if not comparison:
                raise DataValidationError("No models with valid metrics found")
            
            # Calculate scores for each model
            scores = {}
            for model_name, metrics in comparison.items():
                # Primary metric has highest weight
                primary_value = metrics.get(self.primary_metric, 0.0)
                
                # Calculate average of other metrics
                other_metrics = [v for k, v in metrics.items() if k != self.primary_metric]
                avg_other = np.mean(other_metrics) if other_metrics else 0.0
                
                # Combined score: 70% primary metric, 30% average of others
                scores[model_name] = 0.7 * primary_value + 0.3 * avg_other
                logger.debug(f"{model_name}: score={scores[model_name]:.4f}, "
                           f"{self.primary_metric}={primary_value:.4f}")
            
            # Select champion (highest score)
            champion = max(scores.items(), key=lambda x: x[1])
            champion_name = champion[0]
            champion_score = champion[1]
            
            logger.info(f"Champion selected: {champion_name} (score={champion_score:.4f})")
            
            return {
                'comparison': comparison,
                'scores': scores,
                'champion': {
                    'name': champion_name,
                    'score': champion_score,
                    'metrics': comparison[champion_name]
                },
                'primary_metric': self.primary_metric
            }
        
        except DataValidationError:
            raise
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            raise RegistryError(f"Failed to compare models: {e}") from e
    
    def print_comparison(self, comparison_results: Dict[str, Any]):
        """
        Pretty print comparison results.
        
        Args:
            comparison_results: Results from compare_models()
        """
        print("\n" + "=" * 70)
        print("MODEL COMPARISON & CHAMPION SELECTION")
        print("=" * 70)
        
        comparison = comparison_results['comparison']
        champion = comparison_results['champion']
        scores = comparison_results['scores']
        
        # Print metric comparison table
        print(f"\nPrimary Metric: {self.primary_metric}")
        print("\nMetrics Comparison:")
        print("-" * 70)
        
        # Get all metrics
        all_metrics = list(next(iter(comparison.values())).keys())
        
        # Header
        header = f"{'Metric':<20}"
        for model_name in comparison.keys():
            header += f"{model_name:>15}"
        header += f"{'Winner':>15}"
        print(header)
        print("-" * 70)
        
        # Metrics rows
        for metric in all_metrics:
            row = f"{metric.capitalize():<20}"
            values = {name: comp[metric] for name, comp in comparison.items()}
            max_val = max(values.values())
            
            for model_name in comparison.keys():
                val = values[model_name]
                row += f"{val:>15.4f}"
            
            # Mark winner for this metric
            winner = max(values.items(), key=lambda x: x[1])[0]
            row += f"{winner:>15}"
            print(row)
        
        print("-" * 70)
        
        # Overall scores
        print("\nOverall Scores (70% primary metric + 30% others):")
        print("-" * 70)
        for model_name, score in scores.items():
            is_champion = "🏆 CHAMPION" if model_name == champion['name'] else ""
            print(f"{model_name:<20} {score:>10.4f}  {is_champion}")
        
        print("-" * 70)
        print(f"\n🏆 Champion Model: {champion['name']}")
        print(f"   Champion Score: {champion['score']:.4f}")
        print(f"   {self.primary_metric.upper()}: {champion['metrics'][self.primary_metric]:.4f}")


def register_champion_model(model_name: str,
                           run_id: str,
                           model_path: str = "model",
                           stage: str = "Production",
                           description: Optional[str] = None) -> Dict[str, Any]:
    """
    Register a model to MLflow Model Registry and promote to production.
    
    Args:
        model_name: Name for the registered model
        run_id: MLflow run ID containing the model
        model_path: Path to model artifacts within the run
        stage: Stage to transition to (None, Staging, Production, Archived)
        description: Optional description for the model version
        
    Returns:
        Dictionary with registration details
        
    Raises:
        RegistryError: If registration fails
        DataValidationError: If inputs are invalid
    """
    try:
        # Validate inputs
        if not model_name:
            raise DataValidationError("model_name cannot be empty")
        if not run_id:
            raise DataValidationError("run_id cannot be empty")
        
        valid_stages = ['None', 'Staging', 'Production', 'Archived']
        if stage not in valid_stages:
            raise DataValidationError(
                f"Invalid stage '{stage}'. Must be one of {valid_stages}"
            )
        
        logger.info(f"Registering model '{model_name}' from run {run_id[:8]}...")
        
        # Get the model URI
        model_uri = f"runs:/{run_id}/{model_path}"
        
        # Register the model
        print(f"\n📦 Registering model '{model_name}' from run {run_id[:8]}...")
        model_version = mlflow.register_model(model_uri, model_name)
        
        logger.info(f"✓ Model registered: {model_name} version {model_version.version}")
        print(f"✓ Model registered: {model_name} version {model_version.version}")
        
        # Add description if provided
        if description:
            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
            logger.debug(f"Description added to model version")
            print(f"✓ Description added")
        
        # Transition to specified stage
        if stage and stage != "None":
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            logger.info(f"Model transitioned to '{stage}' stage")
            print(f"✓ Model transitioned to '{stage}' stage")
        
        print(f"\n🎉 Model '{model_name}' v{model_version.version} is ready for {stage}!")
        logger.info(f"Registration complete: {model_name} v{model_version.version}")
        
        return {
            'model_name': model_name,
            'version': model_version.version,
            'stage': stage,
            'run_id': run_id,
            'model_uri': model_uri
        }
    
    except DataValidationError:
        raise
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"MLflow error during registration: {e}")
        print(f"❌ MLflow error: {e}")
        raise RegistryError(f"Failed to register model to MLflow: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
        print(f"❌ Error registering model: {e}")
        raise RegistryError(f"Model registration failed: {e}") from e


def get_production_model(model_name: str) -> Optional[Any]:
    """
    Get the current production model from MLflow Model Registry.
    
    Args:
        model_name: Name of the registered model
        
    Returns:
        Loaded model or None if not found
    """
    try:
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✓ Loaded production model: {model_name}")
        return model
    except Exception as e:
        print(f"⚠️  Could not load production model '{model_name}': {e}")
        return None


def compare_and_register_champion(models_results: Dict[str, Dict[str, Any]],
                                  primary_metric: str = 'f1_score',
                                  auto_register: bool = True,
                                  stage: str = "Production") -> Dict[str, Any]:
    """
    Compare models, select champion, and optionally register to Model Registry.
    
    Args:
        models_results: Dictionary of model results with run IDs
        primary_metric: Primary metric for champion selection
        auto_register: Whether to automatically register the champion
        stage: Stage to transition champion to
        
    Returns:
        Dictionary with comparison and registration results
    """
    # Compare models
    comparator = ModelComparator(primary_metric=primary_metric)
    comparison = comparator.compare_models(models_results)
    comparator.print_comparison(comparison)
    
    # Get champion details
    champion_name = comparison['champion']['name']
    champion_results = models_results[champion_name]
    
    # Register champion if requested
    registration = None
    if auto_register:
        # Get run ID from results
        run_id = champion_results.get('run_id')
        
        if run_id:
            # Create model name from champion name
            model_name = f"Engine{champion_name.replace('_', '')}"
            
            # Create description
            metrics = comparison['champion']['metrics']
            description = (
                f"Champion model selected based on {primary_metric}. "
                f"Metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            )
            
            # Register
            registration = register_champion_model(
                model_name=model_name,
                run_id=run_id,
                stage=stage,
                description=description
            )
        else:
            print("⚠️  Warning: No run_id found for champion model. Cannot register.")
    
    return {
        'comparison': comparison,
        'champion': champion_name,
        'registration': registration
    }
