"""
CNN training script aligned with baseline conventions.
Uses dict-based config, EngineHealthCNN, EngineDataset, and prepare_data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import EngineHealthCNN
from data.preprocessing import EngineDataset, prepare_data
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# --- Configuration (Baseline Style: Dict-based) ---
LOADING_CONFIG = {
    "test_size": 0.2
}

TRAINING_CONFIG = {
    "window_size": 10,
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32,
    "hidden_channels": 16
}

MLFLOW_CONFIG = {
    "experiment_name": "Engine Health - CNN Training",
    "run_name": "baseline_cnn"
}


def train_cnn_with_mlflow(data_path: str = None, 
                          X_train: np.ndarray = None,
                          y_train: np.ndarray = None,
                          X_test: np.ndarray = None,
                          y_test: np.ndarray = None,
                          hyperparams: dict = None):
    """
    Train CNN model with MLflow tracking.
    Can accept either data_path OR arrays.
    """
    print("=" * 60)
    print("CNN Training Pipeline (Baseline Architecture)")
    print("=" * 60)
    
    # Set up MLflow
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
    
    with mlflow.start_run(run_name=MLFLOW_CONFIG["run_name"]):
        # Log configuration
        mlflow.log_params({
            "model_type": "EngineHealthCNN",
            "window_size": TRAINING_CONFIG["window_size"],
            "hidden_channels": TRAINING_CONFIG["hidden_channels"],
            "learning_rate": TRAINING_CONFIG["learning_rate"],
            "epochs": TRAINING_CONFIG["epochs"],
            "batch_size": TRAINING_CONFIG["batch_size"],
            "input_channels": 6  # Baseline uses 6 features
        })
        
        # Use hyperparams if provided
        config = TRAINING_CONFIG.copy()
        if hyperparams:
            config.update(hyperparams)
            # Log overridden params
            for k, v in hyperparams.items():
                mlflow.log_param(f"tuned_{k}", v)
        
        # 1. Prepare data (baseline function or use provided arrays)
        print("\n1. Loading and preparing data...")
        if data_path:
            train_X, train_y, test_X, test_y, scaler = prepare_data(data_path, LOADING_CONFIG)
        elif X_train is not None:
            # Use provided arrays directly
            train_X, train_y = X_train, y_train
            test_X, test_y = X_test, y_test
            scaler = None
        else:
            raise ValueError("Must provide either data_path or training arrays")
        print(f"   Train samples: {len(train_X)}, Test samples: {len(test_X)}")
        
        # 2. Create datasets (baseline class)
        print("\n2. Creating data loaders...")
        train_ds = EngineDataset(train_X, train_y, config["window_size"])
        test_ds = EngineDataset(test_X, test_y, config["window_size"])
        
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=config["batch_size"])
        
        print(f"   Train windows: {len(train_ds)}, Test windows: {len(test_ds)}")
        
        # 3. Initialize model (baseline architecture)
        print("\n3. Initializing model...")
        model = EngineHealthCNN(
            input_channels=6,
            window_size=config["window_size"],
            hidden_channels=config["hidden_channels"]
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        mlflow.log_param("total_parameters", total_params)
        
        # 4. Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        # 5. Training loop
        print(f"\n4. Training for {config['epochs']} epochs...")
        train_losses = []
        
        model.train()
        for epoch in range(config["epochs"]):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{TRAINING_CONFIG['epochs']} - Loss: {avg_loss:.4f}")
            
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
        
        # 6. Evaluation
        print("\n5. Evaluating model...")
        model.eval()
        all_preds, all_targets, all_probs = [], [], []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                
                probs = outputs.cpu().numpy()
                preds = (outputs > 0.5).float().cpu().numpy()
                
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_targets.extend(batch_y.cpu().numpy().tolist())
        
        # Calculate metrics
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        accuracy = accuracy_score(all_targets, all_preds)
        
        # Log metrics
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy
        }
        
        mlflow.log_metrics(metrics)
        
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        for name, value in metrics.items():
            print(f"{name.upper():15s}: {value:.4f}")
        
        # 7. Save model
        print("\n6. Saving model artifacts...")
        mlflow.pytorch.log_model(model, "model")
        
        # Save additional artifacts
        import pickle
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        torch.save(model.state_dict(), model_dir / "baseline_cnn_model.pth")
        with open(model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        
        return {
            "model": model,
            "metrics": metrics,
            "scaler": scaler,
            "run_id": mlflow.active_run().info.run_id
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CNN model (baseline architecture)")
    parser.add_argument("--data-path", type=str, default="data/raw/engine_data.csv",
                       help="Path to data CSV file")
    args = parser.parse_args()
    
    results = train_cnn_with_mlflow(args.data_path)
    print(f"\nRun ID: {results['run_id']}")
    print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
