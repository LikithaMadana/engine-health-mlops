"""
Enhanced MLOps Pipeline Built Upon User's Baseline

This script enhances the simple baseline with MLOps features while maintaining
the original structure and simplicity.

Original baseline: Simple CNN for engine health classification
Enhancements: MLflow tracking, model registry, comprehensive metrics, artifacts
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.pytorch
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Configuration (Same as baseline, with MLOps additions) ---
LOADING_CONFIG = {
    "test_size": 0.2
}

TRAINING_CONFIG = {
    "window_size": 10,
    "step": 1,
    "learning_rate": 0.001,
    "epochs": 50,  # Production training
    "batch_size": 32,
    "hidden_channels": 16
}

MLFLOW_CONFIG = {
    "experiment_name": "Engine Health - Baseline Enhanced",
    "run_name": "baseline_cnn_mlops"
}

# --- 2. Dataset Level Windowing (UNCHANGED from baseline) ---
class EngineDataset(Dataset):
    """User's original Dataset class - kept as-is"""
    def __init__(self, features, targets, window_size):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size - 1]
        return x, y

# --- 3. Data Preparation (UNCHANGED from baseline) ---
def prepare_data(filepath, config):
    """User's original data preparation - kept as-is"""
    df = pd.read_csv(filepath)
    
    target_col = 'Engine Condition'
    feature_cols = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 
                    'Coolant pressure', 'lub oil temp', 'Coolant temp']
    
    # 1. Standardization
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 2. Split before windowing
    train_df, test_df = train_test_split(
        df, test_size=config["test_size"], random_state=42, shuffle=False
    )
    
    return (train_df[feature_cols].values, train_df[target_col].values,
            test_df[feature_cols].values, test_df[target_col].values, scaler)

# --- 4. Model Architecture (UNCHANGED from baseline) ---
class EngineHealthCNN(nn.Module):
    """User's original CNN architecture - kept as-is"""
    def __init__(self, input_channels, window_size, hidden_channels):
        super(EngineHealthCNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((hidden_channels * 2) * (window_size - 4), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.conv_stack(x)

# --- 5. Enhanced Metrics Calculation (NEW - MLOps Enhancement) ---
def calculate_comprehensive_metrics(all_targets, all_preds, all_probs):
    """
    Calculate comprehensive metrics beyond precision/recall
    Enhancement: Added F1, accuracy, AUC, confusion matrix
    """
    metrics = {
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1_score': f1_score(all_targets, all_preds, zero_division=0),
        'accuracy': accuracy_score(all_targets, all_preds),
        'auc_roc': roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.0
    }
    
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, zero_division=0)
    
    return metrics, cm, report

# --- 6. Visualization Functions (NEW - MLOps Enhancement) ---
def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Create confusion matrix visualization"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Unhealthy'],
                yticklabels=['Healthy', 'Unhealthy'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_training_history(train_losses, save_path='training_loss.png'):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

# --- 7. Model Saving/Loading (NEW - MLOps Enhancement) ---
def save_model_artifacts(model, scaler, config, metrics, save_dir='models'):
    """Save model and related artifacts"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = Path(save_dir) / 'baseline_cnn_model.pth'
    torch.save(model.state_dict(), model_path)
    
    # Save scaler
    import pickle
    scaler_path = Path(save_dir) / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save config
    config_path = Path(save_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({**TRAINING_CONFIG, **LOADING_CONFIG}, f, indent=2)
    
    # Save metrics
    metrics_path = Path(save_dir) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return {
        'model': str(model_path),
        'scaler': str(scaler_path),
        'config': str(config_path),
        'metrics': str(metrics_path)
    }

# --- 8. Enhanced Training Function (Built upon baseline) ---
def train_model_with_mlflow(model, train_loader, criterion, optimizer, config):
    """
    Training loop with MLflow tracking
    Based on user's baseline, enhanced with logging
    """
    model.train()
    train_losses = []
    
    for epoch in range(config["epochs"]):
        total_loss = 0
        batch_count = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        train_losses.append(avg_loss)
        
        # Log to MLflow every epoch
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {avg_loss:.4f}")
    
    return train_losses

# --- 9. Enhanced Evaluation Function (Built upon baseline) ---
def evaluate_model_with_mlflow(model, test_loader):
    """
    Evaluation with comprehensive metrics
    Based on user's baseline, enhanced with more metrics
    """
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze()
            
            # Handle potential scalar
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            targets = batch_y.cpu().numpy()
            
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
    
    # Calculate comprehensive metrics
    metrics, cm, report = calculate_comprehensive_metrics(
        all_targets, all_preds, all_probs
    )
    
    return metrics, cm, report, all_targets, all_preds

# --- 10. Main Execution with MLflow (Enhanced baseline) ---
def run_experiment_with_mlflow(data_path):
    """
    Main execution function - built upon user's baseline
    Enhancements: MLflow tracking, comprehensive metrics, model registry
    """
    
    # Set MLflow experiment
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
    
    # Start MLflow run
    with mlflow.start_run(run_name=MLFLOW_CONFIG["run_name"]) as run:
        
        print("="*60)
        print("Enhanced Baseline MLOps Pipeline")
        print("="*60)
        
        # Log configurations
        mlflow.log_params({
            **TRAINING_CONFIG,
            **LOADING_CONFIG,
            "model_type": "baseline_cnn"
        })
        
        # Prepare data (unchanged from baseline)
        print("\n1. Loading and preparing data...")
        train_X, train_y, test_X, test_y, scaler = prepare_data(data_path, LOADING_CONFIG)
        print(f"   Train samples: {len(train_X)}, Test samples: {len(test_X)}")
        
        # Create DataLoaders (unchanged from baseline)
        print("\n2. Creating data loaders...")
        train_ds = EngineDataset(train_X, train_y, TRAINING_CONFIG["window_size"])
        test_ds = EngineDataset(test_X, test_y, TRAINING_CONFIG["window_size"])
        
        train_loader = DataLoader(train_ds, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=TRAINING_CONFIG["batch_size"])
        print(f"   Train windows: {len(train_ds)}, Test windows: {len(test_ds)}")
        
        # Initialize Model (unchanged from baseline)
        print("\n3. Initializing model...")
        model = EngineHealthCNN(
            input_channels=6, 
            window_size=TRAINING_CONFIG["window_size"], 
            hidden_channels=TRAINING_CONFIG["hidden_channels"]
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        mlflow.log_param("total_parameters", total_params)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
        
        # Training (enhanced with MLflow logging)
        print(f"\n4. Training for {TRAINING_CONFIG['epochs']} epochs...")
        train_losses = train_model_with_mlflow(model, train_loader, criterion, optimizer, TRAINING_CONFIG)
        
        # Evaluation (enhanced with comprehensive metrics)
        print("\n5. Evaluating model...")
        metrics, cm, report, all_targets, all_preds = evaluate_model_with_mlflow(model, test_loader)
        
        # Print metrics
        print(f"\n{'='*60}")
        print("EVALUATION METRICS")
        print(f"{'='*60}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.upper():15s}: {metric_value:.4f}")
            mlflow.log_metric(metric_name, metric_value)
        
        print(f"\n{report}")
        
        # Create visualizations
        print("\n6. Creating visualizations...")
        cm_plot = plot_confusion_matrix(cm)
        loss_plot = plot_training_history(train_losses)
        
        # Log artifacts to MLflow
        print("\n7. Logging artifacts to MLflow...")
        mlflow.log_artifact(cm_plot)
        mlflow.log_artifact(loss_plot)
        
        # Save model artifacts
        artifact_paths = save_model_artifacts(model, scaler, TRAINING_CONFIG, metrics)
        for name, path in artifact_paths.items():
            mlflow.log_artifact(path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")
        
        # Register model to MLflow Model Registry
        print("\n8. Registering model...")
        model_uri = f"runs:/{run.info.run_id}/model"
        model_name = "BaselineCNN_EngineHealth"
        
        try:
            mv = mlflow.register_model(model_uri, model_name)
            print(f"   Model registered: {model_name} version {mv.version}")
            
            # Transition to production if good performance
            if metrics['f1_score'] > 0.85:
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Production"
                )
                print(f"   Model promoted to Production stage!")
        except Exception as e:
            print(f"   Model registration note: {e}")
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Experiment: {MLFLOW_CONFIG['experiment_name']}")
        print(f"\nView results:")
        print(f"  mlflow ui --port 5000")
        print(f"  http://localhost:5000")
        print(f"{'='*60}\n")
        
        return {
            'run_id': run.info.run_id,
            'metrics': metrics,
            'model': model
        }

# --- 11. Inference Function (NEW - Production Deployment) ---
def load_and_predict(model_path, scaler_path, config_path, new_data):
    """
    Load trained model and make predictions
    For production deployment
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load scaler
    import pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load model
    model = EngineHealthCNN(
        input_channels=6,
        window_size=config['window_size'],
        hidden_channels=config['hidden_channels']
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Prepare data
    scaled_data = scaler.transform(new_data)
    tensor_data = torch.FloatTensor(scaled_data)
    
    # Make prediction
    with torch.no_grad():
        if len(tensor_data) >= config['window_size']:
            # Take last window
            window = tensor_data[-config['window_size']:].unsqueeze(0)
            output = model(window).squeeze()
            prediction = (output > 0.5).item()
            probability = output.item()
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'condition': 'Unhealthy' if prediction else 'Healthy'
            }
        else:
            return {'error': 'Insufficient data for window'}

# --- 12. Main Entry Point ---
if __name__ == "__main__":
    import sys
    
    # Check for data file
    DATA_PATH = "data/raw/engine_data.csv"
    
    if not Path(DATA_PATH).exists():
        print(f"Error: '{DATA_PATH}' not found.")
        print("Please ensure your dataset is in data/raw/engine_data.csv")
        sys.exit(1)
    
    try:
        # Run enhanced pipeline
        results = run_experiment_with_mlflow(DATA_PATH)
        
        print("\nPipeline completed successfully!")
        print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
        print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
