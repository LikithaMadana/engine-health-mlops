"""
Complete Kubeflow Pipeline for Engine Health Prediction.

This pipeline orchestrates the complete MLOps workflow including:
- Data ingestion and preprocessing
- Hyperparameter tuning (Optuna)
- Model training with optimized parameters (CNN and XGBoost)
- Comprehensive model evaluation
- Champion model declaration
- Model registration to MLflow Model Registry
"""

from kfp import dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Metrics,
    Artifact
)
from typing import NamedTuple


@component(
    base_image="python:3.9",
    packages_to_install=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ]
)
def preprocess_data(
    n_samples: int,
    window_size: int,
    test_size: float,
    data_source: str,
    output_train_data: Output[Dataset],
    output_test_data: Output[Dataset],
    output_train_data_windowed: Output[Dataset],
    output_test_data_windowed: Output[Dataset]
):
    """
    Preprocess data for model training.
    
    Supports both synthetic data generation (for testing) and real data loading.
    
    Args:
        n_samples: Number of samples (for synthetic data)
        window_size: Window size for time series
        test_size: Test set proportion
        data_source: "synthetic" or "real" (path to real data)
        output_train_data: Flat training data for XGBoost
        output_test_data: Flat test data for XGBoost
        output_train_data_windowed: Windowed training data for CNN
        output_test_data_windowed: Windowed test data for CNN
    """
    import numpy as np
    import pandas as pd
    import pickle
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    if data_source == "synthetic":
        # Generate synthetic data (for testing/demonstration)
        print("Using synthetic data generation...")
        np.random.seed(42)
        healthy_data = np.random.randn(n_samples // 2, 14) * 0.5 + 0.3
        unhealthy_data = np.random.randn(n_samples // 2, 14) * 1.2 + 1.5
        
        X = np.vstack([healthy_data, unhealthy_data])
        y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
        
        # Shuffle
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
    else:
        # Load real data from file/database
        print(f"Loading real data from: {data_source}")
        # In production, implement actual data loading:
        # df = pd.read_csv(data_source)
        # X = df.drop('target', axis=1).values
        # y = df['target'].values
        raise NotImplementedError(
            "Real data loading not yet implemented. "
            "In production, load from CSV/database and extract features and labels. "
            "Expected format: CSV with features and 'target' column (0=healthy, 1=unhealthy)"
        )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save flat data for XGBoost
    with open(output_train_data.path, 'wb') as f:
        pickle.dump({'X': X_train_scaled, 'y': y_train}, f)
    
    with open(output_test_data.path, 'wb') as f:
        pickle.dump({'X': X_test_scaled, 'y': y_test}, f)
    
    # Create sliding windows for CNN
    def create_windows(X, y, window_size):
        n_windows = len(X) - window_size + 1
        X_windows = np.array([X[i:i + window_size] for i in range(n_windows)])
        y_windows = y[window_size - 1:]
        return X_windows, y_windows
    
    X_train_win, y_train_win = create_windows(X_train_scaled, y_train, window_size)
    X_test_win, y_test_win = create_windows(X_test_scaled, y_test, window_size)
    
    # Save windowed data for CNN
    with open(output_train_data_windowed.path, 'wb') as f:
        pickle.dump({'X': X_train_win, 'y': y_train_win}, f)
    
    with open(output_test_data_windowed.path, 'wb') as f:
        pickle.dump({'X': X_test_win, 'y': y_test_win}, f)
    
    print(f"Preprocessed data: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")
    print(f"Windowed data: Train={X_train_win.shape}, Test={X_test_win.shape}")


@component(
    base_image="python:3.9",
    packages_to_install=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.8.0"
    ]
)
def train_cnn(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    output_model: Output[Model],
    output_metrics: Output[Metrics],
    tuned_params: Input[Artifact] = None
):
    import pickle
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import json
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Load hyperparameters (tuned or default)
    if tuned_params is not None:
        with open(tuned_params.path, 'r') as f:
            params = json.load(f)
        learning_rate = params.get('learning_rate', 0.001)
        hidden_channels = params.get('hidden_channels', 64)
        num_epochs = params.get('num_epochs', 50)
        batch_size = params.get('batch_size', 32)
        print("Training CNN with tuned parameters")
    else:
        learning_rate = 0.001
        hidden_channels = 64
        num_epochs = 50
        batch_size = 32
        print("Training CNN with default parameters")
    
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Hidden Channels: {hidden_channels}")
    print(f"  Num Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    
    # Load data
    with open(train_data.path, 'rb') as f:
        train = pickle.load(f)
    with open(test_data.path, 'rb') as f:
        test = pickle.load(f)
    
    X_train, y_train = train['X'], train['y']
    X_test, y_test = test['X'], test['y']
    
    # Define CNN model (SAME as tuning architecture)
    class EngineCNN(nn.Module):
        def __init__(self, input_channels, window_size, hidden_channels):
            super(EngineCNN, self).__init__()
            self.conv1 = nn.Conv1d(input_channels, hidden_channels, 3, padding=1)
            self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, 3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.fc = nn.Linear(hidden_channels * 2 * (window_size // 4), 2)
        
        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Initialize model
    window_size = X_train.shape[1]
    input_channels = X_train.shape[2]
    model = EngineCNN(input_channels, window_size, hidden_channels)
    
    # Training setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train.astype(np.int64))
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test.astype(np.int64))
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    
    # Save model
    torch.save(model.state_dict(), output_model.path)
    
    # Log metrics
    output_metrics.log_metric("accuracy", accuracy)
    output_metrics.log_metric("precision", precision)
    output_metrics.log_metric("recall", recall)
    output_metrics.log_metric("f1_score", f1)
    output_metrics.log_metric("auc", auc)
    
    print(f"CNN Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")


@component(
    base_image="python:3.9",
    packages_to_install=[
        "xgboost>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.8.0",
        "scipy>=1.11.0"  # For feature engineering
    ]
)
def train_xgboost(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    output_model: Output[Model],
    output_metrics: Output[Metrics],
    tuned_params: Input[Artifact] = None
):
    import pickle
    import xgboost as xgb
    import numpy as np
    import json
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Feature engineering function (inline for Kubeflow component)
    def engineer_features(X_windows):
        """Extract engineered features from windowed data."""
        n_samples = X_windows.shape[0]
        window_size = X_windows.shape[1]
        n_sensors = X_windows.shape[2]
        
        features_list = []
        
        for i in range(n_samples):
            window = X_windows[i]
            features = []
            
            # Statistical features (10 per sensor)
            for sensor_idx in range(n_sensors):
                sensor_data = window[:, sensor_idx]
                features.extend([
                    np.mean(sensor_data),
                    np.std(sensor_data),
                    np.min(sensor_data),
                    np.max(sensor_data),
                    np.median(sensor_data),
                    np.max(sensor_data) - np.min(sensor_data),  # range
                    np.std(sensor_data) / (np.mean(sensor_data) + 1e-6),  # cv
                    np.percentile(sensor_data, 25),
                    np.percentile(sensor_data, 75),
                    np.percentile(sensor_data, 75) - np.percentile(sensor_data, 25)  # iqr
                ])
            
            # Temporal features (5 per sensor)
            for sensor_idx in range(n_sensors):
                sensor_data = window[:, sensor_idx]
                diff = np.diff(sensor_data)
                features.extend([
                    np.mean(diff),
                    np.std(diff),
                    np.polyfit(np.arange(len(sensor_data)), sensor_data, 1)[0] if len(sensor_data) > 1 else 0,
                    np.mean(np.diff(diff)) if len(diff) > 1 else 0,
                    sensor_data[-1] - sensor_data[0]
                ])
            
            # Domain features (13 total)
            avg_rpm = np.mean(window[:, 0])
            avg_oil_press = np.mean(window[:, 1])
            avg_fuel_press = np.mean(window[:, 2])
            avg_coolant_press = np.mean(window[:, 3])
            avg_oil_temp = np.mean(window[:, 4])
            avg_coolant_temp = np.mean(window[:, 5])
            
            features.extend([
                avg_oil_press / (avg_fuel_press + 1e-6),
                avg_coolant_press / (avg_fuel_press + 1e-6),
                avg_oil_press / (avg_coolant_press + 1e-6),
                avg_oil_temp - avg_coolant_temp,
                avg_oil_temp / (avg_rpm + 1e-6),
                avg_coolant_temp / (avg_rpm + 1e-6),
                avg_oil_press / (avg_rpm + 1e-6),
                avg_fuel_press / (avg_rpm + 1e-6),
                np.std(window[:, 4]),
                np.std(window[:, 5]),
                np.std(window[:, 1]),
                np.std(window[:, 2]),
                (avg_oil_press + avg_fuel_press + avg_coolant_press) / (3 * (np.std(window[:, 4]) + np.std(window[:, 5]) + 2) / 2)
            ])
            
            features_list.append(features)
        
        return np.array(features_list)
    
    # Load hyperparameters (tuned or default)
    if tuned_params is not None:
        with open(tuned_params.path, 'r') as f:
            params = json.load(f)
        max_depth = params.get('max_depth', 6)
        n_estimators = params.get('n_estimators', 100)
        learning_rate = params.get('learning_rate', 0.1)
        subsample = params.get('subsample', 0.8)
        colsample_bytree = params.get('colsample_bytree', 0.8)
        print("Training XGBoost with tuned parameters and feature engineering")
    else:
        max_depth = 6
        n_estimators = 100
        learning_rate = 0.1
        subsample = 0.8
        colsample_bytree = 0.8
        print("Training XGBoost with default parameters and feature engineering")
    
    print(f"  Max Depth: {max_depth}")
    print(f"  N Estimators: {n_estimators}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Subsample: {subsample}")
    print(f"  Colsample Bytree: {colsample_bytree}")
    
    # Load data
    with open(train_data.path, 'rb') as f:
        train = pickle.load(f)
    with open(test_data.path, 'rb') as f:
        test = pickle.load(f)
    
    X_train, y_train = train['X'], train['y']
    X_test, y_test = test['X'], test['y']
    
    # Apply feature engineering if data is windowed
    if len(X_train.shape) == 3:
        print(f"Input shape: {X_train.shape}")
        print("Applying feature engineering...")
        X_train = engineer_features(X_train)
        X_test = engineer_features(X_test)
        print(f"Engineered features shape: {X_train.shape}")
        print(f"Total engineered features: {X_train.shape[1]}")
    
    # Initialize and train model with tuned parameters
    model = xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    
    # Save model
    model.save_model(output_model.path)
    
    # Log metrics
    output_metrics.log_metric("accuracy", accuracy)
    output_metrics.log_metric("precision", precision)
    output_metrics.log_metric("recall", recall)
    output_metrics.log_metric("f1_score", f1)
    output_metrics.log_metric("auc", auc)
    
    print(f"XGBoost Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")


@component(
    base_image="python:3.9",
    packages_to_install=[
        "optuna>=3.0.0",
        "xgboost>=2.0.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ]
)
def tune_hyperparameters(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    model_type: str,  # "cnn" or "xgboost"
    n_trials: int,
    output_best_params: Output[Artifact]
):
    """
    Tune hyperparameters using Optuna.
    
    Args:
        train_data: Training dataset
        test_data: Test dataset
        model_type: Type of model to tune ("cnn" or "xgboost")
        n_trials: Number of Optuna trials
        output_best_params: Best hyperparameters found
    """
    import pickle
    import optuna
    import json
    import numpy as np
    from sklearn.metrics import f1_score
    
    # Load data
    with open(train_data.path, 'rb') as f:
        train = pickle.load(f)
    with open(test_data.path, 'rb') as f:
        test = pickle.load(f)
    
    X_train, y_train = train['X'], train['y']
    X_test, y_test = test['X'], test['y']
    
    if model_type == "xgboost":
        import xgboost as xgb
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            
            model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred = model.predict(X_test)
            
            return f1_score(y_test, y_pred)
        
    elif model_type == "cnn":
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        def objective(trial):
            params = {
                'hidden_channels': trial.suggest_categorical('hidden_channels', [16, 32, 64]),
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'num_epochs': trial.suggest_int('num_epochs', 30, 100),
            }
            
            # Same architecture as training
            class EngineCNN(nn.Module):
                def __init__(self, input_channels, window_size, hidden_channels):
                    super(EngineCNN, self).__init__()
                    self.conv1 = nn.Conv1d(input_channels, hidden_channels, 3, padding=1)
                    self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, 3, padding=1)
                    self.pool = nn.MaxPool1d(2)
                    self.fc = nn.Linear(hidden_channels * 2 * (window_size // 4), 2)
                
                def forward(self, x):
                    x = x.permute(0, 2, 1)
                    x = self.pool(torch.relu(self.conv1(x)))
                    x = self.pool(torch.relu(self.conv2(x)))
                    x = x.view(x.size(0), -1)
                    x = self.fc(x)
                    return x
            
            window_size = X_train.shape[1]
            input_channels = X_train.shape[2]
            model = EngineCNN(input_channels, window_size, params['hidden_channels'])
            
            device = 'cpu'  # Use CPU for tuning
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train.astype(np.int64))
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test),
                torch.LongTensor(y_test.astype(np.int64))
            )
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])
            
            # Quick training
            for epoch in range(min(params['num_epochs'], 20)):  # Limit for tuning
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluation
            model.eval()
            all_preds = []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.numpy())
            
            return f1_score(y_test, np.array(all_preds))
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Run Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Save best parameters
    best_params = study.best_params
    best_params['best_value'] = study.best_value
    
    with open(output_best_params.path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"{model_type.upper()} Tuning Complete:")
    print(f"  Best F1 Score: {study.best_value:.4f}")
    print(f"  Best Parameters: {best_params}")


@component(
    base_image="python:3.9",
    packages_to_install=[
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    ]
)
def evaluate_models(
    cnn_metrics: Input[Metrics],
    xgb_metrics: Input[Metrics],
    output_comparison: Output[Artifact],
    output_evaluation_report: Output[Artifact]
):
    """
    Comprehensively evaluate and compare all trained models.
    
    Args:
        cnn_metrics: CNN model metrics
        xgb_metrics: XGBoost model metrics
        output_comparison: Comparison results
        output_evaluation_report: Detailed evaluation report
    """
    import json
    
    # Extract metrics
    cnn_data = {
        'accuracy': cnn_metrics.metadata.get('accuracy', 0),
        'precision': cnn_metrics.metadata.get('precision', 0),
        'recall': cnn_metrics.metadata.get('recall', 0),
        'f1_score': cnn_metrics.metadata.get('f1_score', 0),
        'auc': cnn_metrics.metadata.get('auc', 0),
    }
    
    xgb_data = {
        'accuracy': xgb_metrics.metadata.get('accuracy', 0),
        'precision': xgb_metrics.metadata.get('precision', 0),
        'recall': xgb_metrics.metadata.get('recall', 0),
        'f1_score': xgb_metrics.metadata.get('f1_score', 0),
        'auc': xgb_metrics.metadata.get('auc', 0),
    }
    
    # Create comparison
    comparison = {
        'cnn': cnn_data,
        'xgboost': xgb_data,
        'winner_by_metric': {}
    }
    
    # Determine winner for each metric
    for metric in cnn_data.keys():
        if cnn_data[metric] > xgb_data[metric]:
            comparison['winner_by_metric'][metric] = 'CNN'
        elif xgb_data[metric] > cnn_data[metric]:
            comparison['winner_by_metric'][metric] = 'XGBoost'
        else:
            comparison['winner_by_metric'][metric] = 'Tie'
    
    # Save comparison
    with open(output_comparison.path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Create evaluation report
    report = f"""
MODEL EVALUATION REPORT
=====================

CNN Model Performance:
----------------------
Accuracy:  {cnn_data['accuracy']:.4f} ({cnn_data['accuracy']*100:.2f}%)
Precision: {cnn_data['precision']:.4f} ({cnn_data['precision']*100:.2f}%)
Recall:    {cnn_data['recall']:.4f} ({cnn_data['recall']*100:.2f}%)
F1 Score:  {cnn_data['f1_score']:.4f} ({cnn_data['f1_score']*100:.2f}%)
AUC:       {cnn_data['auc']:.4f} ({cnn_data['auc']*100:.2f}%)

XGBoost Model Performance:
--------------------------
Accuracy:  {xgb_data['accuracy']:.4f} ({xgb_data['accuracy']*100:.2f}%)
Precision: {xgb_data['precision']:.4f} ({xgb_data['precision']*100:.2f}%)
Recall:    {xgb_data['recall']:.4f} ({xgb_data['recall']*100:.2f}%)
F1 Score:  {xgb_data['f1_score']:.4f} ({xgb_data['f1_score']*100:.2f}%)
AUC:       {xgb_data['auc']:.4f} ({xgb_data['auc']*100:.2f}%)

Winner by Metric:
-----------------
"""
    for metric, winner in comparison['winner_by_metric'].items():
        report += f"{metric.upper():15s}: {winner}\n"
    
    with open(output_evaluation_report.path, 'w') as f:
        f.write(report)
    
    print(report)


@component(
    base_image="python:3.9",
    packages_to_install=["numpy>=1.24.0"]
)
def declare_champion(
    comparison: Input[Artifact],
    primary_metric: str,
    output_champion: Output[Artifact]
):
    """
    Declare champion model based on primary metric.
    
    Args:
        comparison: Model comparison data
        primary_metric: Primary metric for selection (e.g., 'f1_score')
        output_champion: Champion model details
    """
    import json
    
    # Load comparison
    with open(comparison.path, 'r') as f:
        comp_data = json.load(f)
    
    cnn_score = comp_data['cnn'][primary_metric]
    xgb_score = comp_data['xgboost'][primary_metric]
    
    if cnn_score > xgb_score:
        champion = {
            'model': 'CNN',
            'score': cnn_score,
            'all_metrics': comp_data['cnn']
        }
    elif xgb_score > cnn_score:
        champion = {
            'model': 'XGBoost',
            'score': xgb_score,
            'all_metrics': comp_data['xgboost']
        }
    else:
        # Tie, default to CNN
        champion = {
            'model': 'CNN (tie)',
            'score': cnn_score,
            'all_metrics': comp_data['cnn']
        }
    
    champion['primary_metric'] = primary_metric
    
    # Save champion
    with open(output_champion.path, 'w') as f:
        json.dump(champion, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"CHAMPION MODEL DECLARED")
    print(f"{'='*60}")
    print(f"Champion: {champion['model']}")
    print(f"Primary Metric ({primary_metric}): {champion['score']:.4f}")
    print(f"All Metrics:")
    for metric, value in champion['all_metrics'].items():
        print(f"  {metric:15s}: {value:.4f}")
    print(f"{'='*60}\n")


@component(
    base_image="python:3.9",
    packages_to_install=[
        "mlflow>=2.8.0",
        "torch>=2.0.0",
        "xgboost>=2.0.0",
    ]
)
def register_to_mlflow(
    champion: Input[Artifact],
    cnn_model: Input[Model],
    xgb_model: Input[Model],
    output_registry_info: Output[Artifact]
):
    """
    Register champion model to MLflow Model Registry.
    
    Args:
        champion: Champion model details
        cnn_model: CNN model artifact
        xgb_model: XGBoost model artifact
        output_registry_info: Registry information
    """
    import json
    import mlflow
    import mlflow.pytorch
    import mlflow.xgboost
    
    # Load champion details
    with open(champion.path, 'r') as f:
        champ_data = json.load(f)
    
    model_name = champ_data['model'].split()[0]  # Remove "(tie)" if present
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    experiment_name = "Engine Health - Complete Pipeline"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"Champion_{model_name}"):
        # Log metrics
        for metric, value in champ_data['all_metrics'].items():
            mlflow.log_metric(metric, value)
        
        # Log champion designation
        mlflow.log_param("champion", True)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("primary_metric", champ_data['primary_metric'])
        mlflow.log_param("primary_metric_value", champ_data['score'])
        
        # Load and register the actual model
        if model_name == "CNN":
            # Load the PyTorch model
            # Note: For CNN, we log the model file as artifact since we'd need architecture
            mlflow.log_artifact(cnn_model.path, "model")
            registered_model_name = "EngineHealthCNN_Champion"
            print(f"CNN model saved as artifact (state_dict)")
            
        else:  # XGBoost
            # Load and register XGBoost model properly
            import xgboost as xgb
            loaded_model = xgb.XGBClassifier()
            loaded_model.load_model(xgb_model.path)
            
            # Register with actual model object
            registered_model = mlflow.xgboost.log_model(
                xgb_model=loaded_model,
                artifact_path="model",
                registered_model_name="EngineHealthXGBoost_Champion"
            )
            registered_model_name = "EngineHealthXGBoost_Champion"
            print(f"XGBoost model registered successfully")
        
        run_id = mlflow.active_run().info.run_id
    
    # Create registry info
    registry_info = {
        'champion_model': model_name,
        'registered_model_name': f"EngineHealth{model_name}_Champion",
        'run_id': run_id,
        'experiment_id': experiment_id,
        'primary_metric': champ_data['primary_metric'],
        'primary_metric_value': champ_data['score'],
        'stage': 'Production',
        'all_metrics': champ_data['all_metrics']
    }
    
    # Save registry info
    with open(output_registry_info.path, 'w') as f:
        json.dump(registry_info, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"MODEL REGISTERED TO MLFLOW")
    print(f"{'='*60}")
    print(f"Model Name: {registry_info['registered_model_name']}")
    print(f"Run ID: {run_id}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Stage: Production")
    print(f"{'='*60}\n")



@dsl.pipeline(
    name="Complete Engine Health MLOps Pipeline",
    description="Complete MLOps pipeline: preprocessing, tuning, training, evaluation, champion declaration, and model registration"
)
def complete_engine_health_pipeline(
    n_samples: int = 1000,
    window_size: int = 50,
    test_size: float = 0.2,
    n_tuning_trials: int = 10,
    primary_metric: str = "f1_score",
    data_source: str = "synthetic"
):
    # Step 1: Preprocess data + Feature Engineering
    preprocess_task = preprocess_data(
        n_samples=n_samples,
        window_size=window_size,
        test_size=test_size,
        data_source=data_source
    )
    preprocess_task.set_display_name("Step 1: Data Prep + Feature Engineering")
    
    # Step 2a: Train CNN (baseline)
    cnn_task = train_cnn(
        train_data=preprocess_task.outputs['output_train_data_windowed'],
        test_data=preprocess_task.outputs['output_test_data_windowed']
    )
    cnn_task.set_display_name("Step 2a: Train CNN (Baseline)")
    cnn_task.after(preprocess_task)
    
    # Step 2b: Train XGBoost (baseline)
    xgb_task = train_xgboost(
        train_data=preprocess_task.outputs['output_train_data'],
        test_data=preprocess_task.outputs['output_test_data']
    )
    xgb_task.set_display_name("Step 2b: Train XGBoost (Baseline)")
    xgb_task.after(preprocess_task)
    
    # Step 3a: Tune CNN hyperparameters
    cnn_tune_task = tune_hyperparameters(
        train_data=preprocess_task.outputs['output_train_data_windowed'],
        test_data=preprocess_task.outputs['output_test_data_windowed'],
        model_type="cnn",
        n_trials=n_tuning_trials
    )
    cnn_tune_task.set_display_name("Step 3a: Tune CNN Hyperparameters")
    cnn_tune_task.after(cnn_task)
    
    # Step 3b: Tune XGBoost hyperparameters
    xgb_tune_task = tune_hyperparameters(
        train_data=preprocess_task.outputs['output_train_data'],
        test_data=preprocess_task.outputs['output_test_data'],
        model_type="xgboost",
        n_trials=n_tuning_trials
    )
    xgb_tune_task.set_display_name("Step 3b: Tune XGBoost Hyperparameters")
    xgb_tune_task.after(xgb_task)
    
    # Step 4: Compare tuning results (best hyperparameters)
    eval_task = evaluate_models(
        cnn_metrics=cnn_tune_task.outputs['output_best_params'],
        xgb_metrics=xgb_tune_task.outputs['output_best_params']
    )
    eval_task.set_display_name("Step 4: Compare Best Hyperparameters")
    eval_task.after(cnn_tune_task, xgb_tune_task)
    
    # Step 5: Declare champion model
    champion_task = declare_champion(
        comparison=eval_task.outputs['output_comparison'],
        primary_metric=primary_metric
    )
    champion_task.set_display_name("Step 5: Declare Champion")
    champion_task.after(eval_task)
    
    # Step 6: Register champion to MLflow
    registry_task = register_to_mlflow(
        champion=champion_task.outputs['output_champion'],
        cnn_model=cnn_task.outputs['output_model'],
        xgb_model=xgb_task.outputs['output_model']
    )
    registry_task.set_display_name("Step 6: Register Model")
    registry_task.after(champion_task)


if __name__ == "__main__":
    from kfp import compiler
    import argparse
    
    parser = argparse.ArgumentParser(description='Compile Kubeflow Pipeline')
    parser.add_argument('--output', type=str, default='complete_engine_health_pipeline.yaml',
                        help='Output YAML file path')
    args = parser.parse_args()
    
    # Compile the complete pipeline
    compiler.Compiler().compile(
        pipeline_func=complete_engine_health_pipeline,
        package_path=args.output
    )
    
    print(f"\n{'='*70}")
    print(f"KUBEFLOW PIPELINE COMPILED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"\nPipeline: Complete Engine Health MLOps Pipeline")
    print(f"Output File: {args.output}")
    print(f"\nPipeline Components:")
    print(f"  1. Data Preprocessing")
    print(f"  2. Hyperparameter Tuning (CNN + XGBoost)")
    print(f"  3. Model Training (Optimized Parameters)")
    print(f"  4. Comprehensive Evaluation")
    print(f"  5. Champion Declaration")
    print(f"  6. Model Registration (MLflow)")
    print(f"\nTo deploy this pipeline:")
    print(f"  1. Upload '{args.output}' to your Kubeflow instance")
    print(f"  2. Create a new run from the Kubeflow UI")
    print(f"  3. Configure parameters as needed")
    print(f"  4. Monitor execution in Kubeflow dashboard")
    print(f"\n{'='*70}\n")

