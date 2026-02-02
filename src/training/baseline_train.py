import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# --- 1. Configuration ---
# In a real MLOps pipeline, these would be passed via CLI arguments or environment variables

LOADING_CONFIG = {
    "test_size": 0.2
}

TRAINING_CONFIG = {
    "window_size": 10,
    "step": 1,
    "learning_rate": 0.001,
    "epochs": 1,
    "batch_size": 32,
    "hidden_channels": 16
}

# --- 2. Dataset Level Windowing ---
class EngineDataset(Dataset):
    def __init__(self, features, targets, window_size):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.window_size = window_size

    def __len__(self):
        # The number of available windows
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        # Slice the window: (window_size, n_features)
        x = self.features[idx : idx + self.window_size]
        # Target is the scalar value at the last index of the window
        y = self.targets[idx + self.window_size - 1]
        return x, y

# --- 3. Data Preparation ---
def prepare_data(filepath, config):
    df = pd.read_csv(filepath)
    
    target_col = 'Engine Condition'
    feature_cols = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 
                    'Coolant pressure', 'lub oil temp', 'Coolant temp']
    
    # 1. Standardization
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 2. Split before windowing
    train_df, test_df = train_test_split(df, test_size=config["test_size"], random_state=42, shuffle=False)
    
    return train_df[feature_cols].values, train_df[target_col].values, \
           test_df[feature_cols].values, test_df[target_col].values

# --- 4. Model Architecture ---
class EngineHealthCNN(nn.Module):
    def __init__(self, input_channels, window_size, hidden_channels):
        super(EngineHealthCNN, self).__init__()
        # Conv1d expects (Batch, Channels, Length)
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            # Linear input size: (window_size - kernel_reduction) * channels
            nn.Linear((hidden_channels * 2) * (window_size - 4), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input x is (Batch, Window, Features) -> needs (Batch, Features, Window)
        x = x.transpose(1, 2)
        return self.conv_stack(x)

# --- 5. Main Execution ---
def run_experiment(data_path):
    # Prepare data
    train_X, train_y, test_X, test_y = prepare_data(data_path, LOADING_CONFIG)
    
    # Create DataLoaders
    train_ds = EngineDataset(train_X, train_y, TRAINING_CONFIG["window_size"])
    test_ds = EngineDataset(test_X, test_y, TRAINING_CONFIG["window_size"])
    
    train_loader = DataLoader(train_ds, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=TRAINING_CONFIG["batch_size"])

    # Initialize Model
    model = EngineHealthCNN(input_channels=6, 
                            window_size=TRAINING_CONFIG["window_size"], 
                            hidden_channels=TRAINING_CONFIG["hidden_channels"])
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])

    # Training Loop
    print(f"Starting training for {TRAINING_CONFIG['epochs']} epochs...")
    model.train()
    for epoch in range(TRAINING_CONFIG["epochs"]):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']} - Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze()
            # Handle potential scalar if batch size is 1
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_y.cpu().numpy().tolist())

    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    
    print(f"\n--- Metrics (Window Size: {TRAINING_CONFIG['window_size']}) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

if __name__ == "__main__":
    DATA_PATH = "engine_data.csv" 
    try:
        run_experiment(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: '{DATA_PATH}' not found. Please download it from Kaggle.")