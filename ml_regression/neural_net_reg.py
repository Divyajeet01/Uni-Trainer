"""
Neural Network Regression Module for Uni Trainer
Uses PyTorch for neural network implementation
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


class NeuralNetModel(nn.Module):
    """Simple feedforward neural network for regression."""

    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.2):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class NeuralNetRegressor:
    """Neural Network Regression class for tabular data."""

    # Default configuration
    DEFAULT_CONFIG = {
        "hidden_dims": [128, 64, 32],
        "dropout": 0.2,
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
        "patience": 10,
        "random_state": 42,
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Neural Net Regressor.

        Args:
            config: Configuration dictionary for NN parameters
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = ""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set random seed
        torch.manual_seed(self.config["random_state"])
        np.random.seed(self.config["random_state"])

        print(f"Using device: {self.device}")

    def preprocess_data(self, df: pd.DataFrame, target_column: str,
                        fit_preprocessors: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the dataframe for training/prediction.

        Args:
            df: Input dataframe
            target_column: Name of target column
            fit_preprocessors: Whether to fit the scalers/encoders

        Returns:
            Tuple of (X, y) arrays
        """
        df = df.copy()
        self.target_name = target_column

        # Separate features and target
        y = df[target_column].values.astype(np.float32)
        X_df = df.drop(columns=[target_column])

        if fit_preprocessors:
            self.feature_names = list(X_df.columns)
            self.label_encoders = {}

        # Handle categorical columns
        for col in X_df.columns:
            if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
                if fit_preprocessors:
                    le = LabelEncoder()
                    X_df[col] = le.fit_transform(X_df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        X_df[col] = self.label_encoders[col].transform(X_df[col].astype(str))

        # Handle missing values
        X_df = X_df.fillna(X_df.median(numeric_only=True))

        X = X_df.values.astype(np.float32)

        # Scale features
        if fit_preprocessors:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the Neural Network model.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Training results dictionary
        """
        print(f"\n{'='*50}")
        print("Starting Neural Network Regression Training")
        print(f"{'='*50}")
        print(f"Training samples: {len(X)}")
        if X_val is not None:
            print(f"Validation samples: {len(X_val)}")
        print(f"Features: {X.shape[1]}")
        print(f"Device: {self.device}")
        print(f"Hidden layers: {self.config['hidden_dims']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"{'='*50}\n")

        # Create model
        input_dim = X.shape[1]
        self.model = NeuralNetModel(
            input_dim=input_dim,
            hidden_dims=self.config["hidden_dims"],
            dropout=self.config["dropout"]
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=0.5, patience=5)

        # Create data loaders
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"],
                                  shuffle=True)

        # Initialize validation tensors
        X_val_tensor = None
        y_val_tensor = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.config["epochs"]):
            # Training
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_loader)
            train_losses.append(train_loss)

            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                val_losses.append(val_loss)

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.config['epochs']} - "
                          f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

                if patience_counter >= self.config["patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.config['epochs']} - Train Loss: {train_loss:.4f}")

        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_pred = self.model(X_tensor).cpu().numpy()

        # Calculate metrics
        train_metrics = {
            "mse": mean_squared_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
        }

        results = {"train": train_metrics, "train_losses": train_losses}

        if X_val is not None and y_val is not None:
            with torch.no_grad():
                y_val_pred = self.model(X_val_tensor).cpu().numpy()
            val_metrics = {
                "mse": mean_squared_error(y_val, y_val_pred),
                "rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
                "mae": mean_absolute_error(y_val, y_val_pred),
                "r2": r2_score(y_val, y_val_pred),
            }
            results["val"] = val_metrics
            results["val_losses"] = val_losses

        print(f"\n{'='*50}")
        print("Training Complete!")
        print(f"Train RMSE: {train_metrics['rmse']:.4f}")
        print(f"Train R²: {train_metrics['r2']:.4f}")
        if "val" in results:
            print(f"Val RMSE: {results['val']['rmse']:.4f}")
            print(f"Val R²: {results['val']['r2']:.4f}")
        print(f"{'='*50}\n")

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance using gradient-based method.
        Note: For neural networks, this is an approximation.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Use absolute weights from first layer as importance proxy
        first_layer = None
        for module in self.model.network.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break

        if first_layer is not None:
            weights = first_layer.weight.data.cpu().numpy()
            importance = np.abs(weights).mean(axis=0)
            importance = importance / importance.sum()  # Normalize
            return dict(zip(self.feature_names, importance))

        return {name: 1.0 / len(self.feature_names) for name in self.feature_names}

    def save(self, path: str):
        """Save model and preprocessors."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': len(self.feature_names),
            'hidden_dims': self.config['hidden_dims'],
            'dropout': self.config['dropout'],
        }, path / "model.pt")

        # Save scaler
        joblib.dump(self.scaler, path / "scaler.joblib")

        # Save label encoders
        joblib.dump(self.label_encoders, path / "label_encoders.joblib")

        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "config": self.config,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save args.yaml for compatibility
        self._save_args_yaml(path)

        print(f"Model saved to: {path}")

    def _save_args_yaml(self, path: Path):
        """Save configuration as args.yaml."""
        import yaml

        args = {
            "model": "NeuralNetwork",
            "task": "regression",
            "hidden_dims": self.config["hidden_dims"],
            "dropout": self.config["dropout"],
            "learning_rate": self.config["learning_rate"],
            "epochs": self.config["epochs"],
            "batch_size": self.config["batch_size"],
            "random_state": self.config["random_state"],
            "feature_names": self.feature_names,
            "target_name": self.target_name,
        }

        with open(path / "args.yaml", "w") as f:
            yaml.dump(args, f, default_flow_style=False)

    def load(self, path: str):
        """Load model and preprocessors."""
        path = Path(path)

        # Load model
        checkpoint = torch.load(path / "model.pt", map_location=self.device)
        self.model = NeuralNetModel(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            dropout=checkpoint['dropout']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load scaler
        self.scaler = joblib.load(path / "scaler.joblib")

        # Load label encoders
        self.label_encoders = joblib.load(path / "label_encoders.joblib")

        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.feature_names = metadata["feature_names"]
        self.target_name = metadata["target_name"]
        self.config = metadata["config"]

        print(f"Model loaded from: {path}")


def get_next_run_folder(base_path: str, prefix: str = "neuralnet") -> str:
    """Get the next available run folder (neuralnet1, neuralnet2, etc.)."""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # Find existing folders
    existing = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith(prefix)]

    if not existing:
        return str(base_path / f"{prefix}1")

    # Get max number
    max_num = 0
    for folder in existing:
        try:
            num = int(folder.name.replace(prefix, ""))
            max_num = max(max_num, num)
        except ValueError:
            continue

    return str(base_path / f"{prefix}{max_num + 1}")


def train_with_user_data(data_path: str, config: Dict[str, Any]) -> Tuple[str, Dict]:
    """
    Train Neural Network with user-provided CSV data.

    Args:
        data_path: Path to CSV file
        config: Training configuration containing:
            - target_column: Name of target column
            - epochs: Number of training epochs
            - batch_size: Batch size
            - learning_rate: Learning rate
            - hidden_dims: List of hidden layer dimensions
            - test_size: Test split ratio (default 0.2)

    Returns:
        Tuple of (model_path, results)
    """
    print(f"\n{'='*50}")
    print("Neural Network Regression Training")
    print(f"{'='*50}")

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Get target column
    target_column = config.get("target_column")
    if not target_column or target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data. Available: {list(df.columns)}")

    # Prepare NN config
    nn_config = {
        "hidden_dims": config.get("hidden_dims", [128, 64, 32]),
        "dropout": config.get("dropout", 0.2),
        "learning_rate": config.get("learning_rate", 0.001),
        "epochs": config.get("epochs", 100),
        "batch_size": config.get("batch_size", 32),
        "patience": config.get("patience", 10),
        "random_state": 42,
    }

    # Initialize model
    regressor = NeuralNetRegressor(nn_config)

    # Preprocess
    X, y = regressor.preprocess_data(df, target_column, fit_preprocessors=True)

    # Split data
    test_size = config.get("test_size", 0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Train
    results = regressor.train(X_train, y_train, X_val, y_val)

    # Get feature importance
    feature_importance = regressor.get_feature_importance()
    results["feature_importance"] = feature_importance

    # Save model
    project_dir = config.get("project", "runs/regression")
    model_path = get_next_run_folder(project_dir, "neuralnet")
    regressor.save(model_path)

    # Save results (convert train/val losses lists for JSON)
    with open(Path(model_path) / "results.json", "w") as f:
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                     for k, v in value.items()}
            elif isinstance(value, list):
                json_results[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v
                                     for v in value]
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Model saved to: {model_path}")
    print(f"{'='*50}\n")

    return model_path, results

