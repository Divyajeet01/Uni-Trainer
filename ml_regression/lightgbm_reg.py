"""
LightGBM Regression Module for Uni Trainer
"""
import json
import joblib
import numpy as np
import pandas as pd
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Try to import lightgbm, install if not available
try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


class LightGBMRegressor:
    """LightGBM Regression class for tabular data."""

    # Default configuration
    DEFAULT_CONFIG = {
        "n_estimators": 100,
        "max_depth": -1,  # -1 means no limit
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0,
        "reg_lambda": 0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 1,
        "boosting_type": "gbdt",
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LightGBM Regressor.

        Args:
            config: Configuration dictionary for LightGBM parameters
        """

        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = ""

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
        y = df[target_column].values
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

        X = X_df.values

        # Scale features
        if fit_preprocessors:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              early_stopping_rounds: int = 10) -> Dict[str, Any]:
        """
        Train the LightGBM model.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features
            y_val: Validation targets
            early_stopping_rounds: Early stopping patience

        Returns:
            Training results dictionary
        """
        print(f"\n{'='*50}")
        print("Starting LightGBM Regression Training")
        print(f"{'='*50}")
        print(f"Training samples: {len(X)}")
        if X_val is not None:
            print(f"Validation samples: {len(X_val)}")
        print(f"Features: {X.shape[1]}")
        print(f"Parameters: {json.dumps(self.config, indent=2)}")
        print(f"{'='*50}\n")

        # Create model
        self.model = lgb.LGBMRegressor(**self.config)

        # Prepare callbacks
        callbacks = []
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
            callbacks.append(lgb.log_evaluation(period=10))

        # Train
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X, y,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None
        )

        # Get predictions
        y_pred = self.model.predict(X)

        # Calculate metrics
        train_metrics = {
            "mse": mean_squared_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
        }

        results = {"train": train_metrics}

        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_metrics = {
                "mse": mean_squared_error(y_val, y_val_pred),
                "rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
                "mae": mean_absolute_error(y_val, y_val_pred),
                "r2": r2_score(y_val, y_val_pred),
            }
            results["val"] = val_metrics

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
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

    def save(self, path: str):
        """Save model and preprocessors."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.model, path / "model.joblib")

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
            "model": "LightGBM",
            "task": "regression",
            "n_estimators": self.config["n_estimators"],
            "max_depth": self.config["max_depth"],
            "learning_rate": self.config["learning_rate"],
            "num_leaves": self.config["num_leaves"],
            "subsample": self.config["subsample"],
            "colsample_bytree": self.config["colsample_bytree"],
            "min_child_samples": self.config["min_child_samples"],
            "boosting_type": self.config["boosting_type"],
            "random_state": self.config["random_state"],
            "feature_names": self.feature_names,
            "target_name": self.target_name,
        }

        with open(path / "args.yaml", "w") as f:
            yaml.dump(args, f, default_flow_style=False)

    def load(self, path: str):
        """Load model and preprocessors."""
        path = Path(path)

        self.model = joblib.load(path / "model.joblib")
        self.scaler = joblib.load(path / "scaler.joblib")
        self.label_encoders = joblib.load(path / "label_encoders.joblib")

        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.feature_names = metadata["feature_names"]
        self.target_name = metadata["target_name"]
        self.config = metadata["config"]

        print(f"Model loaded from: {path}")


def get_next_run_folder(base_path: str, prefix: str = "lightgbm") -> str:
    """Get the next available run folder (lightgbm1, lightgbm2, etc.)."""
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
    Train LightGBM with user-provided CSV data.

    Args:
        data_path: Path to CSV file
        config: Training configuration containing:
            - target_column: Name of target column
            - n_estimators: Number of trees
            - max_depth: Max tree depth
            - learning_rate: Learning rate
            - test_size: Test split ratio (default 0.2)

    Returns:
        Tuple of (model_path, results)
    """
    print(f"\n{'='*50}")
    print("LightGBM Regression Training")
    print(f"{'='*50}")

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Get target column
    target_column = config.get("target_column")
    if not target_column or target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data. Available: {list(df.columns)}")

    # Prepare LightGBM config
    lgb_config = {
        "n_estimators": config.get("n_estimators", 100),
        "max_depth": config.get("max_depth", -1),
        "learning_rate": config.get("learning_rate", 0.1),
        "num_leaves": config.get("num_leaves", 31),
        "subsample": config.get("subsample", 0.8),
        "colsample_bytree": config.get("colsample_bytree", 0.8),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 1,
    }

    # Initialize model
    regressor = LightGBMRegressor(lgb_config)

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
    model_path = get_next_run_folder(project_dir, "lightgbm")
    regressor.save(model_path)

    # Save results
    with open(Path(model_path) / "results.json", "w") as f:
        # Convert numpy types to python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                     for k, v in value.items()}
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Model saved to: {model_path}")
    print(f"{'='*50}\n")

    return model_path, results

