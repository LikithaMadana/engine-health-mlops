"""
XGBoost model for engine health prediction.
"""

import numpy as np
import xgboost as xgb
from typing import Optional, Dict, Any


class EngineXGBoost:
    """XGBoost classifier for engine health prediction."""
    
    def __init__(self,
                 max_depth: int = 6,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize the XGBoost model.
        
        Args:
            max_depth: Maximum tree depth
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            random_state: Random seed
            **kwargs: Additional XGBoost parameters
        """
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Initialize the model
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            **kwargs
        )
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
           X_val: Optional[np.ndarray] = None,
           y_val: Optional[np.ndarray] = None,
           verbose: bool = False) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        
        # Get evaluation results
        evals_result = self.model.evals_result()
        
        return evals_result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            'max_depth': self.max_depth,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state
        }
