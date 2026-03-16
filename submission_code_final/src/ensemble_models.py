"""
Ensemble Model Stack for Mule Account Detection

Trains and manages multiple models:
1. XGBoost - Gradient boosting
2. Random Forest - Ensemble decision trees
3. Logistic Regression - Linear baseline
4. Isolation Forest - Unsupervised anomaly detection
5. Neural Network - Deep learning (optional)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import pickle
from pathlib import Path
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None


class EnsembleModelStack:
    """Train and manage ensemble of models"""
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize ensemble.
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.model_weights = {
            'xgboost': 0.3,
            'random_forest': 0.25,
            'logistic_regression': 0.2,
            'isolation_forest': 0.15,
            'neural_network': 0.1
        }
        
        logger.info("EnsembleModelStack initialized")
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: np.ndarray,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[np.ndarray] = None,
                        use_smote: bool = True) -> Dict[str, ModelMetrics]:
        """
        Train all models in ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            use_smote: Whether to use SMOTE for balancing (default True)
            
        Returns:
            Dictionary of model_name -> ModelMetrics
        """
        self.feature_names = X_train.columns.tolist()
        metrics = {}
        
        logger.info(f"Training ensemble on {len(X_train)} samples with {len(X_train.columns)} features")
        logger.info(f"Class distribution: {np.bincount(y_train.astype(int))}")
        
        # Apply SMOTE if requested and available
        X_train_balanced = X_train
        y_train_balanced = y_train
        
        if use_smote and HAS_IMBLEARN:
            logger.info("Applying SMOTE for class balancing...")
            try:
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                logger.info(f"After SMOTE: {len(X_train_balanced)} samples")
                logger.info(f"New class distribution: {np.bincount(y_train_balanced.astype(int))}")
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}. Continuing without SMOTE.")
                X_train_balanced = X_train
                y_train_balanced = y_train
        
        # Train each model
        if HAS_XGBOOST:
            metrics['xgboost'] = self.train_xgboost(X_train_balanced, y_train_balanced, X_val, y_val)
        
        metrics['random_forest'] = self.train_random_forest(X_train_balanced, y_train_balanced, X_val, y_val)
        metrics['logistic_regression'] = self.train_logistic_regression(X_train_balanced, y_train_balanced, X_val, y_val)
        metrics['isolation_forest'] = self.train_isolation_forest(X_train_balanced, y_train_balanced, X_val, y_val)
        
        if HAS_TENSORFLOW:
            metrics['neural_network'] = self.train_neural_network(X_train_balanced, y_train_balanced, X_val, y_val)
        
        logger.info(f"Ensemble training complete: {len(metrics)} models trained")
        return metrics
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: np.ndarray,
                     X_val: Optional[pd.DataFrame] = None,
                     y_val: Optional[np.ndarray] = None) -> ModelMetrics:
        """Train XGBoost model"""
        if not HAS_XGBOOST:
            logger.warning("XGBoost not available")
            return ModelMetrics()
        
        logger.info("Training XGBoost model...")
        
        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        scale_pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0
        logger.info(f"Class weight (scale_pos_weight): {scale_pos_weight:.4f}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['xgboost'] = scaler
        
        # Hyperparameter tuning
        params = {
            'max_depth': [5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight)
        grid_search = GridSearchCV(xgb_model, params, cv=3, n_jobs=-1, scoring='roc_auc')
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        self.models['xgboost'] = best_model
        
        logger.info(f"XGBoost best params: {grid_search.best_params_}")
        
        # Evaluate
        metrics = self._evaluate_model(best_model, X_train_scaled, y_train, X_val, y_val, scaler)
        
        # Feature importance
        if self.feature_names:
            metrics.feature_importance = dict(zip(
                self.feature_names,
                best_model.feature_importances_
            ))
        else:
            metrics.feature_importance = {}
        
        return metrics
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: np.ndarray,
                           X_val: Optional[pd.DataFrame] = None,
                           y_val: Optional[np.ndarray] = None) -> ModelMetrics:
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1] if len(class_weights) > 1 else 1.0}
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['random_forest'] = scaler
        
        # Hyperparameter tuning
        params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10]
        }
        
        rf_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        grid_search = GridSearchCV(rf_model, params, cv=3, n_jobs=-1, scoring='roc_auc')
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        self.models['random_forest'] = best_model
        
        logger.info(f"Random Forest best params: {grid_search.best_params_}")
        
        # Evaluate
        metrics = self._evaluate_model(best_model, X_train_scaled, y_train, X_val, y_val, scaler)
        
        # Feature importance
        if self.feature_names:
            metrics.feature_importance = dict(zip(
                self.feature_names,
                best_model.feature_importances_
            ))
        else:
            metrics.feature_importance = {}
        
        return metrics
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: np.ndarray,
                                 X_val: Optional[pd.DataFrame] = None,
                                 y_val: Optional[np.ndarray] = None) -> ModelMetrics:
        """Train Logistic Regression model"""
        logger.info("Training Logistic Regression model...")
        
        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1] if len(class_weights) > 1 else 1.0}
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['logistic_regression'] = scaler
        
        # Train model with class weights
        lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced')
        lr_model.fit(X_train_scaled, y_train)
        
        self.models['logistic_regression'] = lr_model
        
        # Evaluate
        metrics = self._evaluate_model(lr_model, X_train_scaled, y_train, X_val, y_val, scaler)
        
        # Feature importance (coefficients)
        if self.feature_names:
            metrics.feature_importance = dict(zip(
                self.feature_names,
                np.abs(lr_model.coef_[0])
            ))
        else:
            metrics.feature_importance = {}
        
        return metrics
    
    def train_isolation_forest(self, X_train: pd.DataFrame, y_train: np.ndarray,
                              X_val: Optional[pd.DataFrame] = None,
                              y_val: Optional[np.ndarray] = None) -> ModelMetrics:
        """Train Isolation Forest model"""
        logger.info("Training Isolation Forest model...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['isolation_forest'] = scaler
        
        # Train model (unsupervised)
        iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        iso_forest.fit(X_train_scaled)
        
        self.models['isolation_forest'] = iso_forest
        
        # Evaluate
        metrics = self._evaluate_model(iso_forest, X_train_scaled, y_train, X_val, y_val, scaler)
        
        return metrics
    
    def train_neural_network(self, X_train: pd.DataFrame, y_train: np.ndarray,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[np.ndarray] = None) -> ModelMetrics:
        """Train Neural Network model"""
        if not HAS_TENSORFLOW:
            logger.warning("TensorFlow not available")
            return ModelMetrics()
        
        logger.info("Training Neural Network model...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['neural_network'] = scaler
        
        # Build model
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        model.fit(
            X_train_scaled, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.models['neural_network'] = model
        
        # Evaluate
        metrics = self._evaluate_model(model, X_train_scaled, y_train, X_val, y_val, scaler)
        
        return metrics
    
    def _evaluate_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: Optional[np.ndarray] = None,
                       y_val: Optional[np.ndarray] = None,
                       scaler: Optional[StandardScaler] = None) -> ModelMetrics:
        """Evaluate model performance"""
        metrics = ModelMetrics()
        
        # Predictions on training set
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_train)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_train)
            if hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_train)
            else:
                y_pred_proba = y_pred
        
        # Calculate metrics
        metrics.accuracy = accuracy_score(y_train, y_pred)
        metrics.precision = precision_score(y_train, y_pred, zero_division=0, average='weighted')
        metrics.recall = recall_score(y_train, y_pred, zero_division=0, average='weighted')
        metrics.f1_score = f1_score(y_train, y_pred, zero_division=0, average='weighted')
        
        try:
            metrics.auc_roc = roc_auc_score(y_train, y_pred_proba)
        except:
            metrics.auc_roc = 0.0
        
        metrics.confusion_matrix = confusion_matrix(y_train, y_pred)
        
        logger.info(f"Model metrics - Accuracy: {metrics.accuracy:.3f}, Precision: {metrics.precision:.3f}, "
                   f"Recall: {metrics.recall:.3f}, F1: {metrics.f1_score:.3f}, AUC: {metrics.auc_roc:.3f}")
        
        return metrics
    
    def predict_ensemble(self, X: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions.
        
        Args:
            X: Feature dataframe
            threshold: Decision threshold for mule classification (default 0.5)
            
        Returns:
            Tuple of (predictions, confidences)
        """
        predictions = []
        
        for model_name, model in self.models.items():
            scaler = self.scalers.get(model_name)
            X_scaled = scaler.transform(X) if scaler else X
            
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_scaled)[:, 1]
            elif hasattr(model, 'decision_function'):
                pred = model.decision_function(X_scaled)
                # Normalize to 0-1
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
            else:
                pred = model.predict(X_scaled)
            
            weight = self.model_weights.get(model_name, 1.0 / len(self.models))
            predictions.append(pred * weight)
        
        # Weighted ensemble
        ensemble_pred = np.sum(predictions, axis=0)
        
        # Confidence based on model agreement
        pred_array = np.array([p / self.model_weights.get(list(self.models.keys())[i], 1.0) 
                              for i, p in enumerate(predictions)])
        confidence = 1.0 - np.std(pred_array, axis=0)
        
        logger.info(f"Ensemble predictions - Min: {ensemble_pred.min():.4f}, Max: {ensemble_pred.max():.4f}, Mean: {ensemble_pred.mean():.4f}")
        logger.info(f"Mules detected (threshold={threshold}): {(ensemble_pred >= threshold).sum()}")
        
        return ensemble_pred, confidence
    
    def find_optimal_threshold(self, X_val: pd.DataFrame, y_val: np.ndarray, 
                              metric: str = 'f1') -> float:
        """
        Find optimal prediction threshold using validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to optimize ('f1', 'precision', 'recall', 'roc_auc')
            
        Returns:
            float: Optimal threshold
        """
        logger.info(f"Finding optimal threshold using {metric} metric...")
        
        # Get predictions
        predictions, _ = self.predict_ensemble(X_val, threshold=0.5)
        
        best_threshold = 0.5
        best_score = 0.0
        
        # Try different thresholds
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (predictions >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_val, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred, zero_division=0)
            elif metric == 'roc_auc':
                try:
                    score = roc_auc_score(y_val, predictions)
                except:
                    score = 0.0
            else:
                score = f1_score(y_val, y_pred, zero_division=0)
            
            logger.info(f"Threshold {threshold:.2f}: {metric}={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.4f} with {metric}={best_score:.4f}")
        return best_threshold
    
    def save_models(self) -> None:
        """Save trained models to disk"""
        for model_name, model in self.models.items():
            model_path = self.model_dir / f"{model_name}.pkl"
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved model: {model_path}")
            except Exception as e:
                logger.error(f"Error saving {model_name}: {e}")
    
    def load_models(self) -> None:
        """Load trained models from disk"""
        for model_name in ['xgboost', 'random_forest', 'logistic_regression', 'isolation_forest', 'neural_network']:
            model_path = self.model_dir / f"{model_name}.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    logger.info(f"Loaded model: {model_path}")
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {e}")
