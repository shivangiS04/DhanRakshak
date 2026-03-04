"""
Cross-Validation Evaluation Module

Implements k-fold cross-validation for honest model performance estimation.
Provides unbiased AUC, F1, and other metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_curve, auc
)
import logging
from typing import Dict, List, Tuple
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class CrossValidationEvaluator:
    """Performs k-fold cross-validation evaluation."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """Initialize with number of folds."""
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_results = {}
        
    def evaluate_xgboost(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Evaluate XGBoost with k-fold CV."""
        logger.info(f"Evaluating XGBoost with {self.n_splits}-fold CV...")
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        fold_results = {
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'accuracy': []
        }
        
        fold_num = 0
        for train_idx, val_idx in skf.split(X, y):
            fold_num += 1
            logger.info(f"  Fold {fold_num}/{self.n_splits}...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            scale_pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0
            
            model = xgb.XGBClassifier(
                max_depth=5, learning_rate=0.01, n_estimators=100,
                random_state=42, scale_pos_weight=scale_pos_weight, n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on validation fold
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            fold_results['auc'].append(roc_auc_score(y_val, y_pred_proba))
            fold_results['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            fold_results['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            fold_results['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            fold_results['accuracy'].append(accuracy_score(y_val, y_pred))
        
        # Aggregate results
        results = {
            'model': 'XGBoost',
            'auc_mean': np.mean(fold_results['auc']),
            'auc_std': np.std(fold_results['auc']),
            'auc_folds': fold_results['auc'],
            'f1_mean': np.mean(fold_results['f1']),
            'f1_std': np.std(fold_results['f1']),
            'f1_folds': fold_results['f1'],
            'precision_mean': np.mean(fold_results['precision']),
            'precision_std': np.std(fold_results['precision']),
            'recall_mean': np.mean(fold_results['recall']),
            'recall_std': np.std(fold_results['recall']),
            'accuracy_mean': np.mean(fold_results['accuracy']),
            'accuracy_std': np.std(fold_results['accuracy']),
        }
        
        return results
    
    def evaluate_random_forest(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Evaluate Random Forest with k-fold CV."""
        logger.info(f"Evaluating Random Forest with {self.n_splits}-fold CV...")
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        fold_results = {
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'accuracy': []
        }
        
        fold_num = 0
        for train_idx, val_idx in skf.split(X, y):
            fold_num += 1
            logger.info(f"  Fold {fold_num}/{self.n_splits}...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train
            model = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                random_state=42, class_weight='balanced', n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on validation fold
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            fold_results['auc'].append(roc_auc_score(y_val, y_pred_proba))
            fold_results['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            fold_results['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            fold_results['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            fold_results['accuracy'].append(accuracy_score(y_val, y_pred))
        
        # Aggregate results
        results = {
            'model': 'Random Forest',
            'auc_mean': np.mean(fold_results['auc']),
            'auc_std': np.std(fold_results['auc']),
            'auc_folds': fold_results['auc'],
            'f1_mean': np.mean(fold_results['f1']),
            'f1_std': np.std(fold_results['f1']),
            'f1_folds': fold_results['f1'],
            'precision_mean': np.mean(fold_results['precision']),
            'precision_std': np.std(fold_results['precision']),
            'recall_mean': np.mean(fold_results['recall']),
            'recall_std': np.std(fold_results['recall']),
            'accuracy_mean': np.mean(fold_results['accuracy']),
            'accuracy_std': np.std(fold_results['accuracy']),
        }
        
        return results
    
    def evaluate_logistic_regression(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Evaluate Logistic Regression with k-fold CV."""
        logger.info(f"Evaluating Logistic Regression with {self.n_splits}-fold CV...")
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        fold_results = {
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'accuracy': []
        }
        
        fold_num = 0
        for train_idx, val_idx in skf.split(X, y):
            fold_num += 1
            logger.info(f"  Fold {fold_num}/{self.n_splits}...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train
            model = LogisticRegression(
                max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on validation fold
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            fold_results['auc'].append(roc_auc_score(y_val, y_pred_proba))
            fold_results['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            fold_results['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            fold_results['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            fold_results['accuracy'].append(accuracy_score(y_val, y_pred))
        
        # Aggregate results
        results = {
            'model': 'Logistic Regression',
            'auc_mean': np.mean(fold_results['auc']),
            'auc_std': np.std(fold_results['auc']),
            'auc_folds': fold_results['auc'],
            'f1_mean': np.mean(fold_results['f1']),
            'f1_std': np.std(fold_results['f1']),
            'f1_folds': fold_results['f1'],
            'precision_mean': np.mean(fold_results['precision']),
            'precision_std': np.std(fold_results['precision']),
            'recall_mean': np.mean(fold_results['recall']),
            'recall_std': np.std(fold_results['recall']),
            'accuracy_mean': np.mean(fold_results['accuracy']),
            'accuracy_std': np.std(fold_results['accuracy']),
        }
        
        return results
    
    def evaluate_all_models(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Dict]:
        """Evaluate all models with k-fold CV."""
        logger.info(f"\n{'='*80}")
        logger.info(f"CROSS-VALIDATION EVALUATION ({self.n_splits}-FOLD)")
        logger.info(f"{'='*80}\n")
        
        all_results = {}
        
        all_results['xgboost'] = self.evaluate_xgboost(X, y)
        all_results['random_forest'] = self.evaluate_random_forest(X, y)
        all_results['logistic_regression'] = self.evaluate_logistic_regression(X, y)
        
        self.cv_results = all_results
        return all_results
    
    def print_results(self):
        """Print CV results in a nice format."""
        logger.info(f"\n{'='*80}")
        logger.info("CROSS-VALIDATION RESULTS")
        logger.info(f"{'='*80}\n")
        
        for model_name, results in self.cv_results.items():
            logger.info(f"{results['model']}:")
            logger.info(f"  AUC:       {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
            logger.info(f"  F1:        {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
            logger.info(f"  Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
            logger.info(f"  Recall:    {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
            logger.info(f"  Accuracy:  {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
            logger.info(f"  Folds:     {results['auc_folds']}\n")
        
        logger.info(f"{'='*80}\n")
    
    def get_summary(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        rows = []
        for model_name, results in self.cv_results.items():
            rows.append({
                'Model': results['model'],
                'AUC (mean)': results['auc_mean'],
                'AUC (std)': results['auc_std'],
                'F1 (mean)': results['f1_mean'],
                'F1 (std)': results['f1_std'],
                'Precision (mean)': results['precision_mean'],
                'Recall (mean)': results['recall_mean'],
                'Accuracy (mean)': results['accuracy_mean'],
            })
        
        return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_redundant=5, weights=[0.9, 0.1], random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    
    evaluator = CrossValidationEvaluator(n_splits=5)
    results = evaluator.evaluate_all_models(X, y)
    evaluator.print_results()
    
    summary = evaluator.get_summary()
    print(summary)
