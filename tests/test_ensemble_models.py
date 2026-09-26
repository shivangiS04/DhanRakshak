"""
Unit tests for ensemble_models module
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from src.ensemble_models import EnsembleModelStack, ModelMetrics


class TestModelMetrics:
    """Test ModelMetrics dataclass"""
    
    def test_initialization(self):
        """Test ModelMetrics initialization"""
        metrics = ModelMetrics()
        
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.auc_roc == 0.0


class TestEnsembleModelStack:
    """Test EnsembleModelStack class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=42
        )
        
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        
        return X_df, y
    
    def test_initialization(self):
        """Test EnsembleModelStack initialization"""
        ensemble = EnsembleModelStack()
        
        assert ensemble.models == {}
        assert ensemble.model_weights is not None
        assert len(ensemble.model_weights) == 5
    
    def test_model_weights_sum(self):
        """Test that model weights sum to 1"""
        ensemble = EnsembleModelStack()
        
        total_weight = sum(ensemble.model_weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_train_logistic_regression(self, sample_data):
        """Test training Logistic Regression model"""
        X, y = sample_data
        ensemble = EnsembleModelStack()
        
        metrics = ensemble.train_logistic_regression(X, y)
        
        assert metrics.accuracy >= 0
        assert metrics.precision >= 0
        assert metrics.recall >= 0
        assert 'logistic_regression' in ensemble.models
        assert metrics.feature_importance is not None
    
    def test_train_random_forest(self, sample_data):
        """Test training Random Forest model"""
        X, y = sample_data
        ensemble = EnsembleModelStack()
        
        metrics = ensemble.train_random_forest(X, y)
        
        assert metrics.accuracy >= 0
        assert 'random_forest' in ensemble.models
        assert metrics.feature_importance is not None
    
    def test_train_isolation_forest(self, sample_data):
        """Test training Isolation Forest model"""
        X, y = sample_data
        ensemble = EnsembleModelStack()
        
        metrics = ensemble.train_isolation_forest(X, y)
        
        assert metrics.accuracy >= 0
        assert 'isolation_forest' in ensemble.models
    
    def test_predict_ensemble(self, sample_data):
        """Test ensemble prediction"""
        X, y = sample_data
        ensemble = EnsembleModelStack()
        
        # Train models
        ensemble.train_logistic_regression(X, y)
        ensemble.train_random_forest(X, y)
        
        # Make predictions
        predictions, confidences = ensemble.predict_ensemble(X)
        
        assert len(predictions) == len(X)
        assert len(confidences) == len(X)
        assert all(0 <= p <= 1 for p in predictions)
        assert all(0 <= c <= 1 for c in confidences)
    
    def test_prediction_bounds(self, sample_data):
        """Test that predictions are within valid bounds"""
        X, y = sample_data
        ensemble = EnsembleModelStack()
        
        ensemble.train_logistic_regression(X, y)
        predictions, confidences = ensemble.predict_ensemble(X)
        
        assert all(0 <= p <= 1 for p in predictions)
        assert all(0 <= c <= 1 for c in confidences)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
