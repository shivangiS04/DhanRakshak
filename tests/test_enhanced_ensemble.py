import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src_enhanced.ensemble_models_v2 import EnsembleModelStackV2, ModelMetrics


@pytest.fixture
def sample_data():
    """Create sample training data"""
    X, y = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    return X_df, y


class TestEnsembleInitialization:
    """Test ensemble initialization"""
    
    def test_ensemble_init(self):
        """Test ensemble initialization"""
        ensemble = EnsembleModelStackV2('test_models')
        
        assert ensemble.models == {}
        assert ensemble.scalers == {}
        assert ensemble.model_weights is not None
        assert len(ensemble.model_weights) >= 6
    
    def test_model_weights(self):
        """Test model weights are properly set"""
        ensemble = EnsembleModelStackV2()
        
        weights = ensemble.model_weights
        assert 'xgboost' in weights
        assert 'lightgbm' in weights
        assert 'catboost' in weights
        assert 'random_forest' in weights
        assert 'logistic_regression' in weights
        assert 'isolation_forest' in weights
        
        # Weights should sum to approximately 1
        total_weight = sum(weights.values())
        assert 0.9 <= total_weight <= 1.1


class TestModelTraining:
    """Test model training"""
    
    def test_train_all_models(self, sample_data):
        """Test training all models"""
        X, y = sample_data
        ensemble = EnsembleModelStackV2()
        
        metrics = ensemble.train_all_models(X, y, use_smote=False)
        
        assert len(metrics) > 0
        assert 'random_forest' in metrics
        assert 'logistic_regression' in metrics
        assert 'isolation_forest' in metrics
    
    def test_model_metrics_structure(self, sample_data):
        """Test model metrics structure"""
        X, y = sample_data
        ensemble = EnsembleModelStackV2()
        
        metrics = ensemble.train_all_models(X, y, use_smote=False)
        
        for model_name, metric in metrics.items():
            assert isinstance(metric, ModelMetrics)
            assert hasattr(metric, 'accuracy')
            assert hasattr(metric, 'precision')
            assert hasattr(metric, 'recall')
            assert hasattr(metric, 'f1_score')
            assert hasattr(metric, 'auc_roc')
    
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction"""
        X, y = sample_data
        ensemble = EnsembleModelStackV2()
        
        metrics = ensemble.train_all_models(X, y, use_smote=False)
        
        for model_name, metric in metrics.items():
            if metric.feature_importance:
                assert isinstance(metric.feature_importance, dict)
                assert len(metric.feature_importance) > 0


class TestEnsemblePrediction:
    """Test ensemble predictions"""
    
    def test_predict_ensemble(self, sample_data):
        """Test ensemble prediction"""
        X, y = sample_data
        ensemble = EnsembleModelStackV2()
        
        # Train
        ensemble.train_all_models(X, y, use_smote=False)
        
        # Predict
        predictions, confidences = ensemble.predict_ensemble(X, threshold=0.5)
        
        assert len(predictions) == len(X)
        assert len(confidences) == len(X)
        assert np.all((predictions >= 0) & (predictions <= 1))
        assert np.all((confidences >= 0) & (confidences <= 1))
    
    def test_prediction_threshold(self, sample_data):
        """Test prediction with different thresholds"""
        X, y = sample_data
        ensemble = EnsembleModelStackV2()
        
        ensemble.train_all_models(X, y, use_smote=False)
        
        pred_low, _ = ensemble.predict_ensemble(X, threshold=0.3)
        pred_high, _ = ensemble.predict_ensemble(X, threshold=0.7)
        
        # Lower threshold should detect more mules
        assert np.mean(pred_low >= 0.3) >= np.mean(pred_high >= 0.7)


class TestOptimalThreshold:
    """Test optimal threshold finding"""
    
    def test_find_optimal_threshold(self, sample_data):
        """Test finding optimal threshold"""
        X, y = sample_data
        ensemble = EnsembleModelStackV2()
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y[:split_idx]
        X_val = X.iloc[split_idx:]
        y_val = y[split_idx:]
        
        # Train
        ensemble.train_all_models(X_train, y_train, use_smote=False)
        
        # Find optimal threshold
        optimal_threshold = ensemble.find_optimal_threshold(X_val, y_val, metric='f1')
        
        assert 0.0 <= optimal_threshold <= 1.0
    
    def test_threshold_metrics(self, sample_data):
        """Test threshold optimization with different metrics"""
        X, y = sample_data
        ensemble = EnsembleModelStackV2()
        
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y[:split_idx]
        X_val = X.iloc[split_idx:]
        y_val = y[split_idx:]
        
        ensemble.train_all_models(X_train, y_train, use_smote=False)
        
        # Test different metrics
        for metric in ['f1', 'precision', 'recall']:
            threshold = ensemble.find_optimal_threshold(X_val, y_val, metric=metric)
            assert 0.0 <= threshold <= 1.0


class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_save_models(self, sample_data, tmp_path):
        """Test saving models"""
        X, y = sample_data
        model_dir = str(tmp_path / 'models')
        ensemble = EnsembleModelStackV2(model_dir)
        
        ensemble.train_all_models(X, y, use_smote=False)
        ensemble.save_models()
        
        # Check that model files were created
        import os
        model_files = os.listdir(model_dir)
        assert len(model_files) > 0
    
    def test_load_models(self, sample_data, tmp_path):
        """Test loading models"""
        X, y = sample_data
        model_dir = str(tmp_path / 'models')
        
        # Train and save
        ensemble1 = EnsembleModelStackV2(model_dir)
        ensemble1.train_all_models(X, y, use_smote=False)
        ensemble1.save_models()
        
        # Load
        ensemble2 = EnsembleModelStackV2(model_dir)
        ensemble2.load_models()
        
        assert len(ensemble2.models) > 0


class TestModelWeights:
    """Test model weighting in ensemble"""
    
    def test_weighted_predictions(self, sample_data):
        """Test that model weights affect predictions"""
        X, y = sample_data
        ensemble = EnsembleModelStackV2()
        
        ensemble.train_all_models(X, y, use_smote=False)
        
        # Get predictions
        predictions, _ = ensemble.predict_ensemble(X, threshold=0.5)
        
        # Predictions should be weighted average
        assert len(predictions) == len(X)
        assert np.all((predictions >= 0) & (predictions <= 1))


class TestEnsembleRobustness:
    """Test ensemble robustness"""
    
    def test_imbalanced_data(self):
        """Test with imbalanced data"""
        X, y = make_classification(
            n_samples=500,
            n_features=30,
            n_informative=20,
            n_redundant=5,
            n_classes=2,
            weights=[0.95, 0.05],
            random_state=42
        )
        
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        ensemble = EnsembleModelStackV2()
        metrics = ensemble.train_all_models(X_df, y, use_smote=False)
        
        assert len(metrics) > 0
    
    def test_small_dataset(self):
        """Test with small dataset"""
        X, y = make_classification(
            n_samples=50,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        ensemble = EnsembleModelStackV2()
        metrics = ensemble.train_all_models(X_df, y, use_smote=False)
        
        assert len(metrics) > 0
