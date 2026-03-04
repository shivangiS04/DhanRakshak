"""
Unit tests for feature_engineering module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.feature_engineering import (
    AccountFeatures, FeatureExtractor, FeatureScaler, FeatureCache
)


class TestAccountFeatures:
    """Test AccountFeatures dataclass"""
    
    def test_initialization(self):
        """Test AccountFeatures initialization"""
        features = AccountFeatures(account_id='ACC001')
        
        assert features.account_id == 'ACC001'
        assert features.total_transactions == 0
        assert features.total_inflow == 0.0
        assert features.is_mule is None
    
    def test_feature_bounds(self):
        """Test that features are within expected bounds"""
        features = AccountFeatures(account_id='ACC001')
        features.inflow_outflow_ratio = 1.5
        features.sub_threshold_ratio = 0.3
        
        assert 0 <= features.sub_threshold_ratio <= 1
        assert features.inflow_outflow_ratio >= 0


class TestFeatureExtractor:
    """Test FeatureExtractor class"""
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'transaction_id': [f'T{i}' for i in range(100)],
            'account_id': ['ACC001'] * 100,
            'counterparty_account_id': [f'B{i % 10}' for i in range(100)],
            'amount': np.random.uniform(100, 10000, 100),
            'transaction_date': dates,
            'direction': np.random.choice(['inflow', 'outflow'], 100)
        })
    
    @pytest.fixture
    def accounts_df(self):
        """Create sample accounts data"""
        return pd.DataFrame({
            'account_id': ['ACC001', 'ACC002'],
            'customer_id': ['C1', 'C2'],
            'balance': [5000.0, 10000.0]
        })
    
    def test_initialization(self, accounts_df):
        """Test FeatureExtractor initialization"""
        extractor = FeatureExtractor(accounts_df)
        
        assert extractor.accounts_df is not None
        assert len(extractor.accounts_df) == 2
    
    def test_extract_volume_features(self, accounts_df, sample_transactions):
        """Test volume feature extraction"""
        extractor = FeatureExtractor(accounts_df)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert features.total_transactions == 100
        assert features.inflow_transactions > 0
        assert features.outflow_transactions > 0
        assert features.transaction_frequency_daily > 0
    
    def test_extract_amount_features(self, accounts_df, sample_transactions):
        """Test amount feature extraction"""
        extractor = FeatureExtractor(accounts_df)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert features.total_inflow >= 0
        assert features.total_outflow >= 0
        assert features.avg_inflow_amount >= 0
        assert features.avg_outflow_amount >= 0
    
    def test_extract_structuring_features(self, accounts_df, sample_transactions):
        """Test structuring feature extraction"""
        extractor = FeatureExtractor(accounts_df)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert 0 <= features.sub_threshold_ratio <= 1
        assert 0 <= features.round_amount_ratio <= 1
    
    def test_extract_temporal_features(self, accounts_df, sample_transactions):
        """Test temporal feature extraction"""
        extractor = FeatureExtractor(accounts_df)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert features.account_age_days >= 0
        assert features.dormancy_periods >= 0
        assert features.activity_spike_magnitude >= 0
    
    def test_empty_transactions(self, accounts_df):
        """Test handling of empty transactions"""
        empty_df = pd.DataFrame()
        extractor = FeatureExtractor(accounts_df)
        features = extractor.extract_features_for_account('ACC001', empty_df)
        
        assert features.total_transactions == 0
        assert features.total_inflow == 0.0


class TestFeatureScaler:
    """Test FeatureScaler class"""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features"""
        features_list = []
        for i in range(10):
            f = AccountFeatures(account_id=f'ACC{i:03d}')
            f.total_transactions = np.random.randint(10, 1000)
            f.total_inflow = np.random.uniform(1000, 100000)
            f.total_outflow = np.random.uniform(1000, 100000)
            features_list.append(f)
        return features_list
    
    def test_scaler_fit(self, sample_features):
        """Test scaler fitting"""
        scaler = FeatureScaler()
        scaler.fit(sample_features)
        
        assert scaler.is_fitted
        assert 'total_transactions' in scaler.feature_stats
        assert 'total_inflow' in scaler.feature_stats
    
    def test_scaler_transform(self, sample_features):
        """Test scaler transformation"""
        scaler = FeatureScaler()
        scaler.fit(sample_features)
        
        transformed = scaler.transform(sample_features[0])
        
        assert isinstance(transformed, dict)
        assert 'total_transactions' in transformed
        assert isinstance(transformed['total_transactions'], (int, float))
    
    def test_scaler_not_fitted(self, sample_features):
        """Test error when scaler not fitted"""
        scaler = FeatureScaler()
        
        with pytest.raises(ValueError):
            scaler.transform(sample_features[0])


class TestFeatureCache:
    """Test FeatureCache class"""
    
    def test_cache_put_get(self):
        """Test cache put and get operations"""
        cache = FeatureCache()
        
        features = AccountFeatures(account_id='ACC001')
        features.total_transactions = 100
        
        cache.put(features)
        retrieved = cache.get('ACC001')
        
        assert retrieved is not None
        assert retrieved.account_id == 'ACC001'
        assert retrieved.total_transactions == 100
    
    def test_cache_miss(self):
        """Test cache miss"""
        cache = FeatureCache()
        
        retrieved = cache.get('NONEXISTENT')
        
        assert retrieved is None
    
    def test_cache_clear(self):
        """Test cache clearing"""
        cache = FeatureCache()
        
        features = AccountFeatures(account_id='ACC001')
        cache.put(features)
        
        # Verify it's cached
        retrieved = cache.get('ACC001')
        assert retrieved is not None
        
        # Clear cache
        cache.clear()
        
        # Memory cache should be cleared
        assert len(cache.memory_cache) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
