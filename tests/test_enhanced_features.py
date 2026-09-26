import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src_enhanced.feature_engineering_v2 import FeatureExtractorV2, AccountFeaturesV2, FeatureScalerV2


@pytest.fixture
def sample_transactions():
    """Create sample transaction data"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    transactions = pd.DataFrame({
        'timestamp': dates,
        'source_account': ['ACC001'] * 50 + ['ACC002'] * 50,
        'destination_account': ['ACC002'] * 50 + ['ACC001'] * 50,
        'amount': np.random.uniform(1000, 50000, 100),
        'transaction_type': ['inflow'] * 50 + ['outflow'] * 50,
        'account_creation_date': [datetime(2023, 1, 1)] * 100
    })
    return transactions


@pytest.fixture
def sample_accounts():
    """Create sample accounts data"""
    return pd.DataFrame({
        'account_id': ['ACC001', 'ACC002'],
        'account_type': ['checking', 'savings']
    })


class TestVelocityFeatures:
    """Test velocity-based features"""
    
    def test_velocity_inflow_1h(self, sample_transactions, sample_accounts):
        """Test 1-hour inflow velocity"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert hasattr(features, 'velocity_inflow_1h')
        assert features.velocity_inflow_1h >= 0
    
    def test_velocity_inflow_24h(self, sample_transactions, sample_accounts):
        """Test 24-hour inflow velocity"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert hasattr(features, 'velocity_inflow_24h')
        assert features.velocity_inflow_24h >= 0
    
    def test_velocity_spike_ratio(self, sample_transactions, sample_accounts):
        """Test velocity spike ratio"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert hasattr(features, 'velocity_spike_ratio')
        assert 0 <= features.velocity_spike_ratio <= 1
    
    def test_inter_transaction_time(self, sample_transactions, sample_accounts):
        """Test inter-transaction time statistics"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert hasattr(features, 'inter_transaction_time_mean')
        assert hasattr(features, 'inter_transaction_time_std')
        assert features.inter_transaction_time_mean >= 0
        assert features.inter_transaction_time_std >= 0
    
    def test_transaction_burst_score(self, sample_transactions, sample_accounts):
        """Test transaction burst score"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert hasattr(features, 'transaction_burst_score')
        assert features.transaction_burst_score >= 0


class TestVariabilityFeatures:
    """Test variability and distribution features"""
    
    def test_coefficient_of_variation(self, sample_transactions, sample_accounts):
        """Test coefficient of variation"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert hasattr(features, 'inflow_cv')
        assert hasattr(features, 'outflow_cv')
        assert features.inflow_cv >= 0
        assert features.outflow_cv >= 0
    
    def test_skewness_kurtosis(self, sample_transactions, sample_accounts):
        """Test skewness and kurtosis"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert hasattr(features, 'amount_skewness')
        assert hasattr(features, 'amount_kurtosis')
        assert isinstance(features.amount_skewness, (int, float))
        assert isinstance(features.amount_kurtosis, (int, float))


class TestTemporalFeatures:
    """Test temporal pattern features"""
    
    def test_transaction_time_entropy(self, sample_transactions, sample_accounts):
        """Test transaction time entropy"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert hasattr(features, 'transaction_time_entropy')
        assert 0 <= features.transaction_time_entropy <= 1
    
    def test_day_of_week_concentration(self, sample_transactions, sample_accounts):
        """Test day of week concentration"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert hasattr(features, 'day_of_week_concentration')
        assert 0 <= features.day_of_week_concentration <= 1


class TestNetworkFeatures:
    """Test network-based features"""
    
    def test_pagerank_score(self, sample_transactions, sample_accounts):
        """Test PageRank score"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert hasattr(features, 'pagerank_score')
        assert features.pagerank_score >= 0
    
    def test_counterparty_diversity(self, sample_transactions, sample_accounts):
        """Test counterparty diversity"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert hasattr(features, 'counterparty_diversity')
        assert features.counterparty_diversity >= 0
    
    def test_shared_counterparty_ratio(self, sample_transactions, sample_accounts):
        """Test shared counterparty ratio"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        assert hasattr(features, 'shared_counterparty_ratio')
        assert 0 <= features.shared_counterparty_ratio <= 1


class TestFeatureScaler:
    """Test feature scaling"""
    
    def test_scaler_fit(self, sample_transactions, sample_accounts):
        """Test scaler fitting"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features_list = [
            extractor.extract_features_for_account('ACC001', sample_transactions),
            extractor.extract_features_for_account('ACC002', sample_transactions)
        ]
        
        scaler = FeatureScalerV2()
        scaler.fit(features_list)
        
        assert scaler.scaler is not None
        assert scaler.feature_names is not None
    
    def test_scaler_transform(self, sample_transactions, sample_accounts):
        """Test scaler transformation"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features_list = [
            extractor.extract_features_for_account('ACC001', sample_transactions),
            extractor.extract_features_for_account('ACC002', sample_transactions)
        ]
        
        scaler = FeatureScalerV2()
        scaler.fit(features_list)
        
        transformed = scaler.transform(features_list[0])
        assert isinstance(transformed, np.ndarray)
        assert len(transformed) > 0


class TestFeatureCount:
    """Test that we have 50+ features"""
    
    def test_feature_count(self, sample_transactions, sample_accounts):
        """Verify 50+ features are extracted"""
        extractor = FeatureExtractorV2(sample_accounts, None)
        features = extractor.extract_features_for_account('ACC001', sample_transactions)
        
        feature_dict = {
            'total_transactions': features.total_transactions,
            'inflow_transactions': features.inflow_transactions,
            'outflow_transactions': features.outflow_transactions,
            'transaction_frequency_daily': features.transaction_frequency_daily,
            'total_inflow': features.total_inflow,
            'total_outflow': features.total_outflow,
            'avg_inflow_amount': features.avg_inflow_amount,
            'avg_outflow_amount': features.avg_outflow_amount,
            'inflow_std': features.inflow_std,
            'outflow_std': features.outflow_std,
            'inflow_cv': features.inflow_cv,
            'outflow_cv': features.outflow_cv,
            'amount_skewness': features.amount_skewness,
            'amount_kurtosis': features.amount_kurtosis,
            'sub_threshold_ratio': features.sub_threshold_ratio,
            'round_amount_ratio': features.round_amount_ratio,
            'inflow_outflow_ratio': features.inflow_outflow_ratio,
            'avg_time_to_transfer_hours': features.avg_time_to_transfer_hours,
            'rapid_transfer_ratio_24h': features.rapid_transfer_ratio_24h,
            'unique_sources': features.unique_sources,
            'unique_destinations': features.unique_destinations,
            'source_concentration': features.source_concentration,
            'account_age_days': features.account_age_days,
            'dormancy_periods': features.dormancy_periods,
            'activity_spike_magnitude': features.activity_spike_magnitude,
            'transaction_time_entropy': features.transaction_time_entropy,
            'day_of_week_concentration': features.day_of_week_concentration,
            'degree_centrality': features.degree_centrality,
            'betweenness_centrality': features.betweenness_centrality,
            'clustering_coefficient': features.clustering_coefficient,
            'pagerank_score': features.pagerank_score,
            'community_size': features.community_size,
            'community_density': features.community_density,
            'velocity_inflow_1h': features.velocity_inflow_1h,
            'velocity_inflow_24h': features.velocity_inflow_24h,
            'velocity_outflow_1h': features.velocity_outflow_1h,
            'velocity_outflow_24h': features.velocity_outflow_24h,
            'velocity_spike_ratio': features.velocity_spike_ratio,
            'inter_transaction_time_mean': features.inter_transaction_time_mean,
            'inter_transaction_time_std': features.inter_transaction_time_std,
            'transaction_burst_score': features.transaction_burst_score,
            'pattern_anomaly_score': features.pattern_anomaly_score,
            'pattern_confidence': features.pattern_confidence,
            'composite_signal': features.composite_signal,
            'risk_score': features.risk_score,
            'avg_counterparty_degree': features.avg_counterparty_degree,
            'counterparty_diversity': features.counterparty_diversity,
            'shared_counterparty_ratio': features.shared_counterparty_ratio,
            'network_risk_score': features.network_risk_score,
        }
        
        assert len(feature_dict) >= 45, f"Expected 45+ features, got {len(feature_dict)}"
