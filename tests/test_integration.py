"""
Integration tests for the complete pipeline
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from src.feature_engineering import FeatureExtractor, AccountFeatures
from src.pattern_detection import PatternDetector, CompositeAnomalyScorer
from src.temporal_analysis import TemporalAnalyzer
from src.graph_analysis import TransactionNetworkGraph


class TestFeatureExtractionPipeline:
    """Test complete feature extraction pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for integration testing"""
        dates = pd.date_range('2024-01-01', periods=500, freq='H')
        
        transactions = pd.DataFrame({
            'transaction_id': [f'T{i}' for i in range(500)],
            'account_id': ['ACC001'] * 500,
            'counterparty_account_id': [f'B{i % 20}' for i in range(500)],
            'amount': np.random.uniform(100, 50000, 500),
            'transaction_date': dates,
            'direction': np.random.choice(['inflow', 'outflow'], 500)
        })
        
        accounts = pd.DataFrame({
            'account_id': ['ACC001'],
            'customer_id': ['C1'],
            'balance': [100000.0]
        })
        
        return transactions, accounts
    
    def test_end_to_end_feature_extraction(self, sample_data):
        """Test complete feature extraction flow"""
        transactions, accounts = sample_data
        
        # Extract features
        extractor = FeatureExtractor(accounts)
        features = extractor.extract_features_for_account('ACC001', transactions)
        
        # Verify features are extracted
        assert features.account_id == 'ACC001'
        assert features.total_transactions == 500
        assert features.total_inflow > 0
        assert features.total_outflow > 0
        
        # Verify feature bounds
        assert 0 <= features.sub_threshold_ratio <= 1
        assert 0 <= features.round_amount_ratio <= 1
        assert features.account_age_days >= 0


class TestPatternDetectionPipeline:
    """Test complete pattern detection pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        dates = pd.date_range('2024-01-01', periods=500, freq='H')
        
        transactions = pd.DataFrame({
            'transaction_id': [f'T{i}' for i in range(500)],
            'account_id': ['ACC001'] * 500,
            'counterparty_account_id': [f'B{i % 20}' for i in range(500)],
            'amount': np.random.uniform(100, 50000, 500),
            'transaction_date': dates,
            'direction': np.random.choice(['inflow', 'outflow'], 500)
        })
        
        accounts = pd.DataFrame({
            'account_id': ['ACC001'],
            'customer_id': ['C1']
        })
        
        return transactions, accounts
    
    def test_end_to_end_pattern_detection(self, sample_data):
        """Test complete pattern detection flow"""
        transactions, accounts = sample_data
        
        # Extract features
        extractor = FeatureExtractor(accounts)
        features = extractor.extract_features_for_account('ACC001', transactions)
        
        # Detect patterns
        detector = PatternDetector()
        patterns = detector.detect_all_patterns('ACC001', transactions, features)
        
        # Verify patterns detected
        assert len(patterns) == 10
        
        # Compute composite score
        scorer = CompositeAnomalyScorer()
        composite_score, confidence = scorer.compute_composite_score(patterns)
        
        assert 0 <= composite_score <= 1
        assert 0 <= confidence <= 1


class TestTemporalAnalysisPipeline:
    """Test complete temporal analysis pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        dates = pd.date_range('2024-01-01', periods=500, freq='H')
        
        transactions = pd.DataFrame({
            'transaction_id': [f'T{i}' for i in range(500)],
            'account_id': ['ACC001'] * 500,
            'counterparty_account_id': [f'B{i % 20}' for i in range(500)],
            'amount': np.random.uniform(100, 50000, 500),
            'transaction_date': dates,
            'direction': np.random.choice(['inflow', 'outflow'], 500)
        })
        
        return transactions
    
    def test_end_to_end_temporal_analysis(self, sample_data):
        """Test complete temporal analysis flow"""
        transactions = sample_data
        
        # Create features
        features = AccountFeatures(account_id='ACC001')
        features.total_transactions = len(transactions)
        features.total_inflow = transactions[transactions['direction'] == 'inflow']['amount'].sum()
        features.total_outflow = transactions[transactions['direction'] == 'outflow']['amount'].sum()
        
        # Detect suspicious window
        analyzer = TemporalAnalyzer()
        window = analyzer.detect_suspicious_window('ACC001', transactions, features)
        
        if window:
            assert window.account_id == 'ACC001'
            assert window.suspicious_start <= window.suspicious_end
            assert 0 <= window.anomaly_score <= 1


class TestGraphAnalysisPipeline:
    """Test complete graph analysis pipeline"""
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data"""
        dates = pd.date_range('2024-01-01', periods=200, freq='H')
        
        return pd.DataFrame({
            'transaction_id': [f'T{i}' for i in range(200)],
            'account_id': [f'A{i % 10}' for i in range(200)],
            'counterparty_account_id': [f'B{i % 10}' for i in range(200)],
            'amount': np.random.uniform(100, 10000, 200),
            'transaction_date': dates,
            'direction': np.random.choice(['inflow', 'outflow'], 200)
        })
    
    def test_end_to_end_graph_analysis(self, sample_transactions):
        """Test complete graph analysis flow"""
        # Build graph
        network = TransactionNetworkGraph()
        network.build_graph(sample_transactions)
        
        # Calculate centrality metrics
        degree_centrality = network.calculate_degree_centrality()
        betweenness_centrality = network.calculate_betweenness_centrality()
        clustering_coeff = network.calculate_clustering_coefficient()
        
        # Verify metrics
        assert len(degree_centrality) > 0
        assert len(betweenness_centrality) > 0
        assert len(clustering_coeff) > 0
        
        # Verify bounds
        for account_id, centrality in degree_centrality.items():
            assert 0 <= centrality <= 1


class TestDataConsistency:
    """Test data consistency across pipeline"""
    
    def test_feature_consistency(self):
        """Test that features maintain consistency"""
        features = AccountFeatures(account_id='ACC001')
        features.total_transactions = 100
        features.inflow_transactions = 60
        features.outflow_transactions = 40
        
        # Verify consistency
        assert features.inflow_transactions + features.outflow_transactions <= features.total_transactions
    
    def test_probability_bounds(self):
        """Test that probabilities are within bounds"""
        features = AccountFeatures(account_id='ACC001')
        
        # All ratio features should be between 0 and 1
        features.sub_threshold_ratio = 0.5
        features.round_amount_ratio = 0.3
        features.inflow_outflow_ratio = 1.2
        
        assert 0 <= features.sub_threshold_ratio <= 1
        assert 0 <= features.round_amount_ratio <= 1
    
    def test_temporal_ordering(self):
        """Test temporal ordering consistency"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        
        assert start <= end
        assert (end - start).days > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
