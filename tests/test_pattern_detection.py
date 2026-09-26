"""
Unit tests for pattern_detection module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.pattern_detection import PatternDetector, CompositeAnomalyScorer, PatternScore
from src.feature_engineering import AccountFeatures


class TestPatternScore:
    """Test PatternScore dataclass"""
    
    def test_initialization(self):
        """Test PatternScore initialization"""
        score = PatternScore(
            pattern_name='test_pattern',
            score=0.75,
            confidence=0.85,
            supporting_features=['feature1', 'feature2']
        )
        
        assert score.pattern_name == 'test_pattern'
        assert score.score == 0.75
        assert score.confidence == 0.85
        assert len(score.supporting_features) == 2


class TestPatternDetector:
    """Test PatternDetector class"""
    
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
    def sample_features(self):
        """Create sample features"""
        features = AccountFeatures(account_id='ACC001')
        features.total_transactions = 100
        features.inflow_transactions = 50
        features.outflow_transactions = 50
        features.total_inflow = 50000.0
        features.total_outflow = 50000.0
        features.avg_inflow_amount = 1000.0
        features.avg_outflow_amount = 1000.0
        features.sub_threshold_ratio = 0.2
        features.inflow_outflow_ratio = 1.0
        features.unique_sources = 10
        features.unique_destinations = 10
        features.source_concentration = 0.1
        features.account_age_days = 365
        features.activity_spike_magnitude = 2.0
        return features
    
    def test_initialization(self):
        """Test PatternDetector initialization"""
        detector = PatternDetector()
        assert detector is not None
    
    def test_detect_dormant_activation(self, sample_transactions, sample_features):
        """Test dormant activation pattern detection"""
        detector = PatternDetector()
        pattern = detector.detect_dormant_activation(sample_transactions, sample_features)
        
        assert pattern.pattern_name == 'dormant_activation'
        assert 0 <= pattern.score <= 1
        assert 0 <= pattern.confidence <= 1
    
    def test_detect_structuring(self, sample_features):
        """Test structuring pattern detection"""
        detector = PatternDetector()
        pattern = detector.detect_structuring(sample_features)
        
        assert pattern.pattern_name == 'structuring'
        assert 0 <= pattern.score <= 1
    
    def test_detect_rapid_passthrough(self, sample_features):
        """Test rapid pass-through pattern detection"""
        detector = PatternDetector()
        sample_features.avg_time_to_transfer_hours = 24
        sample_features.rapid_transfer_ratio_48h = 0.8
        
        pattern = detector.detect_rapid_passthrough(sample_features)
        
        assert pattern.pattern_name == 'rapid_passthrough'
        assert 0 <= pattern.score <= 1
    
    def test_detect_fan_in_fan_out(self, sample_features):
        """Test fan-in/fan-out pattern detection"""
        detector = PatternDetector()
        sample_features.unique_sources = 25
        sample_features.unique_destinations = 25
        
        pattern = detector.detect_fan_in_fan_out(sample_features)
        
        assert pattern.pattern_name == 'fan_in_fan_out'
        assert 0 <= pattern.score <= 1
    
    def test_detect_new_account_high_value(self, sample_features):
        """Test new account high value pattern detection"""
        detector = PatternDetector()
        sample_features.account_age_days = 30
        sample_features.total_inflow = 600000.0
        
        pattern = detector.detect_new_account_high_value(sample_features)
        
        assert pattern.pattern_name == 'new_account_high_value'
        assert pattern.score > 0
    
    def test_detect_all_patterns(self, sample_transactions, sample_features):
        """Test detecting all patterns"""
        detector = PatternDetector()
        patterns = detector.detect_all_patterns('ACC001', sample_transactions, sample_features)
        
        assert len(patterns) == 10
        assert 'dormant_activation' in patterns
        assert 'structuring' in patterns
        assert 'rapid_passthrough' in patterns


class TestCompositeAnomalyScorer:
    """Test CompositeAnomalyScorer class"""
    
    @pytest.fixture
    def sample_patterns(self):
        """Create sample pattern scores"""
        patterns = {
            'dormant_activation': PatternScore('dormant_activation', 0.5, 0.8, []),
            'structuring': PatternScore('structuring', 0.3, 0.7, []),
            'rapid_passthrough': PatternScore('rapid_passthrough', 0.7, 0.9, []),
            'fan_in_fan_out': PatternScore('fan_in_fan_out', 0.4, 0.75, []),
            'geographic_anomaly': PatternScore('geographic_anomaly', 0.2, 0.6, []),
            'new_account_high_value': PatternScore('new_account_high_value', 0.0, 0.0, []),
            'income_mismatch': PatternScore('income_mismatch', 0.0, 0.0, []),
            'post_mobile_spike': PatternScore('post_mobile_spike', 0.0, 0.0, []),
            'round_amount': PatternScore('round_amount', 0.0, 0.0, []),
            'salary_cycle': PatternScore('salary_cycle', 0.0, 0.0, [])
        }
        return patterns
    
    def test_initialization(self):
        """Test CompositeAnomalyScorer initialization"""
        scorer = CompositeAnomalyScorer()
        
        assert scorer.pattern_weights is not None
        assert len(scorer.pattern_weights) == 10
    
    def test_compute_composite_score(self, sample_patterns):
        """Test composite score computation"""
        scorer = CompositeAnomalyScorer()
        composite_score, confidence = scorer.compute_composite_score(sample_patterns)
        
        assert 0 <= composite_score <= 1
        assert 0 <= confidence <= 1
    
    def test_composite_score_bounds(self, sample_patterns):
        """Test that composite scores are within bounds"""
        scorer = CompositeAnomalyScorer()
        
        # Test with all high scores
        for pattern_name in sample_patterns:
            sample_patterns[pattern_name].score = 1.0
            sample_patterns[pattern_name].confidence = 1.0
        
        composite_score, confidence = scorer.compute_composite_score(sample_patterns)
        
        assert composite_score <= 1.0
        assert confidence <= 1.0
    
    def test_empty_patterns(self):
        """Test with empty patterns"""
        scorer = CompositeAnomalyScorer()
        composite_score, confidence = scorer.compute_composite_score({})
        
        assert composite_score == 0.0
        assert confidence == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
