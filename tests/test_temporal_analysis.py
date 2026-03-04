"""
Unit tests for temporal_analysis module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.temporal_analysis import (
    TemporalAnalyzer, TemporalIoUCalculator, WindowConfidenceCalculator, SuspiciousWindow
)
from src.feature_engineering import AccountFeatures


class TestSuspiciousWindow:
    """Test SuspiciousWindow dataclass"""
    
    def test_initialization(self):
        """Test SuspiciousWindow initialization"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        
        window = SuspiciousWindow(
            account_id='ACC001',
            suspicious_start=start,
            suspicious_end=end,
            anomaly_score=0.75,
            confidence=0.85,
            pattern_type='rapid_passthrough',
            supporting_features=['feature1']
        )
        
        assert window.account_id == 'ACC001'
        assert window.suspicious_start == start
        assert window.suspicious_end == end
        assert window.anomaly_score == 0.75


class TestTemporalAnalyzer:
    """Test TemporalAnalyzer class"""
    
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
        features.total_inflow = 50000.0
        features.total_outflow = 50000.0
        return features
    
    def test_initialization(self):
        """Test TemporalAnalyzer initialization"""
        analyzer = TemporalAnalyzer()
        assert analyzer is not None
    
    def test_detect_suspicious_window(self, sample_transactions, sample_features):
        """Test suspicious window detection"""
        analyzer = TemporalAnalyzer()
        window = analyzer.detect_suspicious_window(
            'ACC001', sample_transactions, sample_features
        )
        
        if window:
            assert window.account_id == 'ACC001'
            assert window.suspicious_start <= window.suspicious_end
            assert 0 <= window.anomaly_score <= 1
    
    def test_detect_suspicious_window_empty_transactions(self, sample_features):
        """Test with empty transactions"""
        analyzer = TemporalAnalyzer()
        empty_df = pd.DataFrame()
        
        window = analyzer.detect_suspicious_window('ACC001', empty_df, sample_features)
        
        assert window is None
    
    def test_window_sizes(self, sample_transactions, sample_features):
        """Test different window sizes"""
        analyzer = TemporalAnalyzer()
        
        for window_size in [7, 30, 90]:
            window = analyzer.detect_suspicious_window(
                'ACC001', sample_transactions, sample_features,
                window_sizes=[window_size]
            )
            
            if window:
                duration = (window.suspicious_end - window.suspicious_start).days
                assert duration <= window_size + 1


class TestTemporalIoUCalculator:
    """Test TemporalIoUCalculator class"""
    
    def test_calculate_iou_perfect_overlap(self):
        """Test IoU with perfect overlap"""
        predicted = (datetime(2024, 1, 1), datetime(2024, 1, 31))
        actual = (datetime(2024, 1, 1), datetime(2024, 1, 31))
        
        iou = TemporalIoUCalculator.calculate_iou(predicted, actual)
        
        assert iou == 1.0
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU with no overlap"""
        predicted = (datetime(2024, 1, 1), datetime(2024, 1, 10))
        actual = (datetime(2024, 2, 1), datetime(2024, 2, 10))
        
        iou = TemporalIoUCalculator.calculate_iou(predicted, actual)
        
        assert iou == 0.0
    
    def test_calculate_iou_partial_overlap(self):
        """Test IoU with partial overlap"""
        predicted = (datetime(2024, 1, 1), datetime(2024, 1, 20))
        actual = (datetime(2024, 1, 10), datetime(2024, 1, 31))
        
        iou = TemporalIoUCalculator.calculate_iou(predicted, actual)
        
        assert 0 < iou < 1
    
    def test_calculate_iou_batch(self):
        """Test batch IoU calculation"""
        predicted_windows = {
            'ACC001': (datetime(2024, 1, 1), datetime(2024, 1, 31)),
            'ACC002': (datetime(2024, 2, 1), datetime(2024, 2, 28))
        }
        
        actual_windows = {
            'ACC001': (datetime(2024, 1, 1), datetime(2024, 1, 31)),
            'ACC002': (datetime(2024, 2, 1), datetime(2024, 2, 28))
        }
        
        iou_scores = TemporalIoUCalculator.calculate_iou_batch(predicted_windows, actual_windows)
        
        assert len(iou_scores) == 2
        assert iou_scores['ACC001'] == 1.0
        assert iou_scores['ACC002'] == 1.0


class TestWindowConfidenceCalculator:
    """Test WindowConfidenceCalculator class"""
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'transaction_id': [f'T{i}' for i in range(100)],
            'amount': np.random.uniform(100, 10000, 100),
            'transaction_date': dates
        })
    
    def test_calculate_confidence(self, sample_transactions):
        """Test confidence calculation"""
        window_txns = sample_transactions.iloc[:30]
        
        confidence = WindowConfidenceCalculator.calculate_confidence(
            window_txns, sample_transactions, anomaly_score=0.75
        )
        
        assert 0 <= confidence <= 1
    
    def test_confidence_with_high_anomaly(self, sample_transactions):
        """Test confidence with high anomaly score"""
        window_txns = sample_transactions.iloc[:50]
        
        confidence_high = WindowConfidenceCalculator.calculate_confidence(
            window_txns, sample_transactions, anomaly_score=0.9
        )
        
        confidence_low = WindowConfidenceCalculator.calculate_confidence(
            window_txns, sample_transactions, anomaly_score=0.1
        )
        
        assert confidence_high > confidence_low


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
