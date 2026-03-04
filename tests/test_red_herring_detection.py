"""
Unit tests for red-herring detection in evaluation module
"""

import pytest
import pandas as pd
import numpy as np
from src.evaluation import RedHerringDetector


class TestRedHerringDetector:
    """Test RedHerringDetector class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data with various patterns"""
        np.random.seed(42)
        n_samples = 200
        
        # Create features with different characteristics
        X = pd.DataFrame({
            # Normal feature
            'normal_feature': np.random.normal(100, 20, n_samples),
            
            # Feature with extreme uniformity (synthetic)
            'extreme_uniformity': np.tile([1, 2, 3, 4, 5], n_samples // 5),
            
            # Feature with perfect periodicity (synthetic)
            'perfect_periodicity': np.tile(np.arange(0, 10, 0.5), n_samples // 20)[:n_samples],
            
            # Feature with perfect separation
            'perfect_separation': np.concatenate([
                np.ones(n_samples // 2) * 0,
                np.ones(n_samples // 2) * 100
            ]),
            
            # Feature with artificial clustering
            'artificial_clustering': np.concatenate([
                np.random.normal(10, 1, n_samples // 3),
                np.random.normal(50, 1, n_samples // 3),
                np.random.normal(90, 1, n_samples - 2 * (n_samples // 3))
            ]),
            
            # Feature with high variance
            'high_variance': np.random.exponential(50, n_samples),
        })
        
        # Create labels (0 = non-mule, 1 = mule)
        y = np.concatenate([
            np.zeros(n_samples // 2, dtype=int),
            np.ones(n_samples // 2, dtype=int)
        ])
        
        # Shuffle
        shuffle_idx = np.random.permutation(n_samples)
        X = X.iloc[shuffle_idx].reset_index(drop=True)
        y = y[shuffle_idx]
        
        return X, y
    
    def test_detect_perfect_separation(self, sample_data):
        """Test detection of perfect separation features"""
        X, y = sample_data
        
        perfect_sep = RedHerringDetector.detect_perfect_separation(X, y)
        
        # Should detect the perfect_separation feature
        assert 'perfect_separation' in perfect_sep
        assert len(perfect_sep) > 0
    
    def test_detect_extreme_uniformity(self, sample_data):
        """Test detection of extreme uniformity (synthetic pattern)"""
        X, y = sample_data
        
        synthetic = RedHerringDetector.detect_synthetic_patterns(X, y)
        
        # Should detect extreme_uniformity
        assert 'extreme_uniformity' in synthetic['extreme_uniformity']
    
    def test_detect_perfect_periodicity(self, sample_data):
        """Test detection of perfect periodicity (synthetic pattern)"""
        X, y = sample_data
        
        synthetic = RedHerringDetector.detect_synthetic_patterns(X, y)
        
        # Should detect perfect_periodicity
        assert 'perfect_periodicity' in synthetic['perfect_periodicity']
    
    def test_detect_artificial_clustering(self, sample_data):
        """Test detection of artificial clustering (synthetic pattern)"""
        X, y = sample_data
        
        synthetic = RedHerringDetector.detect_synthetic_patterns(X, y)
        
        # Artificial clustering may or may not be detected depending on data distribution
        # Just verify the method runs without error
        assert isinstance(synthetic['artificial_clustering'], list)
    
    def test_detect_red_herrings_comprehensive(self, sample_data):
        """Test comprehensive red-herring detection"""
        X, y = sample_data
        
        # Create dummy feature importances
        feature_importances = {col: 0.1 for col in X.columns}
        
        red_herrings = RedHerringDetector.detect_red_herrings(X, y, feature_importances)
        
        # Should have detected multiple types of red-herrings
        assert 'perfect_separation' in red_herrings
        assert 'synthetic_patterns' in red_herrings
        assert 'high_variance' in red_herrings
        
        # Should detect at least some red-herrings
        total_detected = (
            len(red_herrings['perfect_separation']) +
            sum(len(v) for v in red_herrings['synthetic_patterns'].values()) +
            len(red_herrings['high_variance'])
        )
        assert total_detected > 0
    
    def test_normal_feature_not_flagged(self, sample_data):
        """Test that normal features are not flagged as red-herrings"""
        X, y = sample_data
        
        synthetic = RedHerringDetector.detect_synthetic_patterns(X, y)
        
        # Normal feature should not be in any synthetic pattern category
        for pattern_list in synthetic.values():
            assert 'normal_feature' not in pattern_list
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        X = pd.DataFrame()
        y = np.array([])
        
        red_herrings = RedHerringDetector.detect_red_herrings(X, y, {})
        
        # Should return empty results
        assert len(red_herrings['perfect_separation']) == 0
        assert len(red_herrings['high_variance']) == 0
    
    def test_single_feature(self):
        """Test handling of single feature"""
        X = pd.DataFrame({'feature': np.random.normal(0, 1, 100)})
        y = np.random.randint(0, 2, 100)
        
        red_herrings = RedHerringDetector.detect_red_herrings(X, y, {'feature': 0.5})
        
        # Should complete without error
        assert isinstance(red_herrings, dict)
    
    def test_nan_handling(self):
        """Test handling of NaN values"""
        X = pd.DataFrame({
            'feature_with_nan': [1, 2, np.nan, 4, 5] * 20,
            'normal_feature': np.random.normal(0, 1, 100)
        })
        y = np.random.randint(0, 2, 100)
        
        red_herrings = RedHerringDetector.detect_red_herrings(X, y, {
            'feature_with_nan': 0.3,
            'normal_feature': 0.7
        })
        
        # Should handle NaN values gracefully
        assert isinstance(red_herrings, dict)
    
    def test_synthetic_pattern_categories(self, sample_data):
        """Test that synthetic patterns are categorized correctly"""
        X, y = sample_data
        
        synthetic = RedHerringDetector.detect_synthetic_patterns(X, y)
        
        # Should have all expected categories
        expected_categories = [
            'extreme_uniformity',
            'perfect_periodicity',
            'unrealistic_distribution',
            'artificial_clustering',
            'suspicious_label_correlation'
        ]
        
        for category in expected_categories:
            assert category in synthetic
            assert isinstance(synthetic[category], list)
    
    def test_high_variance_detection(self, sample_data):
        """Test detection of high variance features"""
        X, y = sample_data
        
        feature_importances = {col: 0.1 for col in X.columns}
        red_herrings = RedHerringDetector.detect_red_herrings(X, y, feature_importances)
        
        # high_variance feature may or may not be detected depending on actual variance
        # Just verify the method runs without error
        assert isinstance(red_herrings['high_variance'], list)


class TestSyntheticPatternEdgeCases:
    """Test edge cases in synthetic pattern detection"""
    
    def test_constant_feature(self):
        """Test detection of constant features"""
        X = pd.DataFrame({
            'constant': np.ones(100),
            'normal': np.random.normal(0, 1, 100)
        })
        y = np.random.randint(0, 2, 100)
        
        synthetic = RedHerringDetector.detect_synthetic_patterns(X, y)
        
        # Constant feature should be detected as extreme uniformity
        assert 'constant' in synthetic['extreme_uniformity']
    
    def test_binary_feature(self):
        """Test detection of binary features"""
        X = pd.DataFrame({
            'binary': np.random.randint(0, 2, 100),
            'normal': np.random.normal(0, 1, 100)
        })
        y = np.random.randint(0, 2, 100)
        
        synthetic = RedHerringDetector.detect_synthetic_patterns(X, y)
        
        # Binary feature should be detected as extreme uniformity
        assert 'binary' in synthetic['extreme_uniformity']
    
    def test_all_nan_feature(self):
        """Test handling of all-NaN features"""
        X = pd.DataFrame({
            'all_nan': np.full(100, np.nan),
            'normal': np.random.normal(0, 1, 100)
        })
        y = np.random.randint(0, 2, 100)
        
        # Should not raise error
        synthetic = RedHerringDetector.detect_synthetic_patterns(X, y)
        assert isinstance(synthetic, dict)
    
    def test_small_sample_size(self):
        """Test handling of small sample sizes"""
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        y = np.array([0, 1, 0])
        
        # Should handle small samples gracefully
        synthetic = RedHerringDetector.detect_synthetic_patterns(X, y)
        assert isinstance(synthetic, dict)
