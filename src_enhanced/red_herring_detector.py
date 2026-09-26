"""
Red Herring Detection Module

Identifies features that correlate with labels but don't generalize.
Red herrings are features that appear predictive in training but fail on test data.

Strategies:
1. Temporal stability: Features that change meaning over time
2. Distribution shift: Features with different distributions in train vs test
3. Spurious correlation: Features correlated with labels but not causal
4. Leakage detection: Features that leak information from labels
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import ks_2samp, chi2_contingency

logger = logging.getLogger(__name__)


@dataclass
class RedHerringScore:
    """Red herring risk assessment"""
    feature_name: str
    herring_risk: float  # 0-1, higher = more likely red herring
    temporal_instability: float
    distribution_shift: float
    spurious_correlation: float
    confidence: float


class RedHerringDetector:
    """Detect red herring features"""
    
    def __init__(self):
        """Initialize detector"""
        logger.info("RedHerringDetector initialized")
        self.feature_stats_train = {}
        self.feature_stats_test = {}
        self.is_fitted = False
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """
        Fit red herring detector on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.feature_stats_train = self._compute_feature_stats(X_train, y_train)
        self.is_fitted = True
        logger.info(f"Red herring detector fitted on {len(X_train)} samples")
    
    def detect_red_herrings(self, X_test: pd.DataFrame, y_test: Optional[np.ndarray] = None,
                           threshold: float = 0.6) -> Dict[str, RedHerringScore]:
        """
        Detect red herring features.
        
        Args:
            X_test: Test features
            y_test: Optional test labels for validation
            threshold: Risk threshold (0-1)
            
        Returns:
            Dictionary of feature_name -> RedHerringScore
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        red_herrings = {}
        
        for col in X_test.columns:
            if col not in self.feature_stats_train:
                continue
            
            # Calculate red herring risk
            temporal_instability = self._calculate_temporal_instability(col, X_test)
            distribution_shift = self._calculate_distribution_shift(col, X_test)
            spurious_correlation = self._calculate_spurious_correlation(col, X_test, y_test)
            
            # Composite risk
            herring_risk = (
                0.3 * temporal_instability +
                0.4 * distribution_shift +
                0.3 * spurious_correlation
            )
            
            if herring_risk > threshold:
                red_herrings[col] = RedHerringScore(
                    feature_name=col,
                    herring_risk=min(herring_risk, 1.0),
                    temporal_instability=temporal_instability,
                    distribution_shift=distribution_shift,
                    spurious_correlation=spurious_correlation,
                    confidence=0.7
                )
        
        logger.info(f"Detected {len(red_herrings)} red herring features")
        return red_herrings
    
    def _compute_feature_stats(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Compute statistics for each feature"""
        stats = {}
        
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue
            
            stats[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'min': X[col].min(),
                'max': X[col].max(),
                'median': X[col].median(),
                'q25': X[col].quantile(0.25),
                'q75': X[col].quantile(0.75),
                'skewness': X[col].skew(),
                'kurtosis': X[col].kurtosis(),
                'correlation_with_label': X[col].corr(pd.Series(y))
            }
        
        return stats
    
    def _calculate_temporal_instability(self, feature_name: str, X_test: pd.DataFrame) -> float:
        """
        Calculate temporal instability (feature behavior changes over time).
        
        Args:
            feature_name: Feature name
            X_test: Test features
            
        Returns:
            Instability score (0-1)
        """
        if feature_name not in X_test.columns:
            return 0.0
        
        train_stats = self.feature_stats_train.get(feature_name, {})
        test_mean = X_test[feature_name].mean()
        test_std = X_test[feature_name].std()
        
        train_mean = train_stats.get('mean', 0)
        train_std = train_stats.get('std', 1)
        
        # Measure shift in mean and std
        mean_shift = abs(test_mean - train_mean) / (train_std + 1e-8)
        std_shift = abs(test_std - train_std) / (train_std + 1e-8)
        
        instability = min((mean_shift + std_shift) / 2, 1.0)
        return instability
    
    def _calculate_distribution_shift(self, feature_name: str, X_test: pd.DataFrame) -> float:
        """
        Calculate distribution shift using KS test.
        
        Args:
            feature_name: Feature name
            X_test: Test features
            
        Returns:
            Distribution shift score (0-1)
        """
        if feature_name not in X_test.columns:
            return 0.0
        
        train_stats = self.feature_stats_train.get(feature_name, {})
        
        # Normalize test data using train statistics
        train_mean = train_stats.get('mean', 0)
        train_std = train_stats.get('std', 1)
        
        test_normalized = (X_test[feature_name] - train_mean) / (train_std + 1e-8)
        
        # KS test against standard normal
        ks_stat, p_value = ks_2samp(test_normalized, np.random.normal(0, 1, len(test_normalized)))
        
        # Higher KS stat = more distribution shift
        shift_score = min(ks_stat, 1.0)
        return shift_score
    
    def _calculate_spurious_correlation(self, feature_name: str, X_test: pd.DataFrame,
                                       y_test: Optional[np.ndarray] = None) -> float:
        """
        Calculate spurious correlation risk.
        
        Args:
            feature_name: Feature name
            X_test: Test features
            y_test: Optional test labels
            
        Returns:
            Spurious correlation score (0-1)
        """
        if feature_name not in X_test.columns:
            return 0.0
        
        train_stats = self.feature_stats_train.get(feature_name, {})
        train_corr = train_stats.get('correlation_with_label', 0)
        
        # If no test labels, estimate based on feature variance
        if y_test is None:
            # High variance features are more likely to be spurious
            feature_var = X_test[feature_name].var()
            spurious_score = min(feature_var / (feature_var + 1), 1.0)
        else:
            # Compare correlation in test vs train
            test_corr = X_test[feature_name].corr(pd.Series(y_test))
            corr_diff = abs(test_corr - train_corr)
            spurious_score = min(corr_diff, 1.0)
        
        return spurious_score
    
    def filter_features(self, X: pd.DataFrame, red_herrings: Dict[str, RedHerringScore],
                       threshold: float = 0.6) -> pd.DataFrame:
        """
        Filter out red herring features.
        
        Args:
            X: Feature dataframe
            red_herrings: Dictionary of red herring scores
            threshold: Risk threshold
            
        Returns:
            Filtered feature dataframe
        """
        herring_features = {
            name for name, score in red_herrings.items()
            if score.herring_risk > threshold
        }
        
        filtered_cols = [col for col in X.columns if col not in herring_features]
        logger.info(f"Filtered {len(herring_features)} red herring features, keeping {len(filtered_cols)}")
        
        return X[filtered_cols]


class TemporalStabilityAnalyzer:
    """Analyze temporal stability of features"""
    
    @staticmethod
    def analyze_feature_stability(X: pd.DataFrame, time_column: str,
                                 n_periods: int = 4) -> Dict[str, float]:
        """
        Analyze feature stability across time periods.
        
        Args:
            X: Feature dataframe with time column
            time_column: Name of time column
            n_periods: Number of time periods to analyze
            
        Returns:
            Dictionary of feature_name -> stability_score
        """
        if time_column not in X.columns:
            return {}
        
        X_sorted = X.sort_values(time_column)
        period_size = len(X_sorted) // n_periods
        
        stability_scores = {}
        
        for col in X_sorted.columns:
            if col == time_column or not pd.api.types.is_numeric_dtype(X_sorted[col]):
                continue
            
            # Calculate mean for each period
            period_means = []
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = (i + 1) * period_size if i < n_periods - 1 else len(X_sorted)
                
                period_data = X_sorted.iloc[start_idx:end_idx][col]
                period_means.append(period_data.mean())
            
            # Stability = 1 - coefficient of variation of period means
            period_means = np.array(period_means)
            cv = period_means.std() / (period_means.mean() + 1e-8)
            stability = max(0, 1 - cv)
            
            stability_scores[col] = stability
        
        return stability_scores


class LeakageDetector:
    """Detect potential label leakage in features"""
    
    @staticmethod
    def detect_leakage(X: pd.DataFrame, y: np.ndarray,
                      threshold: float = 0.8) -> List[str]:
        """
        Detect features with potential label leakage.
        
        Args:
            X: Features
            y: Labels
            threshold: Correlation threshold
            
        Returns:
            List of potentially leaking features
        """
        leaking_features = []
        
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue
            
            # Calculate mutual information
            mi = mutual_info_classif(X[[col]], y, random_state=42)
            
            # Normalize MI
            max_mi = np.log2(len(np.unique(y)))
            normalized_mi = mi[0] / (max_mi + 1e-8)
            
            if normalized_mi > threshold:
                leaking_features.append(col)
        
        logger.info(f"Detected {len(leaking_features)} potentially leaking features")
        return leaking_features
