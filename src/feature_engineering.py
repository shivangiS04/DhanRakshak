"""
Feature Engineering Pipeline for Mule Account Detection

Extracts 9 feature categories from transaction data:
1. Transaction Volume & Velocity
2. Amount-Based Features
3. Structuring Indicators
4. Pass-Through Indicators
5. Fan-In/Fan-Out Patterns
6. Geographic & Demographic Anomalies
7. Temporal Patterns
8. Account Lifecycle Features
9. Network/Graph Features
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AccountFeatures:
    """Complete feature vector for an account"""
    account_id: str
    
    # Transaction Volume Features
    total_transactions: int = 0
    inflow_transactions: int = 0
    outflow_transactions: int = 0
    transaction_frequency_daily: float = 0.0
    avg_transaction_interval_days: float = 0.0
    
    # Amount-Based Features
    total_inflow: float = 0.0
    total_outflow: float = 0.0
    avg_inflow_amount: float = 0.0
    avg_outflow_amount: float = 0.0
    inflow_std: float = 0.0
    outflow_std: float = 0.0
    inflow_p25: float = 0.0
    inflow_p50: float = 0.0
    inflow_p75: float = 0.0
    inflow_p95: float = 0.0
    outflow_p25: float = 0.0
    outflow_p50: float = 0.0
    outflow_p75: float = 0.0
    outflow_p95: float = 0.0
    
    # Structuring Indicators
    sub_threshold_count: int = 0
    sub_threshold_ratio: float = 0.0
    round_amount_count: int = 0
    round_amount_ratio: float = 0.0
    
    # Pass-Through Indicators
    inflow_outflow_ratio: float = 0.0
    avg_time_to_transfer_hours: float = 0.0
    rapid_transfer_ratio_24h: float = 0.0
    rapid_transfer_ratio_48h: float = 0.0
    account_balance_volatility: float = 0.0
    
    # Fan-In/Fan-Out Patterns
    unique_sources: int = 0
    unique_destinations: int = 0
    source_concentration: float = 0.0
    destination_concentration: float = 0.0
    source_destination_ratio: float = 0.0
    
    # Geographic & Demographic Anomalies
    cross_border_ratio: float = 0.0
    location_anomaly_score: float = 0.0
    
    # Temporal Patterns
    account_age_days: int = 0
    dormancy_periods: int = 0
    activity_spike_magnitude: float = 0.0
    
    # Account Lifecycle
    new_account_high_value: int = 0
    
    # Network Features (computed later)
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    
    # Label (if available)
    is_mule: Optional[int] = None


class FeatureExtractor:
    """Extract features from transaction data"""
    
    def __init__(self, accounts_df: pd.DataFrame, customers_df: Optional[pd.DataFrame] = None):
        """
        Initialize feature extractor.
        
        Args:
            accounts_df: Accounts dataframe
            customers_df: Optional customers dataframe for demographic features
        """
        self.accounts_df = accounts_df
        self.customers_df = customers_df
        self.feature_cache = {}
        logger.info("FeatureExtractor initialized")
    
    def extract_features_for_account(self, account_id: str, transactions: pd.DataFrame) -> AccountFeatures:
        """
        Extract all features for a single account.
        
        Args:
            account_id: Account ID
            transactions: DataFrame with transactions for this account
            
        Returns:
            AccountFeatures object
        """
        features = AccountFeatures(account_id=account_id)
        
        if transactions.empty:
            return features
        
        # Ensure transaction_date is datetime
        if 'transaction_date' in transactions.columns:
            transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        
        # Extract each feature category
        self._extract_volume_features(features, transactions)
        self._extract_amount_features(features, transactions)
        self._extract_structuring_features(features, transactions)
        self._extract_passthrough_features(features, transactions)
        self._extract_fanin_fanout_features(features, transactions)
        self._extract_temporal_features(features, transactions)
        self._extract_lifecycle_features(features, transactions)
        
        return features
    
    def _extract_volume_features(self, features: AccountFeatures, transactions: pd.DataFrame) -> None:
        """Extract transaction volume features"""
        features.total_transactions = len(transactions)
        
        # Inflow/outflow split
        if 'direction' in transactions.columns:
            features.inflow_transactions = (transactions['direction'] == 'inflow').sum()
            features.outflow_transactions = (transactions['direction'] == 'outflow').sum()
        elif 'amount' in transactions.columns:
            features.inflow_transactions = (transactions['amount'] > 0).sum()
            features.outflow_transactions = (transactions['amount'] < 0).sum()
        
        # Transaction frequency
        if 'transaction_date' in transactions.columns:
            date_range = (transactions['transaction_date'].max() - transactions['transaction_date'].min()).days
            if date_range > 0:
                features.transaction_frequency_daily = features.total_transactions / date_range
                features.avg_transaction_interval_days = date_range / features.total_transactions if features.total_transactions > 0 else 0
    
    def _extract_amount_features(self, features: AccountFeatures, transactions: pd.DataFrame) -> None:
        """Extract amount-based features"""
        if 'amount' not in transactions.columns:
            return
        
        # Separate inflow and outflow
        if 'direction' in transactions.columns:
            inflow = transactions[transactions['direction'] == 'inflow']['amount']
            outflow = transactions[transactions['direction'] == 'outflow']['amount']
        else:
            inflow = transactions[transactions['amount'] > 0]['amount']
            outflow = transactions[transactions['amount'] < 0]['amount'].abs()
        
        # Totals and averages
        features.total_inflow = inflow.sum() if len(inflow) > 0 else 0.0
        features.total_outflow = outflow.sum() if len(outflow) > 0 else 0.0
        features.avg_inflow_amount = inflow.mean() if len(inflow) > 0 else 0.0
        features.avg_outflow_amount = outflow.mean() if len(outflow) > 0 else 0.0
        features.inflow_std = inflow.std() if len(inflow) > 1 else 0.0
        features.outflow_std = outflow.std() if len(outflow) > 1 else 0.0
        
        # Percentiles
        if len(inflow) > 0:
            features.inflow_p25 = inflow.quantile(0.25)
            features.inflow_p50 = inflow.quantile(0.50)
            features.inflow_p75 = inflow.quantile(0.75)
            features.inflow_p95 = inflow.quantile(0.95)
        
        if len(outflow) > 0:
            features.outflow_p25 = outflow.quantile(0.25)
            features.outflow_p50 = outflow.quantile(0.50)
            features.outflow_p75 = outflow.quantile(0.75)
            features.outflow_p95 = outflow.quantile(0.95)
    
    def _extract_structuring_features(self, features: AccountFeatures, transactions: pd.DataFrame) -> None:
        """Extract structuring indicators"""
        if 'amount' not in transactions.columns:
            return
        
        amounts = transactions['amount'].abs()
        
        # Sub-threshold transactions (9k-10k range)
        sub_threshold = ((amounts >= 9000) & (amounts <= 10000)).sum()
        features.sub_threshold_count = sub_threshold
        features.sub_threshold_ratio = sub_threshold / len(transactions) if len(transactions) > 0 else 0.0
        
        # Round amounts (multiples of 1000, 5000, 10000)
        round_amounts = ((amounts % 1000 == 0) | (amounts % 5000 == 0) | (amounts % 10000 == 0)).sum()
        features.round_amount_count = round_amounts
        features.round_amount_ratio = round_amounts / len(transactions) if len(transactions) > 0 else 0.0
    
    def _extract_passthrough_features(self, features: AccountFeatures, transactions: pd.DataFrame) -> None:
        """Extract pass-through indicators"""
        if features.total_outflow == 0:
            features.inflow_outflow_ratio = 0.0
        else:
            features.inflow_outflow_ratio = features.total_inflow / features.total_outflow
        
        # Time to transfer (inflow to outflow)
        if 'transaction_date' in transactions.columns and 'direction' in transactions.columns:
            inflows = transactions[transactions['direction'] == 'inflow'].sort_values('transaction_date')
            outflows = transactions[transactions['direction'] == 'outflow'].sort_values('transaction_date')
            
            if len(inflows) > 0 and len(outflows) > 0:
                time_diffs = []
                for inflow_date in inflows['transaction_date']:
                    # Find next outflow after this inflow
                    next_outflows = outflows[outflows['transaction_date'] > inflow_date]
                    if len(next_outflows) > 0:
                        time_diff = (next_outflows.iloc[0]['transaction_date'] - inflow_date).total_seconds() / 3600
                        time_diffs.append(time_diff)
                
                if time_diffs:
                    features.avg_time_to_transfer_hours = np.mean(time_diffs)
                    
                    # Rapid transfer ratios
                    features.rapid_transfer_ratio_24h = sum(1 for t in time_diffs if t <= 24) / len(time_diffs)
                    features.rapid_transfer_ratio_48h = sum(1 for t in time_diffs if t <= 48) / len(time_diffs)
    
    def _extract_fanin_fanout_features(self, features: AccountFeatures, transactions: pd.DataFrame) -> None:
        """Extract fan-in/fan-out patterns"""
        if 'counterparty_account_id' not in transactions.columns:
            return
        
        if 'direction' in transactions.columns:
            inflows = transactions[transactions['direction'] == 'inflow']
            outflows = transactions[transactions['direction'] == 'outflow']
            
            features.unique_sources = inflows['counterparty_account_id'].nunique()
            features.unique_destinations = outflows['counterparty_account_id'].nunique()
        else:
            features.unique_sources = transactions['counterparty_account_id'].nunique()
            features.unique_destinations = transactions['counterparty_account_id'].nunique()
        
        # Concentration (Herfindahl index)
        if 'counterparty_account_id' in transactions.columns:
            counterparty_counts = transactions['counterparty_account_id'].value_counts()
            total = len(transactions)
            
            if total > 0:
                herfindahl = (counterparty_counts / total) ** 2
                concentration = herfindahl.sum()
                features.source_concentration = concentration
                features.destination_concentration = concentration
        
        # Ratio
        if features.unique_destinations > 0:
            features.source_destination_ratio = features.unique_sources / features.unique_destinations
    
    def _extract_temporal_features(self, features: AccountFeatures, transactions: pd.DataFrame) -> None:
        """Extract temporal patterns"""
        if 'transaction_date' not in transactions.columns:
            return
        
        dates = pd.to_datetime(transactions['transaction_date'])
        first_date = dates.min()
        last_date = dates.max()
        
        # Account age
        features.account_age_days = (last_date - first_date).days
        
        # Dormancy periods (gaps > 30 days)
        sorted_dates = sorted(dates)
        dormancy_count = 0
        for i in range(1, len(sorted_dates)):
            gap = (sorted_dates[i] - sorted_dates[i-1]).days
            if gap > 30:
                dormancy_count += 1
        features.dormancy_periods = dormancy_count
        
        # Activity spike (max transactions in 30-day window)
        if len(dates) > 1:
            max_activity = 0
            for i in range(len(sorted_dates)):
                window_end = sorted_dates[i] + timedelta(days=30)
                activity = sum(1 for d in sorted_dates if sorted_dates[i] <= d <= window_end)
                max_activity = max(max_activity, activity)
            
            avg_activity = len(dates) / (features.account_age_days + 1)
            features.activity_spike_magnitude = max_activity / avg_activity if avg_activity > 0 else 0.0
    
    def _extract_lifecycle_features(self, features: AccountFeatures, transactions: pd.DataFrame) -> None:
        """Extract account lifecycle features"""
        # New account high value: age < 90 days and high volume
        if features.account_age_days < 90 and features.total_inflow > 500000:
            features.new_account_high_value = 1


class FeatureScaler:
    """Normalize and scale features for model input"""
    
    def __init__(self):
        """Initialize scaler"""
        self.feature_stats = {}
        self.is_fitted = False
    
    def fit(self, features_list: List[AccountFeatures]) -> None:
        """
        Fit scaler on feature list.
        
        Args:
            features_list: List of AccountFeatures objects
        """
        # Convert to dataframe for easier computation
        features_dict = [asdict(f) for f in features_list]
        df = pd.DataFrame(features_dict)
        
        # Compute statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        self.is_fitted = True
        logger.info(f"Scaler fitted on {len(features_list)} features")
    
    def transform(self, features: AccountFeatures) -> Dict[str, float]:
        """
        Transform features to normalized values.
        
        Args:
            features: AccountFeatures object
            
        Returns:
            Dictionary of normalized features
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        features_dict = asdict(features)
        normalized = {}
        
        for key, value in features_dict.items():
            if key in self.feature_stats and isinstance(value, (int, float)):
                stats = self.feature_stats[key]
                std = stats['std']
                
                # Avoid division by zero
                if std > 0:
                    normalized[key] = (value - stats['mean']) / std
                else:
                    normalized[key] = 0.0
            else:
                normalized[key] = value
        
        return normalized


class FeatureCache:
    """Cache computed features to avoid recomputation"""
    
    def __init__(self, cache_dir: str = 'data/feature_cache'):
        """
        Initialize feature cache.
        
        Args:
            cache_dir: Directory to store cached features
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
    
    def get(self, account_id: str) -> Optional[AccountFeatures]:
        """Get cached features for account"""
        # Check memory cache first
        if account_id in self.memory_cache:
            return self.memory_cache[account_id]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{account_id}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    features = pickle.load(f)
                    self.memory_cache[account_id] = features
                    return features
            except Exception as e:
                logger.warning(f"Error loading cached features for {account_id}: {e}")
        
        return None
    
    def put(self, features: AccountFeatures) -> None:
        """Cache features for account"""
        # Store in memory
        self.memory_cache[features.account_id] = features
        
        # Store on disk
        cache_file = self.cache_dir / f"{features.account_id}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
        except Exception as e:
            logger.warning(f"Error caching features for {features.account_id}: {e}")
    
    def clear(self) -> None:
        """Clear all caches"""
        self.memory_cache.clear()
        logger.info("Feature cache cleared")
