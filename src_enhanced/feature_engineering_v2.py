"""
Enhanced Feature Engineering for Mule Account Detection

Extends base features with advanced metrics:
- Velocity features (transaction speed patterns)
- Network features (graph-based metrics)
- Behavioral anomaly features
- Temporal concentration features
- Risk scoring features
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AccountFeaturesV2:
    """Enhanced account features (50+ features)"""
    account_id: str = ""
    
    # Volume features (8)
    total_transactions: int = 0
    inflow_transactions: int = 0
    outflow_transactions: int = 0
    transaction_frequency_daily: float = 0.0
    total_inflow: float = 0.0
    total_outflow: float = 0.0
    avg_inflow_amount: float = 0.0
    avg_outflow_amount: float = 0.0
    
    # Variability features (6)
    inflow_std: float = 0.0
    outflow_std: float = 0.0
    inflow_cv: float = 0.0  # Coefficient of variation
    outflow_cv: float = 0.0
    amount_skewness: float = 0.0
    amount_kurtosis: float = 0.0
    
    # Behavioral features (8)
    sub_threshold_ratio: float = 0.0
    round_amount_ratio: float = 0.0
    inflow_outflow_ratio: float = 0.0
    avg_time_to_transfer_hours: float = 0.0
    rapid_transfer_ratio_24h: float = 0.0
    unique_sources: int = 0
    unique_destinations: int = 0
    source_concentration: float = 0.0
    
    # Temporal features (8)
    account_age_days: int = 0
    dormancy_periods: int = 0
    activity_spike_magnitude: float = 0.0
    suspicious_start: Optional[datetime] = None
    suspicious_end: Optional[datetime] = None
    temporal_anomaly_score: float = 0.0
    transaction_time_entropy: float = 0.0
    day_of_week_concentration: float = 0.0
    
    # Graph features (6)
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    pagerank_score: float = 0.0
    community_size: int = 0
    community_density: float = 0.0
    
    # Velocity features (8) - NEW
    velocity_inflow_1h: float = 0.0
    velocity_inflow_24h: float = 0.0
    velocity_outflow_1h: float = 0.0
    velocity_outflow_24h: float = 0.0
    velocity_spike_ratio: float = 0.0
    inter_transaction_time_mean: float = 0.0
    inter_transaction_time_std: float = 0.0
    transaction_burst_score: float = 0.0
    
    # Risk features (6) - NEW
    pattern_anomaly_score: float = 0.0
    pattern_confidence: float = 0.0
    composite_signal: float = 0.0
    risk_score: float = 0.0
    is_mule: Optional[int] = None
    
    # Additional network features (4) - NEW
    avg_counterparty_degree: float = 0.0
    counterparty_diversity: float = 0.0
    shared_counterparty_ratio: float = 0.0
    network_risk_score: float = 0.0


class FeatureExtractorV2:
    """Enhanced feature extractor with 50+ features"""
    
    def __init__(self, accounts_df: pd.DataFrame, customers_df: Optional[pd.DataFrame] = None):
        """
        Initialize enhanced feature extractor.
        
        Args:
            accounts_df: Accounts dataframe
            customers_df: Optional customers dataframe
        """
        self.accounts = accounts_df
        self.customers = customers_df
        logger.info("FeatureExtractorV2 initialized")
    
    def extract_features_for_account(self, account_id: str, transactions: pd.DataFrame) -> AccountFeaturesV2:
        """Extract all features for an account"""
        features = AccountFeaturesV2(account_id=account_id)
        
        if transactions.empty:
            return features
        
        # Extract feature groups
        self._extract_volume_features(features, transactions)
        self._extract_variability_features(features, transactions)
        self._extract_behavioral_features(features, transactions)
        self._extract_temporal_features(features, transactions)
        self._extract_velocity_features(features, transactions)
        self._extract_network_features(features, transactions)
        self._compute_risk_scores(features, transactions)
        
        return features
    
    def _extract_volume_features(self, features: AccountFeaturesV2, transactions: pd.DataFrame) -> None:
        """Extract volume-based features"""
        features.total_transactions = len(transactions)
        
        # Handle both 'transaction_type' and 'txn_type' column names
        txn_type_col = 'txn_type' if 'txn_type' in transactions.columns else 'transaction_type'
        
        if txn_type_col in transactions.columns:
            # Map txn_type values: 'C' = credit/inflow, 'D' = debit/outflow
            if transactions[txn_type_col].dtype == 'object':
                inflow = transactions[transactions[txn_type_col].isin(['C', 'inflow'])]
                outflow = transactions[transactions[txn_type_col].isin(['D', 'outflow'])]
            else:
                inflow = transactions[transactions[txn_type_col] == 'C']
                outflow = transactions[transactions[txn_type_col] == 'D']
        else:
            inflow = pd.DataFrame()
            outflow = pd.DataFrame()
        
        features.inflow_transactions = len(inflow)
        features.outflow_transactions = len(outflow)
        
        # Determine date column
        date_col = None
        if 'transaction_timestamp' in transactions.columns:
            date_col = 'transaction_timestamp'
        elif 'timestamp' in transactions.columns:
            date_col = 'timestamp'
        
        if date_col:
            if not pd.api.types.is_datetime64_any_dtype(transactions[date_col]):
                date_range = (pd.to_datetime(transactions[date_col]).max() - pd.to_datetime(transactions[date_col]).min()).days
            else:
                date_range = (transactions[date_col].max() - transactions[date_col].min()).days
            features.transaction_frequency_daily = features.total_transactions / max(date_range, 1)
        
        features.total_inflow = inflow['amount'].sum() if not inflow.empty else 0.0
        features.total_outflow = outflow['amount'].sum() if not outflow.empty else 0.0
        
        features.avg_inflow_amount = inflow['amount'].mean() if not inflow.empty else 0.0
        features.avg_outflow_amount = outflow['amount'].mean() if not outflow.empty else 0.0
    
    def _extract_variability_features(self, features: AccountFeaturesV2, transactions: pd.DataFrame) -> None:
        """Extract variability and distribution features"""
        # Handle both 'transaction_type' and 'txn_type' column names
        txn_type_col = 'txn_type' if 'txn_type' in transactions.columns else 'transaction_type'
        
        if txn_type_col in transactions.columns:
            if transactions[txn_type_col].dtype == 'object':
                inflow = transactions[transactions[txn_type_col].isin(['C', 'inflow'])]
                outflow = transactions[transactions[txn_type_col].isin(['D', 'outflow'])]
            else:
                inflow = transactions[transactions[txn_type_col] == 'C']
                outflow = transactions[transactions[txn_type_col] == 'D']
        else:
            inflow = pd.DataFrame()
            outflow = pd.DataFrame()
        
        features.inflow_std = inflow['amount'].std() if not inflow.empty else 0.0
        features.outflow_std = outflow['amount'].std() if not outflow.empty else 0.0
        
        # Coefficient of variation
        if features.avg_inflow_amount > 0:
            features.inflow_cv = features.inflow_std / features.avg_inflow_amount
        if features.avg_outflow_amount > 0:
            features.outflow_cv = features.outflow_std / features.avg_outflow_amount
        
        # Skewness and kurtosis
        if not transactions.empty:
            features.amount_skewness = transactions['amount'].skew()
            features.amount_kurtosis = transactions['amount'].kurtosis()
    
    def _extract_behavioral_features(self, features: AccountFeaturesV2, transactions: pd.DataFrame) -> None:
        """Extract behavioral pattern features"""
        # Sub-threshold transactions
        threshold = 10000
        sub_threshold = len(transactions[transactions['amount'] < threshold])
        features.sub_threshold_ratio = sub_threshold / len(transactions) if len(transactions) > 0 else 0.0
        
        # Round amounts
        round_amounts = len(transactions[transactions['amount'] % 1000 == 0])
        features.round_amount_ratio = round_amounts / len(transactions) if len(transactions) > 0 else 0.0
        
        # Inflow/outflow ratio
        if features.total_outflow > 0:
            features.inflow_outflow_ratio = features.total_inflow / features.total_outflow
        
        # Time to transfer - handle both timestamp and transaction_timestamp
        date_col = None
        if 'transaction_timestamp' in transactions.columns:
            date_col = 'transaction_timestamp'
        elif 'timestamp' in transactions.columns:
            date_col = 'timestamp'
        
        if date_col and len(transactions) > 1:
            if not pd.api.types.is_datetime64_any_dtype(transactions[date_col]):
                dates = pd.to_datetime(transactions[date_col])
            else:
                dates = transactions[date_col]
            time_diffs = dates.diff().dt.total_seconds() / 3600
            features.avg_time_to_transfer_hours = time_diffs.mean()
        
        # Rapid transfers in 24h
        if date_col:
            if not pd.api.types.is_datetime64_any_dtype(transactions[date_col]):
                dates = pd.to_datetime(transactions[date_col])
            else:
                dates = transactions[date_col]
            last_24h = transactions[dates >= dates.max() - timedelta(hours=24)]
            features.rapid_transfer_ratio_24h = len(last_24h) / max(len(transactions), 1)
        
        # Unique counterparties
        if 'counterparty_id' in transactions.columns:
            features.unique_sources = transactions['counterparty_id'].nunique()
            features.unique_destinations = transactions['counterparty_id'].nunique()
        
        # Source concentration (Herfindahl index)
        if 'counterparty_id' in transactions.columns:
            source_counts = transactions['counterparty_id'].value_counts()
            total = len(transactions)
            features.source_concentration = (source_counts / total).pow(2).sum()
    
    def _extract_temporal_features(self, features: AccountFeaturesV2, transactions: pd.DataFrame) -> None:
        """Extract temporal pattern features"""
        # Ensure we have a date column
        date_col = None
        if 'transaction_timestamp' in transactions.columns:
            date_col = 'transaction_timestamp'
        elif 'timestamp' in transactions.columns:
            date_col = 'timestamp'
        elif 'transaction_date' in transactions.columns:
            date_col = 'transaction_date'
        else:
            return
        
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(transactions[date_col]):
            transactions[date_col] = pd.to_datetime(transactions[date_col], errors='coerce')
        
        # Account age
        if 'account_creation_date' in transactions.columns:
            creation_date = transactions['account_creation_date'].iloc[0]
            features.account_age_days = (datetime.now() - creation_date).days
        
        # Dormancy periods
        if len(transactions) > 1:
            time_diffs = transactions[date_col].diff().dt.total_seconds() / 86400
            dormancy_threshold = 30
            features.dormancy_periods = (time_diffs > dormancy_threshold).sum()
        
        # Activity spike
        if len(transactions) > 1:
            daily_counts = transactions.groupby(transactions[date_col].dt.date).size()
            features.activity_spike_magnitude = daily_counts.max() / daily_counts.mean() if daily_counts.mean() > 0 else 0.0
        
        # Transaction time entropy
        if len(transactions) > 1:
            hour_counts = transactions[date_col].dt.hour.value_counts()
            total = len(transactions)
            entropy = -(hour_counts / total * np.log2(hour_counts / total + 1e-10)).sum()
            features.transaction_time_entropy = entropy / np.log2(24)  # Normalize
        
        # Day of week concentration
        if len(transactions) > 1:
            dow_counts = transactions[date_col].dt.dayofweek.value_counts()
            total = len(transactions)
            features.day_of_week_concentration = (dow_counts / total).pow(2).sum()
        
        # Detect suspicious window from transaction patterns
        if len(transactions) > 0:
            try:
                # Find peak activity period
                daily_counts = transactions.groupby(transactions[date_col].dt.date).size()
                
                if len(daily_counts) > 0:
                    # Find day with highest activity
                    peak_date = daily_counts.idxmax()
                    peak_txns = transactions[transactions[date_col].dt.date == peak_date]
                    
                    if len(peak_txns) > 0:
                        # Get time range for peak day
                        start_time = peak_txns[date_col].min()
                        end_time = peak_txns[date_col].max()
                        
                        # Extend window to include surrounding high-activity days
                        threshold = daily_counts.mean()
                        high_activity_dates = daily_counts[daily_counts >= threshold].index
                        
                        if len(high_activity_dates) > 0:
                            window_start = pd.to_datetime(high_activity_dates.min())
                            window_end = pd.to_datetime(high_activity_dates.max())
                            
                            features.suspicious_start = window_start
                            features.suspicious_end = window_end
                            
                            # Calculate anomaly score as proportion of transactions in window
                            window_txns = transactions[
                                (transactions[date_col] >= window_start) &
                                (transactions[date_col] <= window_end)
                            ]
                            features.temporal_anomaly_score = len(window_txns) / max(len(transactions), 1)
            except Exception as e:
                logger.debug(f"Could not detect suspicious window for {features.account_id}: {e}")
    
    def _extract_velocity_features(self, features: AccountFeaturesV2, transactions: pd.DataFrame) -> None:
        """Extract velocity and speed-based features"""
        # Determine date column
        date_col = None
        if 'transaction_timestamp' in transactions.columns:
            date_col = 'transaction_timestamp'
        elif 'timestamp' in transactions.columns:
            date_col = 'timestamp'
        else:
            return
        
        if not pd.api.types.is_datetime64_any_dtype(transactions[date_col]):
            dates = pd.to_datetime(transactions[date_col])
        else:
            dates = transactions[date_col]
        
        now = dates.max()
        
        # Handle both 'transaction_type' and 'txn_type' column names
        txn_type_col = 'txn_type' if 'txn_type' in transactions.columns else 'transaction_type'
        
        if txn_type_col in transactions.columns:
            if transactions[txn_type_col].dtype == 'object':
                inflow_mask = (transactions[txn_type_col].isin(['C', 'inflow'])).values
                outflow_mask = (transactions[txn_type_col].isin(['D', 'outflow'])).values
            else:
                inflow_mask = (transactions[txn_type_col] == 'C').values
                outflow_mask = (transactions[txn_type_col] == 'D').values
        else:
            inflow_mask = [False] * len(transactions)
            outflow_mask = [False] * len(transactions)
        
        # Velocity in last 1 hour
        last_1h = transactions[dates >= now - timedelta(hours=1)]
        features.velocity_inflow_1h = last_1h.loc[inflow_mask[:len(last_1h)], 'amount'].sum() if len(last_1h) > 0 else 0.0
        features.velocity_outflow_1h = last_1h.loc[outflow_mask[:len(last_1h)], 'amount'].sum() if len(last_1h) > 0 else 0.0
        
        # Velocity in last 24 hours
        last_24h = transactions[dates >= now - timedelta(hours=24)]
        features.velocity_inflow_24h = last_24h.loc[inflow_mask[:len(last_24h)], 'amount'].sum() if len(last_24h) > 0 else 0.0
        features.velocity_outflow_24h = last_24h.loc[outflow_mask[:len(last_24h)], 'amount'].sum() if len(last_24h) > 0 else 0.0
        
        # Velocity spike ratio
        if features.velocity_inflow_24h > 0 and features.total_inflow > 0:
            features.velocity_spike_ratio = features.velocity_inflow_24h / features.total_inflow
        
        # Inter-transaction time statistics
        if len(transactions) > 1:
            time_diffs = dates.diff().dt.total_seconds() / 60  # in minutes
            features.inter_transaction_time_mean = time_diffs.mean()
            features.inter_transaction_time_std = time_diffs.std()
        
        # Transaction burst score
        if len(transactions) > 1:
            hourly_counts = transactions.groupby(dates.dt.floor('h')).size()
            if len(hourly_counts) > 0:
                features.transaction_burst_score = hourly_counts.max() / hourly_counts.mean() if hourly_counts.mean() > 0 else 0.0
    
    def _extract_network_features(self, features: AccountFeaturesV2, transactions: pd.DataFrame) -> None:
        """Extract network-based features"""
        # These will be populated by graph analysis
        # Placeholder for integration with graph module
        pass
    
    def _compute_risk_scores(self, features: AccountFeaturesV2, transactions: pd.DataFrame) -> None:
        """Compute composite risk scores"""
        risk_factors = []
        
        # High velocity
        if features.velocity_spike_ratio > 0.5:
            risk_factors.append(0.3)
        
        # Rapid transfers
        if features.rapid_transfer_ratio_24h > 0.3:
            risk_factors.append(0.25)
        
        # High concentration
        if features.source_concentration > 0.5:
            risk_factors.append(0.2)
        
        # Unusual amounts
        if features.round_amount_ratio > 0.5:
            risk_factors.append(0.15)
        
        # Low account age
        if features.account_age_days < 30:
            risk_factors.append(0.25)
        
        if risk_factors:
            features.risk_score = np.mean(risk_factors)
        else:
            features.risk_score = 0.0


class FeatureScalerV2:
    """Scale features for model input"""
    
    def __init__(self):
        self.scaler = None
        self.feature_names = None
    
    def fit(self, features_list: List[AccountFeaturesV2]) -> None:
        """Fit scaler on features"""
        from sklearn.preprocessing import StandardScaler
        
        feature_dicts = [self._features_to_dict(f) for f in features_list]
        df = pd.DataFrame(feature_dicts)
        
        self.feature_names = df.columns.tolist()
        self.scaler = StandardScaler()
        self.scaler.fit(df)
    
    def transform(self, features: AccountFeaturesV2) -> np.ndarray:
        """Transform single feature"""
        feature_dict = self._features_to_dict(features)
        df = pd.DataFrame([feature_dict])
        return self.scaler.transform(df)[0]
    
    def _features_to_dict(self, features: AccountFeaturesV2) -> Dict:
        """Convert features to dictionary"""
        return {
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
