"""
Burst Features for Mule Detection

Transaction burst features capture sudden spikes in activity that are characteristic
of mule accounts. These features significantly improve F1 and Temporal IoU.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BurstFeatureExtractor:
    """Extract transaction burst features from account transaction history."""
    
    def __init__(self):
        self.burst_threshold_ratio = 5.0  # Configurable threshold
    
    def extract_burst_features(self, transactions_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract all burst features from transaction data.
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            Dict with burst features
        """
        features = {
            'max_txn_per_day': 0.0,
            'max_txn_per_week': 0.0,
            'avg_txn_per_day': 0.0,
            'std_txn_per_day': 0.0,
            'burst_ratio': 0.0,
            'burst_concentration': 0.0,
            'burst_duration_days': 0.0,
            'burst_intensity': 0.0,
            'burst_frequency': 0.0,
            'inter_burst_gap': 0.0,
        }
        
        if transactions_df is None or len(transactions_df) == 0:
            return features
        
        try:
            # Find timestamp column
            date_col = self._find_date_column(transactions_df)
            if date_col is None:
                return features
            
            # Convert to datetime
            txns = transactions_df.copy()
            txns[date_col] = pd.to_datetime(txns[date_col], errors='coerce')
            txns = txns.dropna(subset=[date_col])
            
            if len(txns) == 0:
                return features
            
            # Daily transaction counts
            daily_counts = txns.groupby(txns[date_col].dt.date).size()
            
            if len(daily_counts) == 0:
                return features
            
            # Basic statistics
            features['max_txn_per_day'] = float(daily_counts.max())
            features['avg_txn_per_day'] = float(daily_counts.mean())
            features['std_txn_per_day'] = float(daily_counts.std())
            
            # Weekly statistics
            weekly_counts = txns.groupby(txns[date_col].dt.isocalendar().week).size()
            features['max_txn_per_week'] = float(weekly_counts.max()) if len(weekly_counts) > 0 else 0.0
            
            # Burst ratio (key feature)
            avg_daily = features['avg_txn_per_day']
            if avg_daily > 0:
                features['burst_ratio'] = features['max_txn_per_day'] / (avg_daily + 1e-8)
            
            # Burst concentration: % of transactions in burst days
            burst_threshold = avg_daily * self.burst_threshold_ratio
            burst_days = daily_counts[daily_counts >= burst_threshold]
            
            if len(burst_days) > 0:
                burst_txn_count = burst_days.sum()
                features['burst_concentration'] = burst_txn_count / len(txns)
                features['burst_frequency'] = float(len(burst_days))
                features['burst_duration_days'] = float((burst_days.index.max() - burst_days.index.min()).days)
            
            # Burst intensity (amount-based)
            amount_col = self._find_amount_column(txns)
            if amount_col is not None:
                daily_amounts = txns.groupby(txns[date_col].dt.date)[amount_col].sum()
                max_daily_amount = daily_amounts.max()
                avg_daily_amount = daily_amounts.mean()
                if avg_daily_amount > 0:
                    features['burst_intensity'] = max_daily_amount / (avg_daily_amount + 1e-8)
            
            # Inter-burst gap
            if len(burst_days) > 1:
                burst_dates = sorted(burst_days.index)
                gaps = [(burst_dates[i+1] - burst_dates[i]).days for i in range(len(burst_dates)-1)]
                features['inter_burst_gap'] = float(np.mean(gaps)) if gaps else 0.0
            
            # Handle NaN/Inf
            for key in features:
                if np.isnan(features[key]) or np.isinf(features[key]):
                    features[key] = 0.0
            
            return features
        
        except Exception as e:
            logger.warning(f"Error extracting burst features: {e}")
            return features
    
    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find timestamp column in dataframe."""
        for col in ['timestamp', 'transaction_timestamp', 'date', 'transaction_date', 'txn_date']:
            if col in df.columns:
                return col
        return None
    
    def _find_amount_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find amount column in dataframe."""
        for col in ['amount', 'transaction_amount', 'txn_amount', 'value']:
            if col in df.columns:
                return col
        return None


def add_burst_features_to_dataframe(features_df: pd.DataFrame, 
                                    transaction_loader) -> pd.DataFrame:
    """
    Add burst features to existing features dataframe.
    
    Args:
        features_df: DataFrame with existing features
        transaction_loader: LazyTransactionLoader instance
        
    Returns:
        DataFrame with burst features added
    """
    logger.info("Adding burst features to dataframe...")
    
    burst_extractor = BurstFeatureExtractor()
    
    # Initialize burst feature columns
    burst_cols = [
        'max_txn_per_day', 'max_txn_per_week', 'avg_txn_per_day', 'std_txn_per_day',
        'burst_ratio', 'burst_concentration', 'burst_duration_days', 'burst_intensity',
        'burst_frequency', 'inter_burst_gap'
    ]
    
    for col in burst_cols:
        features_df[col] = 0.0
    
    # Extract burst features for each account
    account_ids = features_df['account_id'].values
    
    for idx, account_id in enumerate(account_ids):
        if idx % 10000 == 0:
            logger.info(f"  Processing account {idx}/{len(account_ids)}")
        
        try:
            # Load transactions for this account
            transactions = transaction_loader.load_account_transactions(account_id)
            
            # Extract burst features
            burst_feats = burst_extractor.extract_burst_features(transactions)
            
            # Add to dataframe
            for col, val in burst_feats.items():
                features_df.loc[features_df['account_id'] == account_id, col] = val
        
        except Exception as e:
            logger.warning(f"Error processing {account_id}: {e}")
    
    logger.info(f"Added {len(burst_cols)} burst features")
    return features_df
