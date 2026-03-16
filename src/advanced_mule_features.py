"""
Advanced Mule Detection Features

Implements:
1. Pass-Through Money Detection
2. Counterparty Risk
3. Channel Behavior
4. Branch Risk Signals
5. Threshold Optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class AdvancedMuleFeatures:
    """Extract advanced features for mule detection."""
    
    def __init__(self):
        pass
    
    # ============ PASS-THROUGH MONEY DETECTION ============
    def extract_passthrough_features(self, transactions_df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect money pass-through patterns.
        Mules quickly forward money: IN → OUT within hours/days
        """
        features = {
            'incoming_amount': 0.0,
            'outgoing_amount': 0.0,
            'incoming_outgoing_ratio': 0.0,
            'pass_through_score': 0.0,
            'median_time_to_send': 0.0,
            'outgoing_within_24h_ratio': 0.0,
        }
        
        if transactions_df is None or len(transactions_df) == 0:
            return features
        
        try:
            txns = transactions_df.copy()
            
            # Find transaction type column
            txn_type_col = None
            for col in ['transaction_type', 'txn_type', 'type']:
                if col in txns.columns:
                    txn_type_col = col
                    break
            
            if txn_type_col is None:
                return features
            
            # Find amount column
            amount_col = None
            for col in ['amount', 'transaction_amount', 'txn_amount', 'value']:
                if col in txns.columns:
                    amount_col = col
                    break
            
            if amount_col is None:
                return features
            
            # Incoming vs Outgoing
            incoming = txns[txns[txn_type_col].isin(['C', 'credit', 'CREDIT', 'IN'])]
            outgoing = txns[txns[txn_type_col].isin(['D', 'debit', 'DEBIT', 'OUT'])]
            
            features['incoming_amount'] = float(incoming[amount_col].sum())
            features['outgoing_amount'] = float(outgoing[amount_col].sum())
            
            if features['incoming_amount'] > 0:
                features['incoming_outgoing_ratio'] = features['outgoing_amount'] / features['incoming_amount']
            
            # Pass-through score: outgoing within 24h / incoming
            if len(incoming) > 0 and len(outgoing) > 0:
                date_col = self._find_date_column(txns)
                if date_col:
                    txns[date_col] = pd.to_datetime(txns[date_col], errors='coerce')
                    
                    # For each incoming, find outgoing within 24h
                    outgoing_within_24h = 0
                    for _, inc_txn in incoming.iterrows():
                        inc_time = inc_txn[date_col]
                        matching_out = outgoing[
                            (outgoing[date_col] >= inc_time) &
                            (outgoing[date_col] <= inc_time + pd.Timedelta(hours=24))
                        ]
                        outgoing_within_24h += matching_out[amount_col].sum()
                    
                    if features['incoming_amount'] > 0:
                        features['pass_through_score'] = outgoing_within_24h / features['incoming_amount']
                        features['outgoing_within_24h_ratio'] = outgoing_within_24h / features['outgoing_amount'] if features['outgoing_amount'] > 0 else 0
                    
                    # Median time to send
                    time_diffs = []
                    for _, inc_txn in incoming.iterrows():
                        inc_time = inc_txn[date_col]
                        matching_out = outgoing[outgoing[date_col] >= inc_time]
                        if len(matching_out) > 0:
                            first_out = matching_out[date_col].min()
                            time_diff = (first_out - inc_time).total_seconds() / 3600  # hours
                            time_diffs.append(time_diff)
                    
                    if time_diffs:
                        features['median_time_to_send'] = float(np.median(time_diffs))
            
            # Clamp values
            for key in features:
                if np.isnan(features[key]) or np.isinf(features[key]):
                    features[key] = 0.0
                features[key] = min(features[key], 1e6)  # Cap extreme values
            
            return features
        
        except Exception as e:
            logger.warning(f"Error extracting pass-through features: {e}")
            return features
    
    # ============ COUNTERPARTY RISK ============
    def extract_counterparty_features(self, transactions_df: pd.DataFrame, 
                                     mule_accounts: set = None) -> Dict[str, float]:
        """
        Detect risky counterparty interactions.
        Mules interact with other mule accounts.
        """
        features = {
            'unique_counterparties': 0.0,
            'counterparty_mule_ratio': 0.0,
            'transactions_with_high_risk': 0.0,
            'counterparty_risk_score': 0.0,
        }
        
        if transactions_df is None or len(transactions_df) == 0 or mule_accounts is None:
            return features
        
        try:
            txns = transactions_df.copy()
            
            # Find counterparty column
            counterparty_col = None
            for col in ['counterparty', 'counterparty_id', 'recipient', 'sender']:
                if col in txns.columns:
                    counterparty_col = col
                    break
            
            if counterparty_col is None:
                return features
            
            unique_counterparties = txns[counterparty_col].nunique()
            features['unique_counterparties'] = float(unique_counterparties)
            
            # Count mule neighbors
            counterparties = txns[counterparty_col].unique()
            mule_neighbors = sum(1 for cp in counterparties if cp in mule_accounts)
            
            if unique_counterparties > 0:
                features['counterparty_mule_ratio'] = mule_neighbors / unique_counterparties
            
            # Transactions with high-risk accounts
            high_risk_txns = txns[txns[counterparty_col].isin(mule_accounts)]
            features['transactions_with_high_risk'] = float(len(high_risk_txns))
            
            # Risk score
            features['counterparty_risk_score'] = (
                features['counterparty_mule_ratio'] * 0.7 +
                min(features['transactions_with_high_risk'] / max(len(txns), 1), 1.0) * 0.3
            )
            
            return features
        
        except Exception as e:
            logger.warning(f"Error extracting counterparty features: {e}")
            return features
    
    # ============ CHANNEL BEHAVIOR ============
    def extract_channel_features(self, transactions_df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect suspicious channel usage patterns.
        Mules use many channels quickly.
        """
        features = {
            'unique_channels': 0.0,
            'channel_entropy': 0.0,
            'channel_switch_rate': 0.0,
            'channel_concentration': 0.0,
        }
        
        if transactions_df is None or len(transactions_df) == 0:
            return features
        
        try:
            txns = transactions_df.copy()
            
            # Find channel column
            channel_col = None
            for col in ['channel', 'transaction_channel', 'txn_channel', 'method']:
                if col in txns.columns:
                    channel_col = col
                    break
            
            if channel_col is None:
                return features
            
            channels = txns[channel_col].value_counts()
            features['unique_channels'] = float(len(channels))
            
            # Channel entropy (higher = more diverse = more suspicious)
            if len(channels) > 0:
                probs = channels.values / channels.sum()
                features['channel_entropy'] = float(entropy(probs))
            
            # Channel switch rate
            if len(txns) > 1:
                date_col = self._find_date_column(txns)
                if date_col:
                    txns = txns.sort_values(date_col)
                    channel_sequence = txns[channel_col].values
                    switches = sum(1 for i in range(len(channel_sequence)-1) 
                                 if channel_sequence[i] != channel_sequence[i+1])
                    features['channel_switch_rate'] = switches / max(len(txns) - 1, 1)
            
            # Channel concentration (inverse of entropy)
            if features['unique_channels'] > 0:
                features['channel_concentration'] = 1.0 / (1.0 + features['channel_entropy'])
            
            return features
        
        except Exception as e:
            logger.warning(f"Error extracting channel features: {e}")
            return features
    
    # ============ BRANCH RISK ============
    def extract_branch_features(self, account_df: pd.DataFrame, 
                               branch_mule_stats: Dict = None) -> Dict[str, float]:
        """
        Detect branch-level collusion patterns.
        """
        features = {
            'branch_mule_ratio': 0.0,
            'branch_txn_volume': 0.0,
            'branch_avg_txn': 0.0,
            'branch_risk_score': 0.0,
        }
        
        if account_df is None or branch_mule_stats is None:
            return features
        
        try:
            # Find branch column
            branch_col = None
            for col in ['branch', 'branch_id', 'branch_code']:
                if col in account_df.columns:
                    branch_col = col
                    break
            
            if branch_col is None:
                return features
            
            branch = account_df.get(branch_col, None)
            if branch is None or branch not in branch_mule_stats:
                return features
            
            stats = branch_mule_stats[branch]
            features['branch_mule_ratio'] = stats.get('mule_ratio', 0.0)
            features['branch_txn_volume'] = float(stats.get('txn_volume', 0))
            features['branch_avg_txn'] = float(stats.get('avg_txn', 0))
            
            # Risk score
            features['branch_risk_score'] = (
                features['branch_mule_ratio'] * 0.6 +
                min(features['branch_txn_volume'] / 10000, 1.0) * 0.4
            )
            
            return features
        
        except Exception as e:
            logger.warning(f"Error extracting branch features: {e}")
            return features
    
    # ============ HELPER METHODS ============
    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find timestamp column."""
        for col in ['timestamp', 'transaction_timestamp', 'date', 'transaction_date', 'txn_date']:
            if col in df.columns:
                return col
        return None
    
    def _find_amount_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find amount column."""
        for col in ['amount', 'transaction_amount', 'txn_amount', 'value']:
            if col in df.columns:
                return col
        return None


class ThresholdOptimizer:
    """Optimize prediction threshold for F1 score."""
    
    @staticmethod
    def optimize_threshold(y_true: np.ndarray, y_pred: np.ndarray, 
                          thresholds: np.ndarray = None) -> Tuple[float, float]:
        """
        Find optimal threshold that maximizes F1 score.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            thresholds: Thresholds to test (default: 0.1 to 0.5)
            
        Returns:
            (optimal_threshold, best_f1_score)
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.51, 0.02)
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_binary = (y_pred >= threshold).astype(int)
            
            # Calculate F1
            tp = np.sum((y_pred_binary == 1) & (y_true == 1))
            fp = np.sum((y_pred_binary == 1) & (y_true == 0))
            fn = np.sum((y_pred_binary == 0) & (y_true == 1))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.4f} → F1={best_f1:.4f}")
        return best_threshold, best_f1
