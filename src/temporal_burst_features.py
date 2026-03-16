"""
Advanced Temporal Burst Features for Mule Detection

Implements key signals for improving window recall:
1. Freeze/Unfreeze Detection - Long inactivity gaps followed by bursts
2. Pass-Through Money Detection - Money in → money out quickly
3. Counterparty Explosion - Sudden spike in unique counterparties
4. Channel Switching - Rapid switching between transaction channels
5. Multi-Signal Burst Detection - Union of multiple burst indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TemporalBurstFeatureExtractor:
    """Extract advanced temporal burst features for mule detection."""
    
    def __init__(self):
        self.inactivity_threshold_days = 15  # Gap > 15 days = freeze
        self.burst_threshold_multiplier = 2.0  # Burst = mean + 2*std
        self.pass_through_time_hours = 24  # Money out within 24h
        self.counterparty_spike_threshold = 3.0  # 3x normal = spike
        self.channel_entropy_threshold = 1.5  # High entropy = risky
    
    def extract_all_features(self, transactions_df: pd.DataFrame) -> Dict[str, float]:
        """Extract all temporal burst features."""
        features = {}
        
        if transactions_df is None or len(transactions_df) == 0:
            return self._get_empty_features()
        
        try:
            date_col = self._find_date_column(transactions_df)
            if date_col is None:
                return self._get_empty_features()
            
            txns = transactions_df.copy()
            txns[date_col] = pd.to_datetime(txns[date_col], errors='coerce')
            txns = txns.dropna(subset=[date_col])
            
            if len(txns) == 0:
                return self._get_empty_features()
            
            # Extract each feature group
            features.update(self._extract_freeze_unfreeze_features(txns, date_col))
            features.update(self._extract_pass_through_features(txns, date_col))
            features.update(self._extract_counterparty_features(txns, date_col))
            features.update(self._extract_channel_features(txns, date_col))
            features.update(self._extract_multi_signal_bursts(txns, date_col))
            
            # Handle NaN/Inf
            for key in features:
                if np.isnan(features[key]) or np.isinf(features[key]):
                    features[key] = 0.0
            
            return features
        
        except Exception as e:
            logger.warning(f"Error extracting temporal burst features: {e}")
            return self._get_empty_features()
    
    def _extract_freeze_unfreeze_features(self, txns: pd.DataFrame, date_col: str) -> Dict[str, float]:
        """Detect freeze/unfreeze patterns - long gaps followed by bursts."""
        features = {
            'max_inactivity_gap_days': 0.0,
            'avg_inactivity_gap_days': 0.0,
            'burst_after_gap_ratio': 0.0,
            'txn_after_gap_3day': 0.0,
            'amount_after_gap_3day': 0.0,
            'freeze_unfreeze_score': 0.0,
        }
        
        try:
            # Sort by date
            txns_sorted = txns.sort_values(date_col)
            dates = pd.to_datetime(txns_sorted[date_col]).unique()
            dates = np.sort(dates)
            
            if len(dates) < 2:
                return features
            
            # Calculate gaps between consecutive transaction dates
            gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            
            if not gaps:
                return features
            
            # Gap statistics
            features['max_inactivity_gap_days'] = float(max(gaps))
            features['avg_inactivity_gap_days'] = float(np.mean(gaps))
            
            # Find large gaps (freeze periods)
            large_gaps = [i for i, gap in enumerate(gaps) if gap > self.inactivity_threshold_days]
            
            if large_gaps:
                # For each large gap, check burst after
                burst_ratios = []
                
                for gap_idx in large_gaps:
                    gap_end_date = dates[gap_idx + 1]
                    gap_start_date = dates[gap_idx]
                    
                    # Count transactions in 3 days after gap
                    after_gap = txns_sorted[txns_sorted[date_col] >= gap_end_date]
                    after_gap_3day = after_gap[
                        after_gap[date_col] <= gap_end_date + timedelta(days=3)
                    ]
                    
                    # Count transactions in 3 days before gap
                    before_gap = txns_sorted[txns_sorted[date_col] <= gap_start_date]
                    before_gap_3day = before_gap[
                        before_gap[date_col] >= gap_start_date - timedelta(days=3)
                    ]
                    
                    if len(before_gap_3day) > 0:
                        ratio = len(after_gap_3day) / (len(before_gap_3day) + 1e-8)
                        burst_ratios.append(ratio)
                
                if burst_ratios:
                    features['burst_after_gap_ratio'] = float(np.mean(burst_ratios))
                    features['txn_after_gap_3day'] = float(len(after_gap_3day))
                    
                    # Amount after gap
                    amount_col = self._find_amount_column(txns)
                    if amount_col:
                        features['amount_after_gap_3day'] = float(after_gap_3day[amount_col].sum())
                    
                    # Freeze-unfreeze score: high gap + high burst = suspicious
                    features['freeze_unfreeze_score'] = (
                        features['max_inactivity_gap_days'] / 30.0 * 
                        features['burst_after_gap_ratio']
                    )
        
        except Exception as e:
            logger.warning(f"Error in freeze/unfreeze detection: {e}")
        
        return features
    
    def _extract_pass_through_features(self, txns: pd.DataFrame, date_col: str) -> Dict[str, float]:
        """Detect pass-through money patterns - money in then out quickly."""
        features = {
            'pass_through_ratio_24h': 0.0,
            'pass_through_ratio_6h': 0.0,
            'avg_time_to_send_hours': 0.0,
            'pass_through_score': 0.0,
            'rapid_relay_count': 0.0,
        }
        
        try:
            amount_col = self._find_amount_column(txns)
            if amount_col is None:
                return features
            
            # Separate incoming and outgoing
            txns_sorted = txns.sort_values(date_col)
            
            # Assume 'transaction_type' or similar exists, or infer from amount sign
            # For now, we'll look for rapid money movements
            
            total_incoming = 0.0
            total_outgoing_24h = 0.0
            total_outgoing_6h = 0.0
            time_to_send_list = []
            rapid_relays = 0
            
            # Simple heuristic: if we have direction info
            if 'transaction_type' in txns.columns or 'direction' in txns.columns:
                direction_col = 'transaction_type' if 'transaction_type' in txns.columns else 'direction'
                
                for idx, row in txns_sorted.iterrows():
                    if row[direction_col] in ['incoming', 'credit', 'in']:
                        incoming_amount = row[amount_col]
                        incoming_time = pd.to_datetime(row[date_col])
                        
                        # Find outgoing within 24h
                        outgoing_24h = txns_sorted[
                            (txns_sorted[date_col] > incoming_time) &
                            (txns_sorted[date_col] <= incoming_time + timedelta(hours=24)) &
                            (txns_sorted[direction_col].isin(['outgoing', 'debit', 'out']))
                        ]
                        
                        outgoing_6h = txns_sorted[
                            (txns_sorted[date_col] > incoming_time) &
                            (txns_sorted[date_col] <= incoming_time + timedelta(hours=6)) &
                            (txns_sorted[direction_col].isin(['outgoing', 'debit', 'out']))
                        ]
                        
                        total_incoming += incoming_amount
                        total_outgoing_24h += outgoing_24h[amount_col].sum()
                        total_outgoing_6h += outgoing_6h[amount_col].sum()
                        
                        if len(outgoing_24h) > 0:
                            min_time = (outgoing_24h[date_col].min() - incoming_time).total_seconds() / 3600
                            time_to_send_list.append(min_time)
                            
                            if min_time < 6:
                                rapid_relays += 1
            
            # Calculate ratios
            if total_incoming > 0:
                features['pass_through_ratio_24h'] = total_outgoing_24h / total_incoming
                features['pass_through_ratio_6h'] = total_outgoing_6h / total_incoming
                features['pass_through_score'] = features['pass_through_ratio_24h']
            
            if time_to_send_list:
                features['avg_time_to_send_hours'] = float(np.mean(time_to_send_list))
            
            features['rapid_relay_count'] = float(rapid_relays)
        
        except Exception as e:
            logger.warning(f"Error in pass-through detection: {e}")
        
        return features
    
    def _extract_counterparty_features(self, txns: pd.DataFrame, date_col: str) -> Dict[str, float]:
        """Detect counterparty explosion - sudden spike in unique counterparties."""
        features = {
            'unique_counterparties_total': 0.0,
            'unique_counterparties_7day': 0.0,
            'unique_counterparties_1day': 0.0,
            'counterparty_growth_rate': 0.0,
            'daily_counterparty_spike': 0.0,
            'counterparty_explosion_score': 0.0,
        }
        
        try:
            # Find counterparty column
            counterparty_col = None
            for col in ['counterparty', 'recipient', 'sender', 'beneficiary', 'counterparty_id']:
                if col in txns.columns:
                    counterparty_col = col
                    break
            
            if counterparty_col is None:
                return features
            
            txns_sorted = txns.sort_values(date_col)
            
            # Total unique counterparties
            features['unique_counterparties_total'] = float(txns_sorted[counterparty_col].nunique())
            
            # Last 7 days
            last_date = pd.to_datetime(txns_sorted[date_col]).max()
            week_ago = last_date - timedelta(days=7)
            txns_7day = txns_sorted[pd.to_datetime(txns_sorted[date_col]) >= week_ago]
            features['unique_counterparties_7day'] = float(txns_7day[counterparty_col].nunique())
            
            # Last 1 day
            day_ago = last_date - timedelta(days=1)
            txns_1day = txns_sorted[pd.to_datetime(txns_sorted[date_col]) >= day_ago]
            features['unique_counterparties_1day'] = float(txns_1day[counterparty_col].nunique())
            
            # Growth rate
            if len(txns_7day) > 0:
                features['counterparty_growth_rate'] = (
                    features['unique_counterparties_7day'] / 
                    (features['unique_counterparties_total'] + 1e-8)
                )
            
            # Daily spike
            daily_counterparties = txns_sorted.groupby(
                pd.to_datetime(txns_sorted[date_col]).dt.date
            )[counterparty_col].nunique()
            
            if len(daily_counterparties) > 0:
                avg_daily = daily_counterparties.mean()
                max_daily = daily_counterparties.max()
                if avg_daily > 0:
                    features['daily_counterparty_spike'] = max_daily / avg_daily
                    
                    # Explosion score
                    if features['daily_counterparty_spike'] > self.counterparty_spike_threshold:
                        features['counterparty_explosion_score'] = (
                            features['daily_counterparty_spike'] * 
                            features['unique_counterparties_7day'] / 10.0
                        )
        
        except Exception as e:
            logger.warning(f"Error in counterparty detection: {e}")
        
        return features
    
    def _extract_channel_features(self, txns: pd.DataFrame, date_col: str) -> Dict[str, float]:
        """Detect channel switching - rapid switching between transaction channels."""
        features = {
            'unique_channels': 0.0,
            'channel_entropy': 0.0,
            'channel_switch_rate': 0.0,
            'channels_7day': 0.0,
            'channel_switching_score': 0.0,
        }
        
        try:
            # Find channel column
            channel_col = None
            for col in ['channel', 'transaction_channel', 'txn_channel', 'mode']:
                if col in txns.columns:
                    channel_col = col
                    break
            
            if channel_col is None:
                return features
            
            txns_sorted = txns.sort_values(date_col)
            
            # Unique channels
            features['unique_channels'] = float(txns_sorted[channel_col].nunique())
            
            # Channel entropy
            channel_counts = txns_sorted[channel_col].value_counts()
            channel_probs = channel_counts / len(txns_sorted)
            entropy = -np.sum(channel_probs * np.log(channel_probs + 1e-10))
            features['channel_entropy'] = float(entropy)
            
            # Channel switch rate
            channels_seq = txns_sorted[channel_col].values
            if len(channels_seq) > 1:
                switches = np.sum(channels_seq[:-1] != channels_seq[1:])
                features['channel_switch_rate'] = switches / (len(channels_seq) - 1)
            
            # Channels in last 7 days
            last_date = pd.to_datetime(txns_sorted[date_col]).max()
            week_ago = last_date - timedelta(days=7)
            txns_7day = txns_sorted[pd.to_datetime(txns_sorted[date_col]) >= week_ago]
            features['channels_7day'] = float(txns_7day[channel_col].nunique())
            
            # Channel switching score
            if features['channel_entropy'] > self.channel_entropy_threshold:
                features['channel_switching_score'] = (
                    features['channel_entropy'] * 
                    features['channel_switch_rate']
                )
        
        except Exception as e:
            logger.warning(f"Error in channel detection: {e}")
        
        return features
    
    def _extract_multi_signal_bursts(self, txns: pd.DataFrame, date_col: str) -> Dict[str, float]:
        """Detect bursts using multiple signals - union of indicators."""
        features = {
            'multi_signal_burst_days': 0.0,
            'multi_signal_burst_score': 0.0,
            'burst_signal_count': 0.0,
        }
        
        try:
            txns_sorted = txns.sort_values(date_col)
            daily_groups = txns_sorted.groupby(pd.to_datetime(txns_sorted[date_col]).dt.date)
            
            burst_days = []
            
            for date, group in daily_groups:
                signals = 0
                
                # Signal 1: Transaction count spike
                daily_txn_count = len(group)
                avg_daily = len(txns_sorted) / len(daily_groups)
                if daily_txn_count > avg_daily * 2:
                    signals += 1
                
                # Signal 2: Amount spike
                amount_col = self._find_amount_column(txns)
                if amount_col:
                    daily_amount = group[amount_col].sum()
                    avg_amount = txns_sorted[amount_col].sum() / len(daily_groups)
                    if daily_amount > avg_amount * 2:
                        signals += 1
                
                # Signal 3: Counterparty spike
                counterparty_col = None
                for col in ['counterparty', 'recipient', 'sender', 'beneficiary']:
                    if col in group.columns:
                        counterparty_col = col
                        break
                
                if counterparty_col:
                    daily_counterparties = group[counterparty_col].nunique()
                    avg_counterparties = txns_sorted[counterparty_col].nunique() / len(daily_groups)
                    if daily_counterparties > avg_counterparties * 2:
                        signals += 1
                
                # Signal 4: Channel diversity
                channel_col = None
                for col in ['channel', 'transaction_channel', 'mode']:
                    if col in group.columns:
                        channel_col = col
                        break
                
                if channel_col:
                    daily_channels = group[channel_col].nunique()
                    if daily_channels >= 3:
                        signals += 1
                
                # If multiple signals, it's a burst day
                if signals >= 2:
                    burst_days.append(date)
            
            features['multi_signal_burst_days'] = float(len(burst_days))
            features['burst_signal_count'] = float(len(burst_days))
            
            if len(burst_days) > 0:
                features['multi_signal_burst_score'] = len(burst_days) / len(daily_groups)
        
        except Exception as e:
            logger.warning(f"Error in multi-signal burst detection: {e}")
        
        return features
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return empty feature dict."""
        return {
            'max_inactivity_gap_days': 0.0,
            'avg_inactivity_gap_days': 0.0,
            'burst_after_gap_ratio': 0.0,
            'txn_after_gap_3day': 0.0,
            'amount_after_gap_3day': 0.0,
            'freeze_unfreeze_score': 0.0,
            'pass_through_ratio_24h': 0.0,
            'pass_through_ratio_6h': 0.0,
            'avg_time_to_send_hours': 0.0,
            'pass_through_score': 0.0,
            'rapid_relay_count': 0.0,
            'unique_counterparties_total': 0.0,
            'unique_counterparties_7day': 0.0,
            'unique_counterparties_1day': 0.0,
            'counterparty_growth_rate': 0.0,
            'daily_counterparty_spike': 0.0,
            'counterparty_explosion_score': 0.0,
            'unique_channels': 0.0,
            'channel_entropy': 0.0,
            'channel_switch_rate': 0.0,
            'channels_7day': 0.0,
            'channel_switching_score': 0.0,
            'multi_signal_burst_days': 0.0,
            'multi_signal_burst_score': 0.0,
            'burst_signal_count': 0.0,
        }
    
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
