"""
Temporal Analysis Module for Mule Account Detection

Identifies suspicious activity windows and calculates temporal IoU scores:
- Sliding window analysis (7-day, 30-day, 90-day)
- Anomaly score per window
- Suspicious window detection
- Temporal IoU calculation
- Confidence interval estimation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SuspiciousWindow:
    """Suspicious activity window"""
    account_id: str
    suspicious_start: datetime
    suspicious_end: datetime
    anomaly_score: float
    confidence: float
    pattern_type: str
    supporting_features: List[str]


class TemporalAnalyzer:
    """Analyze temporal patterns and detect suspicious windows"""
    
    def __init__(self):
        """Initialize temporal analyzer"""
        logger.info("TemporalAnalyzer initialized")
    
    def detect_suspicious_window(self, account_id: str, transactions: pd.DataFrame,
                                features: 'AccountFeatures',
                                window_sizes: Optional[List[int]] = None) -> Optional[SuspiciousWindow]:
        """
        Detect suspicious activity window for an account.
        
        Args:
            account_id: Account ID
            transactions: Transaction dataframe
            features: AccountFeatures object
            window_sizes: List of window sizes in days (default: [7, 30, 90])
            
        Returns:
            SuspiciousWindow object or None
        """
        if window_sizes is None:
            window_sizes = [7, 30, 90]
        
        if 'transaction_date' not in transactions.columns or len(transactions) < 2:
            return None
        
        # Sort transactions by date
        transactions = transactions.copy()
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        sorted_txns = transactions.sort_values('transaction_date')
        
        best_window = None
        max_anomaly_score = 0
        
        # Try each window size
        for window_size in window_sizes:
            window = self._find_best_window(account_id, sorted_txns, features, window_size)
            
            if window and window.anomaly_score > max_anomaly_score:
                max_anomaly_score = window.anomaly_score
                best_window = window
        
        return best_window
    
    def _find_best_window(self, account_id: str, transactions: pd.DataFrame,
                         features: 'AccountFeatures', window_size_days: int) -> Optional[SuspiciousWindow]:
        """
        Find the best suspicious window of given size.
        
        Args:
            account_id: Account ID
            transactions: Sorted transaction dataframe
            features: AccountFeatures object
            window_size_days: Window size in days
            
        Returns:
            SuspiciousWindow with highest anomaly score
        """
        if len(transactions) == 0:
            return None
        
        dates = transactions['transaction_date'].values
        max_anomaly = 0
        best_start = None
        best_end = None
        
        # Sliding window
        for i in range(len(dates)):
            window_start = pd.Timestamp(dates[i])
            window_end = window_start + timedelta(days=window_size_days)
            
            # Get transactions in window
            window_txns = transactions[
                (transactions['transaction_date'] >= window_start) &
                (transactions['transaction_date'] <= window_end)
            ]
            
            if len(window_txns) == 0:
                continue
            
            # Calculate anomaly score for this window
            anomaly_score = self._calculate_window_anomaly_score(window_txns, features)
            
            if anomaly_score > max_anomaly:
                max_anomaly = anomaly_score
                best_start = window_start
                best_end = window_end
        
        if best_start is None:
            return None
        
        # Determine pattern type
        pattern_type = self._determine_pattern_type(transactions, best_start, best_end)
        
        return SuspiciousWindow(
            account_id=account_id,
            suspicious_start=best_start.to_pydatetime(),
            suspicious_end=best_end.to_pydatetime(),
            anomaly_score=min(max_anomaly, 1.0),
            confidence=0.7,
            pattern_type=pattern_type,
            supporting_features=[]
        )
    
    def _calculate_window_anomaly_score(self, window_txns: pd.DataFrame, 
                                       features: 'AccountFeatures') -> float:
        """
        Calculate anomaly score for a transaction window.
        
        Args:
            window_txns: Transactions in window
            features: AccountFeatures object
            
        Returns:
            Anomaly score (0-1)
        """
        score = 0.0
        
        # High transaction volume in window
        if len(window_txns) > features.total_transactions * 0.3:
            score += 0.3
        
        # High amount in window
        if 'amount' in window_txns.columns:
            window_amount = window_txns['amount'].abs().sum()
            if window_amount > features.total_inflow * 0.4:
                score += 0.3
        
        # Rapid transfers in window
        if 'direction' in window_txns.columns:
            inflows = len(window_txns[window_txns['direction'] == 'inflow'])
            outflows = len(window_txns[window_txns['direction'] == 'outflow'])
            
            if inflows > 0 and outflows > 0:
                ratio = min(inflows, outflows) / max(inflows, outflows)
                if ratio > 0.7:  # Balanced inflow/outflow
                    score += 0.2
        
        # Unusual timing
        if 'transaction_date' in window_txns.columns:
            hours = pd.to_datetime(window_txns['transaction_date']).dt.hour
            if (hours < 6).sum() > len(window_txns) * 0.3:  # Off-hours activity
                score += 0.2
        
        return min(score, 1.0)
    
    def _determine_pattern_type(self, transactions: pd.DataFrame, 
                               window_start: pd.Timestamp, 
                               window_end: pd.Timestamp) -> str:
        """
        Determine the type of suspicious pattern in window.
        
        Args:
            transactions: All transactions
            window_start: Window start date
            window_end: Window end date
            
        Returns:
            Pattern type string
        """
        window_txns = transactions[
            (transactions['transaction_date'] >= window_start) &
            (transactions['transaction_date'] <= window_end)
        ]
        
        if len(window_txns) == 0:
            return 'unknown'
        
        # Check for rapid pass-through
        if 'direction' in window_txns.columns:
            inflows = len(window_txns[window_txns['direction'] == 'inflow'])
            outflows = len(window_txns[window_txns['direction'] == 'outflow'])
            
            if inflows > 0 and outflows > 0:
                ratio = min(inflows, outflows) / max(inflows, outflows)
                if ratio > 0.7:
                    return 'rapid_passthrough'
        
        # Check for high volume
        if 'amount' in window_txns.columns:
            if window_txns['amount'].abs().sum() > 1000000:
                return 'high_volume'
        
        # Check for many transactions
        if len(window_txns) > 100:
            return 'high_frequency'
        
        return 'general_anomaly'


class TemporalIoUCalculator:
    """Calculate Temporal Intersection over Union scores"""
    
    @staticmethod
    def calculate_iou(predicted_window: Tuple[datetime, datetime],
                     actual_window: Tuple[datetime, datetime]) -> float:
        """
        Calculate Temporal IoU between predicted and actual windows.
        
        Args:
            predicted_window: Tuple of (start, end) for predicted window
            actual_window: Tuple of (start, end) for actual window
            
        Returns:
            IoU score (0-1)
        """
        pred_start, pred_end = predicted_window
        actual_start, actual_end = actual_window
        
        # Calculate intersection
        intersection_start = max(pred_start, actual_start)
        intersection_end = min(pred_end, actual_end)
        
        if intersection_start > intersection_end:
            intersection_days = 0
        else:
            intersection_days = (intersection_end - intersection_start).days
        
        # Calculate union
        union_start = min(pred_start, actual_start)
        union_end = max(pred_end, actual_end)
        union_days = (union_end - union_start).days
        
        # Calculate IoU
        if union_days == 0:
            return 0.0
        
        iou = intersection_days / union_days
        return min(iou, 1.0)
    
    @staticmethod
    def calculate_iou_batch(predicted_windows: Dict[str, Tuple[datetime, datetime]],
                           actual_windows: Dict[str, Tuple[datetime, datetime]]) -> Dict[str, float]:
        """
        Calculate Temporal IoU for multiple accounts.
        
        Args:
            predicted_windows: Dictionary of account_id -> (start, end)
            actual_windows: Dictionary of account_id -> (start, end)
            
        Returns:
            Dictionary of account_id -> IoU score
        """
        iou_scores = {}
        
        for account_id in predicted_windows:
            if account_id in actual_windows:
                iou = TemporalIoUCalculator.calculate_iou(
                    predicted_windows[account_id],
                    actual_windows[account_id]
                )
                iou_scores[account_id] = iou
        
        return iou_scores


class WindowConfidenceCalculator:
    """Calculate confidence intervals for suspicious windows"""
    
    @staticmethod
    def calculate_confidence(window_txns: pd.DataFrame, 
                            all_txns: pd.DataFrame,
                            anomaly_score: float) -> float:
        """
        Calculate confidence for suspicious window.
        
        Args:
            window_txns: Transactions in window
            all_txns: All transactions
            anomaly_score: Anomaly score for window
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.0
        
        # More transactions in window = higher confidence
        if len(all_txns) > 0:
            txn_ratio = len(window_txns) / len(all_txns)
            confidence += min(txn_ratio, 0.3)
        
        # Higher anomaly score = higher confidence
        confidence += anomaly_score * 0.4
        
        # Consistency check
        if 'amount' in window_txns.columns:
            amount_std = window_txns['amount'].std()
            if amount_std > 0:
                confidence += 0.3
        
        return min(confidence, 1.0)
