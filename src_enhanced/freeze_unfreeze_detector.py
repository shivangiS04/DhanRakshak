"""
Freeze/Unfreeze Pattern Detection Module

Detects suspicious freeze and unfreeze patterns in accounts:
- Account freeze followed by sudden activity
- Multiple freeze/unfreeze cycles
- Freeze during high-value transactions
- Unfreeze before suspicious activity
- Timing correlation with other accounts
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class FreezeUnfreezePattern:
    """Freeze/unfreeze pattern detection result"""
    account_id: str
    pattern_type: str  # 'freeze_before_activity', 'unfreeze_spike', 'multiple_cycles', 'coordinated'
    freeze_date: Optional[datetime]
    unfreeze_date: Optional[datetime]
    freeze_duration_days: int
    activity_after_unfreeze: int
    risk_score: float  # 0-1
    confidence: float
    supporting_evidence: List[str]


class FreezeUnfreezeDetector:
    """Detect freeze/unfreeze patterns"""
    
    def __init__(self):
        """Initialize detector"""
        logger.info("FreezeUnfreezeDetector initialized")
        self.freeze_events = {}
        self.unfreeze_events = {}
    
    def detect_patterns(self, account_id: str, transactions: pd.DataFrame,
                       account_status_history: Optional[pd.DataFrame] = None) -> List[FreezeUnfreezePattern]:
        """
        Detect freeze/unfreeze patterns for an account.
        
        Args:
            account_id: Account ID
            transactions: Transaction dataframe
            account_status_history: Optional account status history
            
        Returns:
            List of FreezeUnfreezePattern objects
        """
        patterns = []
        
        # Extract freeze/unfreeze events
        freeze_events = self._extract_freeze_events(account_id, transactions, account_status_history)
        
        if not freeze_events:
            return patterns
        
        # Analyze each freeze event
        for freeze_event in freeze_events:
            # Pattern 1: Freeze before high-value activity
            if self._is_freeze_before_activity(freeze_event, transactions):
                patterns.append(self._create_pattern(
                    account_id, freeze_event, 'freeze_before_activity', transactions
                ))
            
            # Pattern 2: Unfreeze followed by spike
            if self._is_unfreeze_spike(freeze_event, transactions):
                patterns.append(self._create_pattern(
                    account_id, freeze_event, 'unfreeze_spike', transactions
                ))
            
            # Pattern 3: Multiple freeze/unfreeze cycles
            if self._is_multiple_cycles(freeze_event, freeze_events):
                patterns.append(self._create_pattern(
                    account_id, freeze_event, 'multiple_cycles', transactions
                ))
        
        logger.info(f"Detected {len(patterns)} freeze/unfreeze patterns for {account_id}")
        return patterns
    
    def _extract_freeze_events(self, account_id: str, transactions: pd.DataFrame,
                              account_status_history: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        Extract freeze/unfreeze events from transaction data.
        
        Args:
            account_id: Account ID
            transactions: Transaction dataframe
            account_status_history: Optional account status history
            
        Returns:
            List of freeze event dictionaries
        """
        freeze_events = []
        
        # Method 1: Detect from transaction gaps and status changes
        if 'transaction_date' in transactions.columns:
            transactions = transactions.copy()
            transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
            sorted_txns = transactions.sort_values('transaction_date')
            
            dates = sorted_txns['transaction_date'].values
            
            # Find gaps > 30 days (potential freeze periods)
            for i in range(len(dates) - 1):
                gap_days = (pd.Timestamp(dates[i+1]) - pd.Timestamp(dates[i])).days
                
                if gap_days > 30:
                    freeze_events.append({
                        'freeze_date': pd.Timestamp(dates[i]),
                        'unfreeze_date': pd.Timestamp(dates[i+1]),
                        'freeze_duration_days': gap_days,
                        'source': 'transaction_gap'
                    })
        
        # Method 2: Use account status history if available
        if account_status_history is not None and 'status' in account_status_history.columns:
            freeze_events.extend(self._extract_from_status_history(account_status_history))
        
        return freeze_events
    
    def _extract_from_status_history(self, status_history: pd.DataFrame) -> List[Dict]:
        """Extract freeze events from account status history"""
        freeze_events = []
        
        if 'date' not in status_history.columns or 'status' not in status_history.columns:
            return freeze_events
        
        status_history = status_history.copy()
        status_history['date'] = pd.to_datetime(status_history['date'])
        status_history = status_history.sort_values('date')
        
        for i in range(len(status_history) - 1):
            current_status = status_history.iloc[i]['status']
            next_status = status_history.iloc[i+1]['status']
            
            # Detect freeze (transition to frozen state)
            if self._is_frozen_status(next_status) and not self._is_frozen_status(current_status):
                freeze_date = status_history.iloc[i]['date']
                unfreeze_date = status_history.iloc[i+1]['date']
                
                freeze_events.append({
                    'freeze_date': freeze_date,
                    'unfreeze_date': unfreeze_date,
                    'freeze_duration_days': (unfreeze_date - freeze_date).days,
                    'source': 'status_history'
                })
        
        return freeze_events
    
    def _is_frozen_status(self, status: str) -> bool:
        """Check if status indicates frozen account"""
        frozen_keywords = ['frozen', 'blocked', 'suspended', 'inactive', 'dormant']
        return any(keyword in str(status).lower() for keyword in frozen_keywords)
    
    def _is_freeze_before_activity(self, freeze_event: Dict, transactions: pd.DataFrame) -> bool:
        """
        Check if freeze occurs before high-value activity.
        
        Args:
            freeze_event: Freeze event dictionary
            transactions: Transaction dataframe
            
        Returns:
            True if pattern detected
        """
        if 'transaction_date' not in transactions.columns or 'amount' not in transactions.columns:
            return False
        
        unfreeze_date = freeze_event['unfreeze_date']
        window_end = unfreeze_date + timedelta(days=7)
        
        # Get transactions after unfreeze
        post_unfreeze = transactions[
            (pd.to_datetime(transactions['transaction_date']) >= unfreeze_date) &
            (pd.to_datetime(transactions['transaction_date']) <= window_end)
        ]
        
        if len(post_unfreeze) == 0:
            return False
        
        # Check for high-value transactions
        high_value_threshold = transactions['amount'].quantile(0.75)
        high_value_count = (post_unfreeze['amount'].abs() > high_value_threshold).sum()
        
        return high_value_count > 0
    
    def _is_unfreeze_spike(self, freeze_event: Dict, transactions: pd.DataFrame) -> bool:
        """
        Check if unfreeze is followed by activity spike.
        
        Args:
            freeze_event: Freeze event dictionary
            transactions: Transaction dataframe
            
        Returns:
            True if pattern detected
        """
        if 'transaction_date' not in transactions.columns:
            return False
        
        freeze_date = freeze_event['freeze_date']
        unfreeze_date = freeze_event['unfreeze_date']
        
        # Activity before freeze
        before_freeze = transactions[
            pd.to_datetime(transactions['transaction_date']) < freeze_date
        ]
        
        # Activity after unfreeze (7-day window)
        after_unfreeze = transactions[
            (pd.to_datetime(transactions['transaction_date']) >= unfreeze_date) &
            (pd.to_datetime(transactions['transaction_date']) <= unfreeze_date + timedelta(days=7))
        ]
        
        if len(before_freeze) == 0:
            return False
        
        # Calculate spike ratio
        avg_before = len(before_freeze) / max((freeze_date - pd.to_datetime(transactions['transaction_date']).min()).days, 1)
        avg_after = len(after_unfreeze) / 7
        
        spike_ratio = avg_after / (avg_before + 1e-8)
        
        return spike_ratio > 3  # 3x spike
    
    def _is_multiple_cycles(self, freeze_event: Dict, all_freeze_events: List[Dict]) -> bool:
        """
        Check if account has multiple freeze/unfreeze cycles.
        
        Args:
            freeze_event: Current freeze event
            all_freeze_events: All freeze events for account
            
        Returns:
            True if multiple cycles detected
        """
        return len(all_freeze_events) > 2
    
    def _create_pattern(self, account_id: str, freeze_event: Dict, pattern_type: str,
                       transactions: pd.DataFrame) -> FreezeUnfreezePattern:
        """Create FreezeUnfreezePattern object"""
        
        # Calculate activity after unfreeze
        unfreeze_date = freeze_event['unfreeze_date']
        window_end = unfreeze_date + timedelta(days=7)
        
        activity_after = len(transactions[
            (pd.to_datetime(transactions['transaction_date']) >= unfreeze_date) &
            (pd.to_datetime(transactions['transaction_date']) <= window_end)
        ])
        
        # Risk scoring
        risk_score = self._calculate_risk_score(pattern_type, freeze_event, activity_after)
        
        evidence = self._generate_evidence(pattern_type, freeze_event, activity_after)
        
        return FreezeUnfreezePattern(
            account_id=account_id,
            pattern_type=pattern_type,
            freeze_date=freeze_event['freeze_date'],
            unfreeze_date=freeze_event['unfreeze_date'],
            freeze_duration_days=freeze_event['freeze_duration_days'],
            activity_after_unfreeze=activity_after,
            risk_score=risk_score,
            confidence=0.75,
            supporting_evidence=evidence
        )
    
    def _calculate_risk_score(self, pattern_type: str, freeze_event: Dict, activity_after: int) -> float:
        """Calculate risk score for pattern"""
        base_scores = {
            'freeze_before_activity': 0.7,
            'unfreeze_spike': 0.8,
            'multiple_cycles': 0.6,
            'coordinated': 0.9
        }
        
        base_score = base_scores.get(pattern_type, 0.5)
        
        # Adjust based on freeze duration
        duration_factor = min(freeze_event['freeze_duration_days'] / 90, 1.0)
        
        # Adjust based on post-unfreeze activity
        activity_factor = min(activity_after / 50, 1.0)
        
        risk_score = base_score * (0.5 + 0.25 * duration_factor + 0.25 * activity_factor)
        
        return min(risk_score, 1.0)
    
    def _generate_evidence(self, pattern_type: str, freeze_event: Dict, activity_after: int) -> List[str]:
        """Generate supporting evidence for pattern"""
        evidence = []
        
        if pattern_type == 'freeze_before_activity':
            evidence.append(f"Account frozen for {freeze_event['freeze_duration_days']} days")
            evidence.append(f"High-value activity detected within 7 days of unfreeze")
        
        elif pattern_type == 'unfreeze_spike':
            evidence.append(f"Activity spike after unfreeze: {activity_after} transactions in 7 days")
            evidence.append(f"Freeze duration: {freeze_event['freeze_duration_days']} days")
        
        elif pattern_type == 'multiple_cycles':
            evidence.append("Multiple freeze/unfreeze cycles detected")
            evidence.append("Pattern suggests deliberate account manipulation")
        
        return evidence


class CoordinatedFreezeDetector:
    """Detect coordinated freeze/unfreeze patterns across accounts"""
    
    @staticmethod
    def detect_coordinated_patterns(account_freeze_events: Dict[str, List[Dict]],
                                   time_window_days: int = 7) -> List[Dict]:
        """
        Detect coordinated freeze/unfreeze patterns across multiple accounts.
        
        Args:
            account_freeze_events: Dictionary of account_id -> list of freeze events
            time_window_days: Time window for coordination detection
            
        Returns:
            List of coordinated pattern dictionaries
        """
        coordinated_patterns = []
        
        # Group freeze events by date
        freeze_timeline = {}
        for account_id, events in account_freeze_events.items():
            for event in events:
                freeze_date = event['freeze_date']
                if freeze_date not in freeze_timeline:
                    freeze_timeline[freeze_date] = []
                freeze_timeline[freeze_date].append(account_id)
        
        # Find clusters of simultaneous freezes
        for freeze_date, accounts in freeze_timeline.items():
            if len(accounts) >= 3:  # 3+ accounts frozen at same time
                coordinated_patterns.append({
                    'pattern_type': 'coordinated_freeze',
                    'freeze_date': freeze_date,
                    'accounts': accounts,
                    'account_count': len(accounts),
                    'risk_score': min(len(accounts) / 10, 1.0)
                })
        
        logger.info(f"Detected {len(coordinated_patterns)} coordinated freeze patterns")
        return coordinated_patterns
