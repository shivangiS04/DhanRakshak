"""
Behavioral Pattern Detection for Mule Account Detection

Detects 10 known mule behavior patterns:
1. Dormant Activation Pattern
2. Structuring Pattern
3. Rapid Pass-Through Pattern
4. Fan-In/Fan-Out Pattern
5. Geographic Anomaly Pattern
6. New Account High Value Pattern
7. Income Mismatch Pattern
8. Post-Mobile-Change Spike Pattern
9. Round Amount Pattern
10. Salary Cycle Exploitation Pattern
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PatternScore:
    """Pattern detection score"""
    pattern_name: str
    score: float  # 0-1
    confidence: float  # 0-1
    supporting_features: List[str]


class PatternDetector:
    """Detect mule behavior patterns"""
    
    def __init__(self):
        """Initialize pattern detector"""
        logger.info("PatternDetector initialized")
    
    def detect_all_patterns(self, account_id: str, transactions: pd.DataFrame, 
                           features: 'AccountFeatures', account_metadata: Optional[Dict] = None) -> Dict[str, PatternScore]:
        """
        Detect all patterns for an account.
        
        Args:
            account_id: Account ID
            transactions: Transaction dataframe
            features: AccountFeatures object
            account_metadata: Optional account metadata
            
        Returns:
            Dictionary of pattern_name -> PatternScore
        """
        patterns = {}
        
        patterns['dormant_activation'] = self.detect_dormant_activation(transactions, features)
        patterns['structuring'] = self.detect_structuring(features)
        patterns['rapid_passthrough'] = self.detect_rapid_passthrough(features)
        patterns['fan_in_fan_out'] = self.detect_fan_in_fan_out(features)
        patterns['geographic_anomaly'] = self.detect_geographic_anomaly(features)
        patterns['new_account_high_value'] = self.detect_new_account_high_value(features)
        patterns['income_mismatch'] = self.detect_income_mismatch(features, account_metadata)
        patterns['post_mobile_spike'] = self.detect_post_mobile_spike(transactions, features)
        patterns['round_amount'] = self.detect_round_amount(features)
        patterns['salary_cycle'] = self.detect_salary_cycle(transactions, features)
        
        return patterns
    
    def detect_dormant_activation(self, transactions: pd.DataFrame, features: 'AccountFeatures') -> PatternScore:
        """
        Detect dormant activation pattern.
        
        Pattern: Account inactive for N days, then sudden activity spike
        Threshold: >50% of lifetime transactions in 30-day window after dormancy
        """
        score = 0.0
        confidence = 0.0
        supporting_features = []
        
        if 'transaction_date' not in transactions.columns or len(transactions) < 2:
            return PatternScore('dormant_activation', score, confidence, supporting_features)
        
        dates = pd.to_datetime(transactions['transaction_date']).sort_values()
        
        # Find dormancy periods (gaps > 30 days)
        max_spike = 0
        for i in range(len(dates) - 1):
            gap = (dates.iloc[i+1] - dates.iloc[i]).days
            
            if gap > 30:
                # Check activity spike after dormancy
                window_start = dates.iloc[i+1]
                window_end = window_start + timedelta(days=30)
                spike_activity = sum(1 for d in dates if window_start <= d <= window_end)
                
                total_activity = len(dates)
                spike_ratio = spike_activity / total_activity if total_activity > 0 else 0
                
                if spike_ratio > 0.5:
                    max_spike = max(max_spike, spike_ratio)
                    supporting_features.append(f"Dormancy gap: {gap} days, spike ratio: {spike_ratio:.2f}")
        
        if max_spike > 0.5:
            score = min(max_spike, 1.0)
            confidence = 0.8
        
        return PatternScore('dormant_activation', score, confidence, supporting_features)
    
    def detect_structuring(self, features: 'AccountFeatures') -> PatternScore:
        """
        Detect structuring pattern.
        
        Pattern: Multiple transactions just below regulatory thresholds
        Threshold: >30% of transactions in 9k-10k range
        """
        score = 0.0
        confidence = 0.0
        supporting_features = []
        
        if features.sub_threshold_ratio > 0.3:
            score = min(features.sub_threshold_ratio, 1.0)
            confidence = 0.85
            supporting_features.append(f"Sub-threshold ratio: {features.sub_threshold_ratio:.2f}")
        
        return PatternScore('structuring', score, confidence, supporting_features)
    
    def detect_rapid_passthrough(self, features: 'AccountFeatures') -> PatternScore:
        """
        Detect rapid pass-through pattern.
        
        Pattern: High inflow-to-outflow ratio, rapid transfers
        Threshold: Ratio 0.95-1.05, avg transfer time <48 hours
        """
        score = 0.0
        confidence = 0.0
        supporting_features = []
        
        # Check inflow-outflow ratio
        if 0.95 <= features.inflow_outflow_ratio <= 1.05:
            score += 0.4
            supporting_features.append(f"Inflow-outflow ratio: {features.inflow_outflow_ratio:.2f}")
        
        # Check rapid transfer
        if features.avg_time_to_transfer_hours > 0 and features.avg_time_to_transfer_hours < 48:
            score += 0.4
            supporting_features.append(f"Avg transfer time: {features.avg_time_to_transfer_hours:.1f} hours")
        
        # Check rapid transfer ratio
        if features.rapid_transfer_ratio_48h > 0.7:
            score += 0.2
            supporting_features.append(f"Rapid transfer ratio (48h): {features.rapid_transfer_ratio_48h:.2f}")
        
        if score > 0:
            confidence = 0.8
            score = min(score, 1.0)
        
        return PatternScore('rapid_passthrough', score, confidence, supporting_features)
    
    def detect_fan_in_fan_out(self, features: 'AccountFeatures') -> PatternScore:
        """
        Detect fan-in/fan-out pattern.
        
        Pattern: Many sources and destinations, low concentration
        Threshold: >20 sources and destinations, Herfindahl <0.15
        """
        score = 0.0
        confidence = 0.0
        supporting_features = []
        
        if features.unique_sources > 20 and features.unique_destinations > 20:
            score += 0.4
            supporting_features.append(f"Sources: {features.unique_sources}, Destinations: {features.unique_destinations}")
        
        if features.source_concentration < 0.15:
            score += 0.3
            supporting_features.append(f"Low concentration: {features.source_concentration:.3f}")
        
        if features.avg_inflow_amount > 0 and features.avg_inflow_amount < 50000:
            score += 0.3
            supporting_features.append(f"Low avg amount: {features.avg_inflow_amount:.0f}")
        
        if score > 0:
            confidence = 0.75
            score = min(score, 1.0)
        
        return PatternScore('fan_in_fan_out', score, confidence, supporting_features)
    
    def detect_geographic_anomaly(self, features: 'AccountFeatures') -> PatternScore:
        """
        Detect geographic anomaly pattern.
        
        Pattern: Cross-border transactions, location mismatch
        Threshold: >50% cross-border transactions
        """
        score = 0.0
        confidence = 0.0
        supporting_features = []
        
        if features.cross_border_ratio > 0.5:
            score = min(features.cross_border_ratio, 1.0)
            confidence = 0.7
            supporting_features.append(f"Cross-border ratio: {features.cross_border_ratio:.2f}")
        
        if features.location_anomaly_score > 0.5:
            score = max(score, features.location_anomaly_score)
            confidence = 0.7
            supporting_features.append(f"Location anomaly: {features.location_anomaly_score:.2f}")
        
        return PatternScore('geographic_anomaly', score, confidence, supporting_features)
    
    def detect_new_account_high_value(self, features: 'AccountFeatures') -> PatternScore:
        """
        Detect new account high value pattern.
        
        Pattern: Young account with high transaction volume
        Threshold: Age <90 days, volume >$500k
        """
        score = 0.0
        confidence = 0.0
        supporting_features = []
        
        if features.account_age_days < 90 and features.total_inflow > 500000:
            score = 0.8
            confidence = 0.85
            supporting_features.append(f"New account ({features.account_age_days} days) with high volume (${features.total_inflow:,.0f})")
        elif features.account_age_days < 90 and features.total_inflow > 250000:
            score = 0.5
            confidence = 0.7
            supporting_features.append(f"New account ({features.account_age_days} days) with moderate volume (${features.total_inflow:,.0f})")
        
        return PatternScore('new_account_high_value', score, confidence, supporting_features)
    
    def detect_income_mismatch(self, features: 'AccountFeatures', account_metadata: Optional[Dict]) -> PatternScore:
        """
        Detect income mismatch pattern.
        
        Pattern: Transaction volume vs. stated income mismatch
        Threshold: Ratio >5x
        """
        score = 0.0
        confidence = 0.0
        supporting_features = []
        
        if account_metadata and 'annual_income' in account_metadata:
            annual_income = account_metadata.get('annual_income', 0)
            if annual_income > 0:
                ratio = features.total_inflow / annual_income
                if ratio > 5:
                    score = min(ratio / 10, 1.0)  # Normalize to 0-1
                    confidence = 0.75
                    supporting_features.append(f"Income mismatch ratio: {ratio:.1f}x")
        
        return PatternScore('income_mismatch', score, confidence, supporting_features)
    
    def detect_post_mobile_spike(self, transactions: pd.DataFrame, features: 'AccountFeatures') -> PatternScore:
        """
        Detect post-mobile-change spike pattern.
        
        Pattern: Activity spike within 7 days of mobile app registration
        Threshold: Spike magnitude >3x normal activity
        """
        score = 0.0
        confidence = 0.0
        supporting_features = []
        
        # This would require mobile registration date in metadata
        # For now, use activity spike magnitude as proxy
        if features.activity_spike_magnitude > 3:
            score = min(features.activity_spike_magnitude / 5, 1.0)
            confidence = 0.6
            supporting_features.append(f"Activity spike magnitude: {features.activity_spike_magnitude:.1f}x")
        
        return PatternScore('post_mobile_spike', score, confidence, supporting_features)
    
    def detect_round_amount(self, features: 'AccountFeatures') -> PatternScore:
        """
        Detect round amount pattern.
        
        Pattern: High percentage of round amounts
        Threshold: >40% of transactions are round amounts
        """
        score = 0.0
        confidence = 0.0
        supporting_features = []
        
        if features.round_amount_ratio > 0.4:
            score = min(features.round_amount_ratio, 1.0)
            confidence = 0.75
            supporting_features.append(f"Round amount ratio: {features.round_amount_ratio:.2f}")
        
        return PatternScore('round_amount', score, confidence, supporting_features)
    
    def detect_salary_cycle(self, transactions: pd.DataFrame, features: 'AccountFeatures') -> PatternScore:
        """
        Detect salary cycle exploitation pattern.
        
        Pattern: Transactions aligned with payroll dates
        Threshold: Regular inflow followed by rapid outflow
        """
        score = 0.0
        confidence = 0.0
        supporting_features = []
        
        if 'transaction_date' not in transactions.columns:
            return PatternScore('salary_cycle', score, confidence, supporting_features)
        
        # Check if inflows are regular (monthly pattern)
        if 'direction' in transactions.columns:
            inflows = transactions[transactions['direction'] == 'inflow'].copy()
            inflows['transaction_date'] = pd.to_datetime(inflows['transaction_date'])
            inflows = inflows.sort_values('transaction_date')
            
            if len(inflows) >= 3:
                # Check for monthly pattern
                inflow_dates = inflows['transaction_date'].dt.day.values
                if len(set(inflow_dates)) <= 3:  # All inflows on similar days of month
                    score = 0.6
                    confidence = 0.7
                    supporting_features.append(f"Regular inflow pattern detected")
        
        return PatternScore('salary_cycle', score, confidence, supporting_features)


class CompositeAnomalyScorer:
    """Combine multiple patterns into composite anomaly score"""
    
    def __init__(self, pattern_weights: Optional[Dict[str, float]] = None):
        """
        Initialize scorer.
        
        Args:
            pattern_weights: Optional weights for each pattern (default: equal)
        """
        self.pattern_weights = pattern_weights or {
            'dormant_activation': 1.0,
            'structuring': 1.2,
            'rapid_passthrough': 1.1,
            'fan_in_fan_out': 1.0,
            'geographic_anomaly': 0.8,
            'new_account_high_value': 0.9,
            'income_mismatch': 1.0,
            'post_mobile_spike': 0.7,
            'round_amount': 0.9,
            'salary_cycle': 0.8
        }
    
    def compute_composite_score(self, pattern_scores: Dict[str, PatternScore]) -> Tuple[float, float]:
        """
        Compute composite anomaly score from individual patterns.
        
        Args:
            pattern_scores: Dictionary of pattern_name -> PatternScore
            
        Returns:
            Tuple of (composite_score, confidence)
        """
        weighted_scores = []
        weighted_confidences = []
        total_weight = 0
        
        for pattern_name, pattern_score in pattern_scores.items():
            weight = self.pattern_weights.get(pattern_name, 1.0)
            
            weighted_scores.append(pattern_score.score * weight)
            weighted_confidences.append(pattern_score.confidence * weight)
            total_weight += weight
        
        if total_weight == 0:
            return 0.0, 0.0
        
        composite_score = sum(weighted_scores) / total_weight
        composite_confidence = sum(weighted_confidences) / total_weight
        
        # Normalize to 0-1
        composite_score = min(composite_score, 1.0)
        composite_confidence = min(composite_confidence, 1.0)
        
        return composite_score, composite_confidence
