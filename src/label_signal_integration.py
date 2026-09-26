"""
Label Signal Integration Module

Integrates pre-computed label signals into the feature engineering pipeline.
Label signals provide behavioral indicators for mule detection based on:
- Account status (frozen/history)
- Activity patterns
- Transaction volume
- Amount patterns
- KYC compliance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class LabelSignalIntegrator:
    """Integrates label signals into feature engineering."""
    
    def __init__(self, signals_path: str = "output/label_signals.csv"):
        """Initialize with path to label signals file."""
        self.signals_path = Path(signals_path)
        self.signals_df = None
        self.signal_dict = {}
        
    def load_signals(self) -> bool:
        """Load label signals from CSV."""
        try:
            if not self.signals_path.exists():
                logger.warning(f"Label signals file not found: {self.signals_path}")
                return False
            
            self.signals_df = pd.read_csv(self.signals_path)
            logger.info(f"Loaded label signals for {len(self.signals_df)} accounts")
            
            # Create lookup dictionary for fast access
            self.signal_dict = {}
            for _, row in self.signals_df.iterrows():
                self.signal_dict[row['account_id']] = {
                    'account_status_signal': row['account_status_signal'],
                    'activity_signal': row['activity_signal'],
                    'txn_count_signal': row['txn_count_signal'],
                    'amount_signal': row['amount_signal'],
                    'kyc_signal': row['kyc_signal'],
                    'composite_signal': row['composite_signal']
                }
            
            logger.info(f"Created lookup dictionary for {len(self.signal_dict)} accounts")
            return True
            
        except Exception as e:
            logger.error(f"Error loading label signals: {e}")
            return False
    
    def get_signals(self, account_id: str) -> Dict[str, float]:
        """Get signals for a specific account."""
        if account_id in self.signal_dict:
            return self.signal_dict[account_id]
        else:
            # Return zero signals if account not found
            return {
                'account_status_signal': 0.0,
                'activity_signal': 0.0,
                'txn_count_signal': 0.0,
                'amount_signal': 0.0,
                'kyc_signal': 0.0,
                'composite_signal': 0.0
            }
    
    def get_composite_signal(self, account_id: str) -> float:
        """Get composite signal for an account."""
        signals = self.get_signals(account_id)
        return signals['composite_signal']
    
    def merge_signals_with_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Merge label signals with feature dataframe."""
        if self.signals_df is None:
            logger.warning("Label signals not loaded")
            return features_df
        
        # Merge on account_id
        merged = features_df.merge(
            self.signals_df[['account_id', 'composite_signal']],
            on='account_id',
            how='left'
        )
        
        # Fill missing values with 0
        merged['composite_signal'] = merged['composite_signal'].fillna(0.0)
        
        logger.info(f"Merged signals with {len(merged)} features")
        return merged
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded signals."""
        if self.signals_df is None:
            return {}
        
        return {
            'total_accounts': len(self.signals_df),
            'accounts_with_signals': (self.signals_df['composite_signal'] > 0).sum(),
            'mean_composite': self.signals_df['composite_signal'].mean(),
            'max_composite': self.signals_df['composite_signal'].max(),
            'high_risk_count': (self.signals_df['composite_signal'] > 0.5).sum(),
            'medium_risk_count': ((self.signals_df['composite_signal'] > 0.3) & 
                                 (self.signals_df['composite_signal'] <= 0.5)).sum(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    integrator = LabelSignalIntegrator()
    if integrator.load_signals():
        stats = integrator.get_statistics()
        print("\nLabel Signal Statistics:")
        for key, val in stats.items():
            print(f"  {key}: {val}")
        
        # Test getting signals for a few accounts
        print("\nSample Signals:")
        for acct_id in ['ACCT_000000', 'ACCT_000718', 'ACCT_001097']:
            signals = integrator.get_signals(acct_id)
            print(f"  {acct_id}: composite={signals['composite_signal']:.4f}")
