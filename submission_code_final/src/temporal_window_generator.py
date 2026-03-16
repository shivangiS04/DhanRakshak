"""
Temporal Window Generator for Mule Detection

Generates suspicious activity windows (suspicious_start, suspicious_end) for predicted mule accounts.
Uses transaction data to identify periods of suspicious activity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TemporalWindowGenerator:
    """Generates suspicious activity windows for mule accounts."""
    
    def __init__(self, data_dir: str = "/Users/shivangisingh/Desktop/Archive"):
        """Initialize with data directory."""
        self.data_dir = Path(data_dir)
        self.transactions = None
        self.windows = {}
        
    def load_transactions(self, sample_size: int = None) -> bool:
        """Load transaction data."""
        try:
            from glob import glob
            logger.info("Loading transaction data...")
            
            parts = sorted(glob(str(self.data_dir / "transactions/batch-*/part_*.parquet")))
            if not parts:
                logger.warning("No transaction files found")
                return False
            
            # Load all parts or sample
            if sample_size:
                parts = parts[:sample_size]
            
            dfs = []
            for i, part in enumerate(parts):
                if i % 50 == 0:
                    logger.info(f"Loading part {i}/{len(parts)}")
                dfs.append(pd.read_parquet(part))
            
            self.transactions = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(self.transactions)} transactions")
            
            # Convert timestamp
            self.transactions['transaction_timestamp'] = pd.to_datetime(
                self.transactions['transaction_timestamp']
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading transactions: {e}")
            return False
    
    def find_suspicious_window(self, account_id: str, mule_score: float) -> Tuple[Optional[str], Optional[str]]:
        """
        Find suspicious activity window for an account.
        
        Returns: (suspicious_start, suspicious_end) as ISO format strings
        """
        if self.transactions is None or self.transactions.empty:
            return None, None
        
        try:
            # Get transactions for this account
            acct_txns = self.transactions[self.transactions['account_id'] == account_id]
            
            if acct_txns.empty:
                return None, None
            
            # Sort by timestamp
            acct_txns = acct_txns.sort_values('transaction_timestamp')
            
            # Find suspicious period based on mule score
            # Higher score = more suspicious = narrower window
            if mule_score > 0.7:
                # High-risk: Find peak activity period
                window = self._find_peak_activity(acct_txns)
            elif mule_score > 0.5:
                # Medium-risk: Find sustained activity period
                window = self._find_sustained_activity(acct_txns)
            else:
                # Low-risk: Find any anomalous period
                window = self._find_anomalous_period(acct_txns)
            
            if window:
                start_ts, end_ts = window
                return start_ts.isoformat(), end_ts.isoformat()
            
            return None, None
            
        except Exception as e:
            logger.debug(f"Error finding window for {account_id}: {e}")
            return None, None
    
    def _find_peak_activity(self, txns: pd.DataFrame) -> Optional[Tuple]:
        """Find peak activity period (highest transaction concentration)."""
        if txns.empty:
            return None
        
        # Group by day and count transactions
        txns['date'] = txns['transaction_timestamp'].dt.date
        daily_counts = txns.groupby('date').size()
        
        if daily_counts.empty:
            return None
        
        # Find day with most transactions
        peak_date = daily_counts.idxmax()
        peak_txns = txns[txns['date'] == peak_date]
        
        if peak_txns.empty:
            return None
        
        # Return time range for that day
        start = peak_txns['transaction_timestamp'].min()
        end = peak_txns['transaction_timestamp'].max()
        
        return (start, end)
    
    def _find_sustained_activity(self, txns: pd.DataFrame) -> Optional[Tuple]:
        """Find sustained activity period (multiple days with high activity)."""
        if txns.empty:
            return None
        
        # Group by day
        txns['date'] = txns['transaction_timestamp'].dt.date
        daily_counts = txns.groupby('date').size()
        
        if daily_counts.empty:
            return None
        
        # Find consecutive days with activity
        dates = sorted(daily_counts.index)
        
        # Find longest consecutive period
        max_start = dates[0]
        max_end = dates[0]
        current_start = dates[0]
        current_end = dates[0]
        
        for i in range(1, len(dates)):
            if (dates[i] - dates[i-1]).days <= 1:
                current_end = dates[i]
            else:
                if (current_end - current_start).days > (max_end - max_start).days:
                    max_start = current_start
                    max_end = current_end
                current_start = dates[i]
                current_end = dates[i]
        
        # Get transactions for this period
        period_txns = txns[(txns['date'] >= max_start) & (txns['date'] <= max_end)]
        
        if period_txns.empty:
            return None
        
        start = period_txns['transaction_timestamp'].min()
        end = period_txns['transaction_timestamp'].max()
        
        return (start, end)
    
    def _find_anomalous_period(self, txns: pd.DataFrame) -> Optional[Tuple]:
        """Find any anomalous period (unusual transaction pattern)."""
        if txns.empty:
            return None
        
        # Group by day
        txns['date'] = txns['transaction_timestamp'].dt.date
        daily_counts = txns.groupby('date').size()
        
        if daily_counts.empty:
            return None
        
        # Find day with above-average activity
        mean_count = daily_counts.mean()
        std_count = daily_counts.std()
        
        if std_count == 0:
            # All days have same activity, use first day
            anomalous_date = daily_counts.index[0]
        else:
            # Find day that's 1+ std above mean
            threshold = mean_count + std_count
            anomalous_dates = daily_counts[daily_counts >= threshold].index
            
            if len(anomalous_dates) == 0:
                # Use day with highest activity
                anomalous_date = daily_counts.idxmax()
            else:
                anomalous_date = anomalous_dates[0]
        
        # Get transactions for that day
        day_txns = txns[txns['date'] == anomalous_date]
        
        if day_txns.empty:
            return None
        
        start = day_txns['transaction_timestamp'].min()
        end = day_txns['transaction_timestamp'].max()
        
        return (start, end)
    
    def generate_windows(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate temporal windows for all predictions."""
        logger.info(f"Generating temporal windows for {len(predictions_df)} accounts...")
        
        suspicious_starts = []
        suspicious_ends = []
        
        for idx, row in predictions_df.iterrows():
            if idx % 5000 == 0:
                logger.info(f"Processing {idx}/{len(predictions_df)}")
            
            account_id = row['account_id']
            mule_score = row['is_mule']
            
            # Only generate windows for predicted mules (score > 0.5)
            if mule_score > 0.5:
                start, end = self.find_suspicious_window(account_id, mule_score)
                suspicious_starts.append(start)
                suspicious_ends.append(end)
            else:
                suspicious_starts.append(None)
                suspicious_ends.append(None)
        
        # Add to dataframe
        predictions_df['suspicious_start'] = suspicious_starts
        predictions_df['suspicious_end'] = suspicious_ends
        
        logger.info(f"Generated windows for {sum(1 for s in suspicious_starts if s)} accounts")
        
        return predictions_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    generator = TemporalWindowGenerator()
    if generator.load_transactions(sample_size=20):
        print("✓ Transactions loaded successfully")
        
        # Test with a sample account
        test_acct = "ACCT_000000"
        start, end = generator.find_suspicious_window(test_acct, 0.7)
        print(f"\nSample window for {test_acct}:")
        print(f"  Start: {start}")
        print(f"  End: {end}")
