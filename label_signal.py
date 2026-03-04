"""Label Signal Generator for Mule Account Detection"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelSignalGenerator:
    """Generates behavioral signals for mule account detection."""
    
    def __init__(self, data_dir: str = "/Users/shivangisingh/Desktop/Archive"):
        self.data_dir = Path(data_dir)
        self.accounts = None
        self.transactions = None
        self.train_labels = None
        
    def load_data(self):
        """Load required data files."""
        logger.info("Loading data files...")
        self.accounts = pd.read_parquet(self.data_dir / "accounts.parquet")
        self.train_labels = pd.read_parquet(self.data_dir / "train_labels.parquet")
        
        from glob import glob
        parts = sorted(glob(str(self.data_dir / "transactions/batch-*/part_*.parquet")))
        if parts:
            self.transactions = pd.concat(
                [pd.read_parquet(p) for p in parts[:5]],
                ignore_index=True
            )
    
    def account_status_signal(self) -> pd.Series:
        """Signal based on account status."""
        signal = pd.Series(0.0, index=self.accounts.index)
        frozen = self.accounts['account_status'] == 'frozen'
        signal[frozen] = 0.7
        return signal
    
    def activity_signal(self) -> pd.Series:
        """Signal based on activity patterns."""
        signal = pd.Series(0.0, index=self.accounts.index)
        self.accounts['account_opening_date'] = pd.to_datetime(self.accounts['account_opening_date'])
        account_age_days = (datetime.now() - self.accounts['account_opening_date']).dt.days
        new_accounts = account_age_days < 90
        signal[new_accounts] = 0.3
        return signal
    
    def txn_count_signal(self) -> pd.Series:
        """Signal based on transaction volume."""
        signal = pd.Series(0.0, index=self.accounts.index)
        if self.transactions is None:
            return signal
        txn_counts = self.transactions.groupby('account_id').size()
        for acct_id in self.accounts['account_id']:
            if acct_id in txn_counts.index:
                count = txn_counts[acct_id]
                if count > 1000:
                    signal[self.accounts['account_id'] == acct_id] = 0.6
        return signal
    
    def amount_signal(self) -> pd.Series:
        """Signal based on amount patterns."""
        signal = pd.Series(0.0, index=self.accounts.index)
        self.accounts['avg_balance'] = pd.to_numeric(self.accounts['avg_balance'], errors='coerce')
        very_high = self.accounts['avg_balance'] > 1000000
        signal[very_high] = 0.4
        return signal
    
    def kyc_signal(self) -> pd.Series:
        """Signal based on KYC compliance."""
        signal = pd.Series(0.0, index=self.accounts.index)
        non_kyc = self.accounts['kyc_compliant'] == 'N'
        signal[non_kyc] = 0.6
        return signal
    
    def generate_signals(self) -> pd.DataFrame:
        """Generate all signals."""
        logger.info("Generating signals...")
        signals_df = self.accounts[['account_id']].copy()
        signals_df['account_status_signal'] = self.account_status_signal()
        signals_df['activity_signal'] = self.activity_signal()
        signals_df['txn_count_signal'] = self.txn_count_signal()
        signals_df['amount_signal'] = self.amount_signal()
        signals_df['kyc_signal'] = self.kyc_signal()
        
        weights = {'account_status_signal': 0.25, 'activity_signal': 0.20, 'txn_count_signal': 0.20, 'amount_signal': 0.20, 'kyc_signal': 0.15}
        signals_df['composite_signal'] = (
            signals_df['account_status_signal'] * weights['account_status_signal'] +
            signals_df['activity_signal'] * weights['activity_signal'] +
            signals_df['txn_count_signal'] * weights['txn_count_signal'] +
            signals_df['amount_signal'] * weights['amount_signal'] +
            signals_df['kyc_signal'] * weights['kyc_signal']
        )
        return signals_df
    
    def run(self, output_path: str = "output/label_signals.csv"):
        """Run full pipeline."""
        logger.info("Starting label signal generation...")
        self.load_data()
        signals_df = self.generate_signals()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        signals_df.to_csv(output_file, index=False)
        logger.info(f"Signals saved to {output_file}")
        
        merged = signals_df.merge(self.train_labels[['account_id', 'is_mule']], on='account_id', how='left')
        mules = merged[merged['is_mule'] == 1]
        non_mules = merged[merged['is_mule'] == 0]
        
        logger.info(f"\nTotal accounts: {len(signals_df)}")
        logger.info(f"High-risk (>0.5): {len(signals_df[signals_df['composite_signal'] > 0.5])}")
        logger.info(f"Mule mean signal: {mules['composite_signal'].mean():.4f} (n={len(mules)})")
        logger.info(f"Non-mule mean signal: {non_mules['composite_signal'].mean():.4f} (n={len(non_mules)})")
        
        return signals_df


if __name__ == "__main__":
    try:
        gen = LabelSignalGenerator()
        signals_df = gen.run()
        print("\n✓ Label signal generation completed!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
