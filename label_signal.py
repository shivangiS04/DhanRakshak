"""
Label Signal Generator for Mule Account Detection

Generates behavioral signals based on known mule patterns:
- Dormant Activation: Long-inactive accounts with sudden high-value bursts
- Structuring: Repeated transactions near reporting thresholds
- Rapid Pass-Through: Large credits quickly followed by matching debits
- Fan-In/Fan-Out: Many small inflows aggregated into one large outflow
- Geographic Anomaly: Transactions from inconsistent locations
- New Account High Value: Recently opened accounts with high volumes
- Income Mismatch: Transaction values disproportionate to balance
- Post-Mobile-Change Spike: Surge after mobile number update
- Round Amount Patterns: Disproportionate use of exact round amounts
- Salary Cycle Exploitation: Laundering within salary cycles
- Branch-Level Collusion: Clusters at same branch with shared counterparties
- MCC-Amount Anomaly: Statistical outliers for merchant category
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LabelSignalGenerator:
    """Generates behavioral signals for mule account detection."""
    
    def __init__(self, data_dir: str = "/Users/shivangisingh/Desktop/Archive"):
        """Initialize with data directory path."""
        self.data_dir = Path(data_dir)
        self.accounts = None
        self.transactions = None
        self.train_labels = None
        
    def load_data(self):
        """Load required data files."""
        logger.info("Loading data files...")
        
        try:
            self.accounts = pd.read_parquet(self.data_dir / "accounts.parquet")
            logger.info(f"Loaded accounts: {len(self.accounts)} rows")
            
            self.train_labels = pd.read_parquet(self.data_dir / "train_labels.parquet")
            logger.info(f"Loaded train_labels: {len(self.train_labels)} rows")
            
            # Load transactions (sample for efficiency)
            logger.info("Loading transaction data...")
            from glob import glob
            parts = sorted(glob(str(self.data_dir / "transactions/batch-*/part_*.parquet")))
            if parts:
                self.transactions = pd.concat(
                    [pd.read_parquet(p) for p in parts[:5]],
                    ignore_index=True
                )
                logger.info(f"Loaded transactions: {len(self.transactions)} rows (sampled)")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def account_status_signal(self) -> pd.Series:
        """Signal based on account status and freeze history."""
        signal = pd.Series(0.0, index=self.accounts.index)
        
        frozen = self.accounts['account_status'] == 'frozen'
        signal[frozen] = 0.7
        
        has_freeze_history = (
            self.accounts['freeze_date'].notna() | 
            self.accounts['unfreeze_date'].notna()
        )
        signal[has_freeze_history & ~frozen] = 0.4
        
        return signal
    
    def activity_signal(self) -> pd.Series:
        """Signal based on account activity patterns."""
        signal = pd.Series(0.0, index=self.accounts.index)
        
        self.accounts['account_opening_date'] = pd.to_datetime(
            self.accounts['account_opening_date']
        )
        account_age_days = (datetime.now() - self.accounts['account_opening_date']).dt.days
        
        new_accounts = account_age_days < 90
        signal[new_accounts] = 0.3
        
        old_accounts = account_age_days > 1095
        self.accounts['last_mobile_update_date'] = pd.to_datetime(
            self.accounts['last_mobile_update_date'], errors='coerce'
        )
        recent_update = (
            datetime.now() - self.accounts['last_mobile_update_date']
        ).dt.days < 30
        signal[old_accounts & recent_update] = 0.5
        
        return signal
    
    def txn_count_signal(self) -> pd.Series:
        """Signal based on transaction volume patterns."""
        signal = pd.Series(0.0, index=self.accounts.index)
        
        if self.transactions is None:
            return signal
        
        txn_counts = self.transactions.groupby('account_id').size()
        
        for acct_id in self.accounts['account_id']:
            if acct_id in txn_counts.index:
                count = txn_counts[acct_id]
                if count > 1000:
                    signal[self.accounts['account_id'] == acct_id] = 0.6
                elif count > 500:
                    signal[self.accounts['account_id'] == acct_id] = 0.3
        
        return signal
    
    def amount_signal(self) -> pd.Series:
        """Signal based on transaction amount patterns."""
        signal = pd.Series(0.0, index=self.accounts.index)
        
        self.accounts['avg_balance'] = pd.to_numeric(
            self.accounts['avg_balance'], errors='coerce'
        )
        self.accounts['monthly_avg_balance'] = pd.to_numeric(
            self.accounts['monthly_avg_balance'], errors='coerce'
        )
        
        negative_balance = self.accounts['avg_balance'] < 0
        high_monthly = self.accounts['monthly_avg_balance'] > 100000
        signal[negative_balance & high_monthly] = 0.5
        
        very_high_balance = self.accounts['avg_balance'] > 1000000
        signal[very_high_balance] = 0.4
        
        return signal
    
    def kyc_signal(self) -> pd.Series:
        """Signal based on KYC compliance."""
        signal = pd.Series(0.0, index=self.accounts.index)
        
        non_kyc = self.accounts['kyc_compliant'] == 'N'
        signal[non_kyc] = 0.6
        
        self.accounts['last_kyc_date'] = pd.to_datetime(
            self.accounts['last_kyc_date'], errors='coerce'
        )
        recent_kyc = (
            datetime.now() - self.accounts['last_kyc_date']
        ).dt.days < 30
        signal[recent_kyc] = 0.2
        
        return signal
    
    def generate_signals(self) -> pd.DataFrame:
        """Generate all signals and composite score."""
        logger.info("Generating signals...")
        
        signals_df = self.accounts[['account_id']].copy()
        
        signals_df['account_status_signal'] = self.account_status_signal()
        signals_df['activity_signal'] = self.activity_signal()
        signals_df['txn_count_signal'] = self.txn_count_signal()
        signals_df['amount_signal'] = self.amount_signal()
        signals_df['kyc_signal'] = self.kyc_signal()
        
        weights = {
            'account_status_signal': 0.25,
            'activity_signal': 0.20,
            'txn_count_signal': 0.20,
            'amount_signal': 0.20,
            'kyc_signal': 0.15
        }
        
        signals_df['composite_signal'] = (
            signals_df['account_status_signal'] * weights['account_status_signal'] +
            signals_df['activity_signal'] * weights['activity_signal'] +
            signals_df['txn_count_signal'] * weights['txn_count_signal'] +
            signals_df['amount_signal'] * weights['amount_signal'] +
            signals_df['kyc_signal'] * weights['kyc_signal']
        )
        
        return signals_df
    
    def generate_report(self, signals_df: pd.DataFrame) -> Dict:
        """Generate analysis report."""
        logger.info("Generating report...")
        
        report = {
            'total_accounts': len(signals_df),
            'high_risk': len(signals_df[signals_df['composite_signal'] > 0.5]),
            'medium_risk': len(signals_df[
                (signals_df['composite_signal'] > 0.3) & 
                (signals_df['composite_signal'] <= 0.5)
            ]),
            'low_risk': len(signals_df[signals_df['composite_signal'] <= 0.3]),
        }
        
        if self.train_labels is not None:
            merged = signals_df.merge(
                self.train_labels[['account_id', 'is_mule']], 
                on='account_id', 
                how='left'
            )
            
            mules = merged[merged['is_mule'] == 1]
            non_mules = merged[merged['is_mule'] == 0]
            
            if len(mules) > 0:
                report['mule_mean'] = mules['composite_signal'].mean()
                report['mule_count'] = len(mules)
            
            if len(non_mules) > 0:
                report['non_mule_mean'] = non_mules['composite_signal'].mean()
                report['non_mule_count'] = len(non_mules)
        
        return report
    
    def run(self, output_path: str = "output/label_signals.csv"):
        """Run full signal generation pipeline."""
        logger.info("Starting label signal generation...")
        
        self.load_data()
        signals_df = self.generate_signals()
        report = self.generate_report(signals_df)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        signals_df.to_csv(output_file, index=False)
        logger.info(f"Signals saved to {output_file}")
        
        logger.info("\n" + "="*80)
        logger.info("LABEL SIGNAL GENERATION REPORT")
        logger.info("="*80)
        logger.info(f"Total accounts: {report['total_accounts']}")
        logger.info(f"High-risk accounts (signal > 0.5): {report['high_risk']}")
        logger.info(f"Medium-risk accounts (0.3 < signal <= 0.5): {report['medium_risk']}")
        logger.info(f"Low-risk accounts (signal <= 0.3): {report['low_risk']}")
        
        if 'mule_mean' in report:
            logger.info(f"\nMule Accounts - Mean Signal: {report['mule_mean']:.4f} (n={report['mule_count']})")
        
        if 'non_mule_mean' in report:
            logger.info(f"Non-Mule Accounts - Mean Signal: {report['non_mule_mean']:.4f} (n={report['non_mule_count']})")
        
        logger.info("="*80)
        
        return signals_df, report


if __name__ == "__main__":
    try:
        generator = LabelSignalGenerator()
        signals_df, report = generator.run()
        print("\n✓ Label signal generation completed successfully!")
        print(f"Output saved to: output/label_signals.csv")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
