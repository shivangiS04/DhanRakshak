import pandas as pd
import numpy as np
import logging
from glob import glob
from collections import defaultdict
import warnings
import gc
import os

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "../archive"
OUTPUT_DIR = "output"

def add_velocity_features():
    """Add velocity features to existing transaction features"""
    logger.info("Adding velocity features...")
    
    # Load existing features
    features_df = pd.read_csv(f"{OUTPUT_DIR}/mega_transaction_features.csv")
    logger.info(f"Loaded {len(features_df)} accounts")
    
    # Initialize velocity feature
    velocity_features = defaultdict(lambda: [])
    
    # Stream transactions to calculate velocity
    parts = sorted(glob(f"{DATA_DIR}/transactions/batch-*/part_*.parquet"))
    logger.info(f"Found {len(parts)} transaction parts")
    
    for i, part in enumerate(parts):
        if i % 50 == 0:
            logger.info(f"Processing part {i+1}/{len(parts)}")
        
        df = pd.read_parquet(
            part,
            columns=["account_id", "transaction_timestamp"],
            engine="pyarrow",
        )
        
        df["timestamp"] = pd.to_datetime(df["transaction_timestamp"], format='mixed')
        
        for account_id, group in df.groupby("account_id"):
            if len(group) > 1:
                sorted_ts = group["timestamp"].sort_values().values
                time_diffs = np.diff(sorted_ts).astype('timedelta64[m]').astype(float)
                velocity_features[account_id].extend(time_diffs.tolist())
        
        del df
        gc.collect()
    
    # Calculate median velocity for each account
    logger.info("Calculating median velocity...")
    median_velocities = {}
    for account_id, diffs in velocity_features.items():
        if len(diffs) > 0:
            median_velocities[account_id] = float(np.median(diffs))
        else:
            median_velocities[account_id] = 0
    
    # Add to features
    features_df['median_time_between_txns'] = features_df['account_id'].map(
        lambda x: median_velocities.get(x, 0)
    )
    
    # Save enhanced features
    output_path = f"{OUTPUT_DIR}/mega_transaction_features_enhanced.csv"
    features_df.to_csv(output_path, index=False)
    logger.info(f"Saved enhanced features to {output_path}")
    logger.info(f"Features shape: {features_df.shape}")
    logger.info(f"Velocity stats - Mean: {features_df['median_time_between_txns'].mean():.2f}, "
                f"Median: {features_df['median_time_between_txns'].median():.2f}, "
                f"Max: {features_df['median_time_between_txns'].max():.2f}")

if __name__ == "__main__":
    add_velocity_features()
