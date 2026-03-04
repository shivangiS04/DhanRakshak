#!/usr/bin/env python3
"""
Generate Improved Submission

Uses optimized models to generate predictions for test accounts.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = '/Users/shivangisingh/Desktop/Archive'
OUTPUT_DIR = 'output'


def load_data():
    """Load all necessary data."""
    logger.info("Loading data...")
    
    # Load test accounts
    test_accounts = pd.read_parquet(f'{DATA_DIR}/test_accounts.parquet')
    logger.info(f"Loaded {len(test_accounts)} test accounts")
    
    # Load accounts
    accounts_df = pd.read_parquet(f'{DATA_DIR}/accounts.parquet')
    logger.info(f"Loaded {len(accounts_df)} accounts")
    
    # Load label signals
    signals_df = pd.read_csv(f'{OUTPUT_DIR}/label_signals.csv')
    logger.info(f"Loaded {len(signals_df)} label signals")
    
    # Merge
    merged = test_accounts.merge(accounts_df, on='account_id', how='left')
    merged = merged.merge(signals_df[['account_id', 'composite_signal']], on='account_id', how='left')
    merged['composite_signal'] = merged['composite_signal'].fillna(0.0)
    
    logger.info(f"Merged data: {len(merged)} rows")
    return merged


def extract_features(data_df):
    """Extract features from account data."""
    logger.info("Extracting features...")
    
    features = pd.DataFrame()
    features['account_id'] = data_df['account_id']
    
    # Account-level features
    features['avg_balance'] = data_df['avg_balance'].fillna(0)
    features['monthly_avg_balance'] = data_df['monthly_avg_balance'].fillna(0)
    features['daily_avg_balance'] = data_df['daily_avg_balance'].fillna(0)
    features['account_age_days'] = (pd.to_datetime('2025-06-30') - pd.to_datetime(data_df['account_opening_date'])).dt.days
    
    # Account status features
    features['is_frozen'] = (data_df['account_status'] == 'frozen').astype(int)
    features['has_freeze_history'] = data_df['freeze_date'].notna().astype(int)
    
    # KYC features
    features['kyc_compliant'] = (data_df['kyc_compliant'] == 'Y').astype(int)
    features['days_since_kyc'] = (pd.to_datetime('2025-06-30') - pd.to_datetime(data_df['last_kyc_date'])).dt.days
    
    # Mobile update features
    features['has_mobile_update'] = data_df['last_mobile_update_date'].notna().astype(int)
    features['days_since_mobile_update'] = (pd.to_datetime('2025-06-30') - pd.to_datetime(data_df['last_mobile_update_date'])).dt.days
    
    # Cheque features
    features['cheque_allowed'] = (data_df['cheque_allowed'] == 'Y').astype(int)
    features['cheque_availed'] = (data_df['cheque_availed'] == 'Y').astype(int)
    features['num_chequebooks'] = data_df['num_chequebooks'].fillna(0)
    
    # Nomination features
    features['nomination_flag'] = (data_df['nomination_flag'] == 'Y').astype(int)
    
    # Product features
    features['is_savings'] = (data_df['product_family'] == 'S').astype(int)
    features['is_kfamily'] = (data_df['product_family'] == 'K').astype(int)
    features['is_overdraft'] = (data_df['product_family'] == 'O').astype(int)
    
    # Rural branch
    features['rural_branch'] = (data_df['rural_branch'] == 'Y').astype(int)
    
    # Label signal (most important feature)
    features['composite_signal'] = data_df['composite_signal'].fillna(0)
    
    # Fill NaN values
    features = features.fillna(0)
    
    logger.info(f"Extracted {len(features.columns) - 1} features")
    return features


def train_models(X_train, y_train):
    """Train optimized models."""
    logger.info("Training models...")
    
    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        n_estimators=200
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    logger.info("XGBoost trained")
    
    # Random Forest
    class_weight = {0: 1, 1: scale_pos_weight}
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    logger.info("Random Forest trained")
    
    # Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lr_model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    lr_model.fit(X_train_scaled, y_train)
    logger.info("Logistic Regression trained")
    
    return xgb_model, rf_model, lr_model, scaler


def generate_predictions(models, X_test, scaler, threshold=0.8):
    """Generate ensemble predictions."""
    logger.info("Generating predictions...")
    
    xgb_model, rf_model, lr_model = models
    
    # Individual predictions
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    rf_pred = rf_model.predict_proba(X_test)[:, 1]
    lr_pred = lr_model.predict_proba(scaler.transform(X_test))[:, 1]
    
    # Weighted ensemble
    ensemble_pred = (xgb_pred * 0.4 + rf_pred * 0.4 + lr_pred * 0.2)
    
    logger.info(f"Predictions range: {ensemble_pred.min():.4f} - {ensemble_pred.max():.4f}")
    logger.info(f"Mean prediction: {ensemble_pred.mean():.4f}")
    
    return ensemble_pred


def generate_temporal_windows(predictions, threshold=0.8):
    """Generate temporal windows for suspicious activity."""
    logger.info("Generating temporal windows...")
    
    windows = []
    
    for pred in predictions:
        if pred >= threshold:
            # High-risk: last 3 months
            start = '2025-04-01T00:00:00'
            end = '2025-06-30T00:00:00'
        elif pred >= 0.5:
            # Medium-risk: last 6 months
            start = '2025-01-01T00:00:00'
            end = '2025-06-30T00:00:00'
        else:
            # Low-risk: no window
            start = ''
            end = ''
        
        windows.append((start, end))
    
    return windows


def main():
    """Main workflow."""
    logger.info("=" * 80)
    logger.info("IMPROVED SUBMISSION GENERATION")
    logger.info("=" * 80)
    
    # Load training data
    logger.info("\nLoading training data...")
    train_labels = pd.read_parquet(f'{DATA_DIR}/train_labels.parquet')
    train_accounts = pd.read_parquet(f'{DATA_DIR}/accounts.parquet')
    train_signals = pd.read_csv(f'{OUTPUT_DIR}/label_signals.csv')
    
    train_merged = train_labels.merge(train_accounts, on='account_id', how='left')
    train_merged = train_merged.merge(train_signals[['account_id', 'composite_signal']], on='account_id', how='left')
    train_merged['composite_signal'] = train_merged['composite_signal'].fillna(0.0)
    
    X_train_df = extract_features(train_merged)
    X_train = X_train_df.drop('account_id', axis=1)
    y_train = train_labels['is_mule'].values
    
    logger.info(f"Training data: {len(X_train)} samples")
    logger.info(f"Class distribution: {np.bincount(y_train)}")
    
    # Train models
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING MODELS")
    logger.info("=" * 80)
    
    models = train_models(X_train, y_train)
    xgb_model, rf_model, lr_model, scaler = models
    
    # Load test data
    logger.info("\n" + "=" * 80)
    logger.info("LOADING TEST DATA")
    logger.info("=" * 80)
    
    test_data = load_data()
    X_test_df = extract_features(test_data)
    X_test = X_test_df.drop('account_id', axis=1)
    account_ids = X_test_df['account_id'].values
    
    logger.info(f"Test data: {len(X_test)} samples")
    
    # Generate predictions
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING PREDICTIONS")
    logger.info("=" * 80)
    
    predictions = generate_predictions((xgb_model, rf_model, lr_model), X_test, scaler, threshold=0.8)
    
    # Generate temporal windows
    windows = generate_temporal_windows(predictions, threshold=0.8)
    
    # Create submission
    logger.info("\n" + "=" * 80)
    logger.info("CREATING SUBMISSION")
    logger.info("=" * 80)
    
    submission = pd.DataFrame({
        'account_id': account_ids,
        'is_mule': predictions,
        'suspicious_start': [w[0] for w in windows],
        'suspicious_end': [w[1] for w in windows]
    })
    
    # Save
    output_path = Path(OUTPUT_DIR) / 'submission_improved.csv'
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    
    # Statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUBMISSION STATISTICS")
    logger.info("=" * 80)
    
    logger.info(f"Total accounts: {len(submission)}")
    logger.info(f"High-risk (>0.8): {(predictions >= 0.8).sum()}")
    logger.info(f"Medium-risk (0.5-0.8): {((predictions >= 0.5) & (predictions < 0.8)).sum()}")
    logger.info(f"Low-risk (<0.5): {(predictions < 0.5).sum()}")
    logger.info(f"Mean prediction: {predictions.mean():.4f}")
    logger.info(f"Std prediction: {predictions.std():.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
