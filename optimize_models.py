#!/usr/bin/env python3
"""
Model Optimization Script

Improves model performance through:
1. Better hyperparameter tuning
2. Feature selection and importance analysis
3. Threshold optimization
4. Ensemble weighting optimization
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
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


def load_training_data():
    """Load training data with features and labels."""
    logger.info("Loading training data...")
    
    # Load labels
    labels_df = pd.read_parquet(f'{DATA_DIR}/train_labels.parquet')
    logger.info(f"Loaded {len(labels_df)} training labels")
    
    # Load accounts
    accounts_df = pd.read_parquet(f'{DATA_DIR}/accounts.parquet')
    logger.info(f"Loaded {len(accounts_df)} accounts")
    
    # Load label signals
    signals_df = pd.read_csv(f'{OUTPUT_DIR}/label_signals.csv')
    logger.info(f"Loaded {len(signals_df)} label signals")
    
    # Merge
    merged = labels_df.merge(accounts_df, on='account_id', how='left')
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


def optimize_xgboost(X_train, y_train, X_val, y_val):
    """Optimize XGBoost hyperparameters."""
    logger.info("Optimizing XGBoost...")
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Best parameters found through tuning
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
    }
    
    # Train
    model = xgb.XGBClassifier(**params, n_estimators=200)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    logger.info(f"XGBoost AUC: {auc:.4f}")
    
    return model, auc


def optimize_random_forest(X_train, y_train, X_val, y_val):
    """Optimize Random Forest hyperparameters."""
    logger.info("Optimizing Random Forest...")
    
    # Calculate class weights
    class_weight = {0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()}
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    logger.info(f"Random Forest AUC: {auc:.4f}")
    
    return model, auc


def optimize_logistic_regression(X_train, y_train, X_val, y_val):
    """Optimize Logistic Regression."""
    logger.info("Optimizing Logistic Regression...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Calculate class weights
    class_weight = {0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()}
    
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    logger.info(f"Logistic Regression AUC: {auc:.4f}")
    
    return model, auc, scaler


def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal prediction threshold."""
    logger.info("Finding optimal threshold...")
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logger.info(f"Optimal threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
    return best_threshold


def main():
    """Main optimization workflow."""
    logger.info("=" * 80)
    logger.info("MODEL OPTIMIZATION")
    logger.info("=" * 80)
    
    # Load data
    data_df = load_training_data()
    
    # Extract features
    features_df = extract_features(data_df)
    
    # Prepare X and y
    X = features_df.drop('account_id', axis=1)
    y = data_df['is_mule'].values
    
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Split into train/val
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Optimize models
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZING MODELS")
    logger.info("=" * 80)
    
    xgb_model, xgb_auc = optimize_xgboost(X_train, y_train, X_val, y_val)
    rf_model, rf_auc = optimize_random_forest(X_train, y_train, X_val, y_val)
    lr_model, lr_auc, lr_scaler = optimize_logistic_regression(X_train, y_train, X_val, y_val)
    
    # Ensemble predictions
    logger.info("\n" + "=" * 80)
    logger.info("ENSEMBLE PREDICTIONS")
    logger.info("=" * 80)
    
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    rf_pred = rf_model.predict_proba(X_val)[:, 1]
    lr_pred = lr_model.predict_proba(lr_scaler.transform(X_val))[:, 1]
    
    # Weighted ensemble
    ensemble_pred = (xgb_pred * 0.4 + rf_pred * 0.4 + lr_pred * 0.2)
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    logger.info(f"Ensemble AUC: {ensemble_auc:.4f}")
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_val, ensemble_pred)
    
    # Final metrics
    y_pred = (ensemble_pred >= optimal_threshold).astype(int)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    
    logger.info(f"\nFinal Metrics:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  AUC-ROC: {ensemble_auc:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
