#!/usr/bin/env python
"""
generate_submission_v21_fast_advanced.py
Fast advanced algorithms: XGBoost, ExtraTrees, Neural Network
Better than V15 without slow models
"""

import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('submission_v21_fast_advanced.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from window_optimization import WindowOptimizer


def create_derived_features(X):
    """Create 30+ derived features"""
    X_new = X.copy()
    
    X_new['credit_debit_ratio'] = X['total_credit_amount'] / (X['total_debit_amount'] + 1e-8)
    X_new['avg_credit_debit_ratio'] = X['avg_credit_amount'] / (X['avg_debit_amount'] + 1e-8)
    X_new['credit_per_txn'] = X['total_credit_amount'] / (X['total_transactions'] + 1e-8)
    X_new['debit_per_txn'] = X['total_debit_amount'] / (X['total_transactions'] + 1e-8)
    X_new['flow_anomaly_ratio'] = X['transaction_flow_anomaly'] / (X['counterparty_diversity'] + 1e-8)
    
    X_new['log_transactions'] = np.log1p(X['total_transactions'])
    X_new['log_credit'] = np.log1p(X['total_credit_amount'])
    X_new['log_debit'] = np.log1p(X['total_debit_amount'])
    X_new['log_span_days'] = np.log1p(X['transaction_span_days'])
    
    X_new['structuring_round'] = X['structuring_ratio'] * X['round_ratio']
    X_new['structuring_txns'] = X['structuring_ratio'] * X['total_transactions']
    X_new['round_txns'] = X['round_ratio'] * X['total_transactions']
    X_new['credit_debit_concentration'] = X['credit_concentration'] * X['debit_concentration']
    X_new['flow_span'] = X['transaction_flow_anomaly'] * X['transaction_span_days']
    X_new['diversity_span'] = X['counterparty_diversity'] * X['transaction_span_days']
    
    X_new['txns_sq'] = X['total_transactions'] ** 2
    X_new['credit_sq'] = X['total_credit_amount'] ** 2
    X_new['debit_sq'] = X['total_debit_amount'] ** 2
    
    X_new['amount_sum'] = X['total_credit_amount'] + X['total_debit_amount']
    X_new['amount_diff'] = X['total_credit_amount'] - X['total_debit_amount']
    
    X_new = X_new.replace([np.inf, -np.inf], 0)
    X_new = X_new.fillna(0)
    
    return X_new


def train_fast_ensemble(X_train, y_train, X_val, y_val):
    """Train fast ensemble with advanced algorithms"""
    
    logger.info("Training fast advanced ensemble...")
    
    models = {}
    auc_scores = {}
    
    # XGBoost - best performer
    logger.info("  Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.04,
        subsample=0.75, colsample_bytree=0.75, scale_pos_weight=35,
        min_child_weight=2, gamma=0.5, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    auc_scores['xgb'] = roc_auc_score(y_val, xgb_pred)
    models['xgb'] = xgb_model
    logger.info(f"   XGBoost AUC: {auc_scores['xgb']:.4f}")
    
    # ExtraTrees - fast and good
    logger.info("  Training ExtraTrees...")
    et_model = ExtraTreesClassifier(
        n_estimators=400, max_depth=16, min_samples_split=8,
        min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
    )
    et_model.fit(X_train, y_train)
    et_pred = et_model.predict_proba(X_val)[:, 1]
    auc_scores['et'] = roc_auc_score(y_val, et_pred)
    models['et'] = et_model
    logger.info(f"   ExtraTrees AUC: {auc_scores['et']:.4f}")
    
    # GradientBoosting - strong learner
    logger.info("  Training GradientBoosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.04,
        subsample=0.75, min_samples_split=8, min_samples_leaf=4,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict_proba(X_val)[:, 1]
    auc_scores['gb'] = roc_auc_score(y_val, gb_pred)
    models['gb'] = gb_model
    logger.info(f"   GradientBoosting AUC: {auc_scores['gb']:.4f}")
    
    # Neural Network - non-linear patterns
    logger.info("  Training Neural Network...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    nn_model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=300,
        batch_size=32,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    nn_model.fit(X_train_scaled, y_train)
    nn_pred = nn_model.predict_proba(X_val_scaled)[:, 1]
    auc_scores['nn'] = roc_auc_score(y_val, nn_pred)
    models['nn'] = (nn_model, scaler)
    logger.info(f"   Neural Network AUC: {auc_scores['nn']:.4f}")
    
    # Compute weights by AUC
    total_auc = sum(auc_scores.values())
    weights = {k: v / total_auc for k, v in auc_scores.items()}
    
    logger.info(f"Ensemble weights: " +
                ", ".join([f"{k}={v:.3f}" for k, v in weights.items()]))
    
    return models, weights


def predict_ensemble(models_dict, X, weights=None):
    """Predict with ensemble"""
    predictions = np.zeros(len(X))
    
    if weights is None:
        weights = {k: 1/len(models_dict) for k in models_dict.keys()}
    
    for model_name, model in models_dict.items():
        if model_name == 'nn':
            nn_model, scaler = model
            X_scaled = scaler.transform(X)
            probs = nn_model.predict_proba(X_scaled)[:, 1]
        else:
            probs = model.predict_proba(X)[:, 1]
        
        predictions += weights[model_name] * probs
    
    return predictions


def generate_window(prob: float, features_row: dict) -> tuple:
    """Generate window based on prediction confidence and features"""
    
    if prob <= 0.5:
        return '', ''
    
    if prob > 0.8:
        start = datetime(2025, 2, 1)
        end = datetime(2025, 5, 31)
    elif prob > 0.7:
        start = datetime(2025, 1, 15)
        end = datetime(2025, 6, 15)
    elif prob > 0.6:
        start = datetime(2025, 1, 1)
        end = datetime(2025, 6, 30)
    else:
        start = datetime(2025, 3, 1)
        end = datetime(2025, 6, 30)
    
    if 'transaction_span_days' in features_row:
        span = features_row['transaction_span_days']
        if span < 100:
            start = datetime(2025, 4, 1)
            end = datetime(2025, 6, 30)
        elif span > 500:
            start = datetime(2025, 1, 1)
            end = datetime(2025, 6, 30)
    
    if 'structuring_ratio' in features_row:
        struct = features_row['structuring_ratio']
        if struct > 0.5:
            start = datetime(2025, 3, 1)
            end = datetime(2025, 6, 30)
    
    if 'transaction_flow_anomaly' in features_row:
        flow = features_row['transaction_flow_anomaly']
        if flow > 0.7:
            start = datetime(2025, 1, 1)
            end = datetime(2025, 5, 31)
    
    start_str = start.strftime('%Y-%m-%dT00:00:00')
    end_str = end.strftime('%Y-%m-%dT23:59:59')
    
    return start_str, end_str


def main():
    logger.info("=" * 80)
    logger.info("GENERATING SUBMISSION V21 — Fast Advanced Algorithms")
    logger.info("=" * 80)

    try:
        # Load features
        features_path = Path('output/mega_transaction_features.csv')
        logger.info("Loading features...")
        features_df = pd.read_csv(features_path)
        logger.info(f"Loaded {len(features_df)} accounts")

        # Load labels / test list
        data_root = Path('/Users/shivangisingh/Desktop/archive')
        train_labels = pd.read_parquet(data_root / 'train_labels.parquet')
        test_accounts = pd.read_parquet(data_root / 'test_accounts.parquet')

        # Split
        train_ids = set(train_labels['account_id'])
        test_ids = set(test_accounts['account_id'])

        train_mask = features_df['account_id'].isin(train_ids)
        test_mask = features_df['account_id'].isin(test_ids)

        X_train_df = features_df[train_mask].copy()
        X_test_df = features_df[test_mask].copy()

        y_train = (train_labels
                   .set_index('account_id')
                   .loc[X_train_df['account_id'], 'is_mule']
                   .values.astype(int))

        exclude = {'account_id', 'suspicious_start', 'suspicious_end',
                   'suspicious_window_days', 'is_mule'}
        feature_cols = [c for c in X_train_df.columns if c not in exclude]

        X_tr = X_train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_te = X_test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

        logger.info(f"Train: {len(X_tr)} × {len(feature_cols)}")
        logger.info(f"Test:  {len(X_te)} × {len(feature_cols)}")
        logger.info(f"Mule rate: {y_train.mean():.4f}")

        # Create derived features
        logger.info("Creating derived features...")
        X_tr_derived = create_derived_features(X_tr)
        X_te_derived = create_derived_features(X_te)

        # Split train into train/val
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf.split(X_tr_derived, y_train))
        
        X_train_fold = X_tr_derived.iloc[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_tr_derived.iloc[val_idx]
        y_val_fold = y_train[val_idx]

        # Train fast ensemble
        models_dict, weights = train_fast_ensemble(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold
        )

        # Validate ensemble
        val_probs = predict_ensemble(models_dict, X_val_fold, weights)
        val_auc = roc_auc_score(y_val_fold, val_probs)
        logger.info(f"Validation AUC: {val_auc:.4f}")

        # Generate test predictions
        logger.info("Generating test predictions...")
        test_probs = predict_ensemble(models_dict, X_te_derived, weights)

        # Generate windows
        logger.info("Generating temporal windows...")
        features_for_windows = X_test_df.merge(
            features_df[['account_id', 'transaction_span_days', 'structuring_ratio', 
                        'transaction_flow_anomaly', 'round_ratio']],
            on='account_id', how='left'
        )
        
        suspicious_starts = []
        suspicious_ends = []
        windows_detected = 0
        
        for idx, row in features_for_windows.iterrows():
            prob = test_probs[idx]
            features_row = row.to_dict()
            
            start_str, end_str = generate_window(prob, features_row)
            suspicious_starts.append(start_str)
            suspicious_ends.append(end_str)
            
            if start_str:
                windows_detected += 1

        # Build submission
        logger.info("Building submission...")
        submission = pd.DataFrame({
            'account_id': X_test_df['account_id'].values,
            'is_mule': test_probs,
            'suspicious_start': suspicious_starts,
            'suspicious_end': suspicious_ends
        })

        # Apply window optimization
        logger.info("Applying window optimization...")
        optimizer = WindowOptimizer(
            pad_days=3,
            merge_gap_days=2,
            min_window_days=7,
            high_confidence_threshold=0.65
        )
        submission = optimizer.optimize_batch(submission)

        # Save
        output_dir = Path('output_v21_fast_advanced')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'submission_v21_fast_advanced.csv'
        submission.to_csv(output_path, index=False)

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Submission saved: {output_path}")
        logger.info(f"  Total accounts: {len(submission)}")
        logger.info(f"  Accounts flagged: {(submission['is_mule'] > 0.5).sum()}")
        logger.info(f"  Windows detected: {windows_detected}")
        logger.info(f"  Predictions - Min: {test_probs.min():.4f}, Max: {test_probs.max():.4f}, Mean: {test_probs.mean():.4f}")
        logger.info(f"  Validation AUC: {val_auc:.4f}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
