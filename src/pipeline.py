"""
Main Pipeline Orchestrator for Mule Account Detection

Coordinates all phases:
1. Data Ingestion & Preprocessing
2. Feature Engineering
3. Behavioral Pattern Detection
4. Graph Analysis
5. Temporal Analysis
6. Model Training
7. Ensemble Prediction
8. Red-Herring Avoidance
9. Output Generation
10. Evaluation & Reporting
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from src.data_loader import DataLoader
from src.feature_engineering import FeatureExtractor, FeatureScaler, FeatureCache, AccountFeatures
from src.pattern_detection import PatternDetector, CompositeAnomalyScorer
from src.graph_analysis import TransactionNetworkGraph, GraphFeatureExtractor, GraphAnomalyDetector
from src.temporal_analysis import TemporalAnalyzer, TemporalIoUCalculator
from src.ensemble_models import EnsembleModelStack
from src.label_signal_integration import LabelSignalIntegrator

logger = logging.getLogger(__name__)


class MuleDetectionPipeline:
    """Main pipeline for mule account detection"""
    
    def __init__(self, data_dir: str, output_dir: str = 'output'):
        """
        Initialize pipeline.
        
        Args:
            data_dir: Path to data directory
            output_dir: Path to output directory
        """
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.data_loader = DataLoader(data_dir, str(self.output_dir / 'processed_data'))
        self.feature_extractor = None
        self.feature_scaler = None
        self.feature_cache = FeatureCache(str(self.output_dir / 'feature_cache'))
        self.pattern_detector = PatternDetector()
        self.anomaly_scorer = CompositeAnomalyScorer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.ensemble = EnsembleModelStack(str(self.output_dir / 'models'))
        self.label_signal_integrator = LabelSignalIntegrator(str(self.output_dir / 'label_signals.csv'))
        
        # Data containers
        self.train_features = {}
        self.test_features = {}
        self.train_labels = None
        self.test_accounts = None
        self.predictions = None
        
        logger.info(f"MuleDetectionPipeline initialized with data_dir: {data_dir}")
    
    def run_phase_1_data_ingestion(self) -> bool:
        """Phase 1: Data Ingestion & Preprocessing"""
        logger.info("=" * 80)
        logger.info("PHASE 1: DATA INGESTION & PREPROCESSING")
        logger.info("=" * 80)
        
        try:
            # Load all data
            if not self.data_loader.load_all_data():
                logger.error("Failed to load data")
                return False
            
            # Validate temporal consistency
            if not self.data_loader.validate_temporal_consistency():
                logger.warning("Temporal consistency validation failed")
            
            # Generate quality report
            report_path = self.data_loader.generate_quality_report(
                str(self.output_dir / 'data_quality_report.txt')
            )
            logger.info(f"Quality report: {report_path}")
            
            # Store labels and test accounts
            self.train_labels = self.data_loader.train_labels
            self.test_accounts = self.data_loader.test_accounts
            
            logger.info("Phase 1 complete")
            return True
            
        except Exception as e:
            logger.error(f"Phase 1 failed: {e}")
            return False
    
    def run_phase_1b_load_label_signals(self) -> bool:
        """Phase 1b: Load Label Signals"""
        logger.info("=" * 80)
        logger.info("PHASE 1B: LOAD LABEL SIGNALS")
        logger.info("=" * 80)
        
        try:
            if self.label_signal_integrator.load_signals():
                stats = self.label_signal_integrator.get_statistics()
                logger.info(f"Label signals loaded successfully")
                logger.info(f"  Total accounts: {stats.get('total_accounts', 0)}")
                logger.info(f"  Accounts with signals: {stats.get('accounts_with_signals', 0)}")
                logger.info(f"  Mean composite signal: {stats.get('mean_composite', 0):.6f}")
                logger.info(f"  High-risk accounts: {stats.get('high_risk_count', 0)}")
                logger.info(f"  Medium-risk accounts: {stats.get('medium_risk_count', 0)}")
                logger.info("Phase 1b complete")
                return True
            else:
                logger.warning("Label signals not available, continuing without them")
                return True
            
        except Exception as e:
            logger.error(f"Phase 1b failed: {e}")
            return False
            return False
    
    def run_phase_2_feature_engineering(self) -> bool:
        """Phase 2: Feature Engineering"""
        logger.info("=" * 80)
        logger.info("PHASE 2: FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        try:
            # Initialize feature extractor
            self.feature_extractor = FeatureExtractor(
                self.data_loader.accounts,
                self.data_loader.customers
            )
            
            # Extract features for training accounts
            logger.info("Extracting features for training accounts...")
            train_account_ids = self.train_labels['account_id'].tolist()
            
            for i, account_id in enumerate(train_account_ids):
                if i % 1000 == 0:
                    logger.info(f"Processing account {i}/{len(train_account_ids)}")
                
                # Check cache first
                cached_features = self.feature_cache.get(account_id)
                if cached_features:
                    self.train_features[account_id] = cached_features
                    continue
                
                # Get transactions for account
                transactions = self.data_loader.get_transactions_for_account(account_id)
                if transactions is None or transactions.empty:
                    transactions = pd.DataFrame()
                
                # Extract features
                features = self.feature_extractor.extract_features_for_account(account_id, transactions)
                
                # Add label
                label = self.train_labels[self.train_labels['account_id'] == account_id]['is_mule'].values
                if len(label) > 0:
                    features.is_mule = int(label[0])
                
                # Cache features
                self.feature_cache.put(features)
                self.train_features[account_id] = features
            
            logger.info(f"Extracted features for {len(self.train_features)} training accounts")
            
            # Extract features for test accounts
            logger.info("Extracting features for test accounts...")
            test_account_ids = self.test_accounts['account_id'].tolist()
            
            for i, account_id in enumerate(test_account_ids):
                if i % 1000 == 0:
                    logger.info(f"Processing account {i}/{len(test_account_ids)}")
                
                # Check cache first
                cached_features = self.feature_cache.get(account_id)
                if cached_features:
                    self.test_features[account_id] = cached_features
                    continue
                
                # Get transactions for account
                transactions = self.data_loader.get_transactions_for_account(account_id)
                if transactions is None or transactions.empty:
                    transactions = pd.DataFrame()
                
                # Extract features
                features = self.feature_extractor.extract_features_for_account(account_id, transactions)
                
                # Cache features
                self.feature_cache.put(features)
                self.test_features[account_id] = features
            
            logger.info(f"Extracted features for {len(self.test_features)} test accounts")
            
            # Initialize scaler
            self.feature_scaler = FeatureScaler()
            self.feature_scaler.fit(list(self.train_features.values()))
            
            logger.info("Phase 2 complete")
            return True
            
        except Exception as e:
            logger.error(f"Phase 2 failed: {e}")
            return False
    
    def run_phase_3_pattern_detection(self) -> bool:
        """Phase 3: Behavioral Pattern Detection"""
        logger.info("=" * 80)
        logger.info("PHASE 3: BEHAVIORAL PATTERN DETECTION")
        logger.info("=" * 80)
        
        try:
            # Detect patterns for training accounts
            logger.info("Detecting patterns for training accounts...")
            
            for account_id in self.train_features:
                features = self.train_features[account_id]
                transactions = self.data_loader.get_transactions_for_account(account_id)
                
                if transactions is None or transactions.empty:
                    continue
                
                # Detect patterns
                patterns = self.pattern_detector.detect_all_patterns(
                    account_id, transactions, features
                )
                
                # Compute composite score
                composite_score, confidence = self.anomaly_scorer.compute_composite_score(patterns)
                
                # Store in features
                features.pattern_anomaly_score = composite_score
                features.pattern_confidence = confidence
            
            logger.info("Phase 3 complete")
            return True
            
        except Exception as e:
            logger.error(f"Phase 3 failed: {e}")
            return False
    
    def run_phase_4_graph_analysis(self) -> bool:
        """Phase 4: Graph Analysis"""
        logger.info("=" * 80)
        logger.info("PHASE 4: GRAPH ANALYSIS")
        logger.info("=" * 80)
        
        try:
            logger.info("Building transaction network graph...")
            
            # Collect all transactions
            all_transactions = []
            for chunk in self.data_loader.iterate_transaction_chunks():
                all_transactions.append(chunk)
            
            if all_transactions:
                all_txns_df = pd.concat(all_transactions, ignore_index=True)
                
                # Build graph
                network = TransactionNetworkGraph()
                network.build_graph(all_txns_df)
                
                # Calculate centrality metrics
                degree_centrality = network.calculate_degree_centrality()
                betweenness_centrality = network.calculate_betweenness_centrality()
                clustering_coeff = network.calculate_clustering_coefficient()
                
                # Store in features
                for account_id in self.train_features:
                    features = self.train_features[account_id]
                    features.degree_centrality = degree_centrality.get(account_id, 0.0)
                    features.betweenness_centrality = betweenness_centrality.get(account_id, 0.0)
                    features.clustering_coefficient = clustering_coeff.get(account_id, 0.0)
                
                for account_id in self.test_features:
                    features = self.test_features[account_id]
                    features.degree_centrality = degree_centrality.get(account_id, 0.0)
                    features.betweenness_centrality = betweenness_centrality.get(account_id, 0.0)
                    features.clustering_coefficient = clustering_coeff.get(account_id, 0.0)
            
            logger.info("Phase 4 complete")
            return True
            
        except Exception as e:
            logger.error(f"Phase 4 failed: {e}")
            return False
    
    def run_phase_5_temporal_analysis(self) -> bool:
        """Phase 5: Temporal Analysis"""
        logger.info("=" * 80)
        logger.info("PHASE 5: TEMPORAL ANALYSIS")
        logger.info("=" * 80)
        
        try:
            logger.info("Detecting suspicious windows...")
            
            # Detect windows for training accounts
            for account_id in self.train_features:
                features = self.train_features[account_id]
                transactions = self.data_loader.get_transactions_for_account(account_id)
                
                if transactions is None or transactions.empty:
                    continue
                
                # Detect suspicious window
                window = self.temporal_analyzer.detect_suspicious_window(
                    account_id, transactions, features
                )
                
                if window:
                    features.suspicious_start = window.suspicious_start
                    features.suspicious_end = window.suspicious_end
                    features.temporal_anomaly_score = window.anomaly_score
            
            # Detect windows for test accounts
            for account_id in self.test_features:
                features = self.test_features[account_id]
                transactions = self.data_loader.get_transactions_for_account(account_id)
                
                if transactions is None or transactions.empty:
                    continue
                
                # Detect suspicious window
                window = self.temporal_analyzer.detect_suspicious_window(
                    account_id, transactions, features
                )
                
                if window:
                    features.suspicious_start = window.suspicious_start
                    features.suspicious_end = window.suspicious_end
                    features.temporal_anomaly_score = window.anomaly_score
            
            logger.info("Phase 5 complete")
            return True
            
        except Exception as e:
            logger.error(f"Phase 5 failed: {e}")
            return False
    
    def run_phase_6_model_training(self) -> bool:
        """Phase 6: Model Training"""
        logger.info("=" * 80)
        logger.info("PHASE 6: MODEL TRAINING")
        logger.info("=" * 80)
        
        try:
            # Prepare training data
            logger.info("Preparing training data...")
            
            X_train_list = []
            y_train_list = []
            
            for account_id, features in self.train_features.items():
                # Convert features to dict
                feature_dict = self._features_to_dict(features)
                X_train_list.append(feature_dict)
                y_train_list.append(features.is_mule if features.is_mule is not None else 0)
            
            X_train = pd.DataFrame(X_train_list)
            y_train = np.array(y_train_list)
            
            logger.info(f"Training data: {len(X_train)} samples, {len(X_train.columns)} features")
            logger.info(f"Class distribution: {np.bincount(y_train.astype(int))}")
            
            # Train ensemble with SMOTE
            logger.info("Training ensemble models with SMOTE balancing...")
            metrics = self.ensemble.train_all_models(X_train, y_train, use_smote=False)
            
            # Save models
            self.ensemble.save_models()
            
            logger.info("Phase 6 complete")
            return True
            
        except Exception as e:
            logger.error(f"Phase 6 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_phase_7_ensemble_prediction(self) -> bool:
        """Phase 7: Ensemble Prediction"""
        logger.info("=" * 80)
        logger.info("PHASE 7: ENSEMBLE PREDICTION")
        logger.info("=" * 80)
        
        try:
            # Prepare test data
            logger.info("Preparing test data...")
            
            X_test_list = []
            account_ids = []
            
            for account_id, features in self.test_features.items():
                feature_dict = self._features_to_dict(features)
                X_test_list.append(feature_dict)
                account_ids.append(account_id)
            
            X_test = pd.DataFrame(X_test_list)
            
            logger.info(f"Test data: {len(X_test)} samples, {len(X_test.columns)} features")
            
            # Find optimal threshold using F1 score
            # Use a subset of training data as validation for threshold tuning
            logger.info("Finding optimal prediction threshold...")
            X_train_list = []
            y_train_list = []
            
            for account_id, features in self.train_features.items():
                feature_dict = self._features_to_dict(features)
                X_train_list.append(feature_dict)
                y_train_list.append(features.is_mule if features.is_mule is not None else 0)
            
            X_train_val = pd.DataFrame(X_train_list)
            y_train_val = np.array(y_train_list)
            
            # Use last 20% of training data for threshold tuning
            split_idx = int(len(X_train_val) * 0.8)
            X_val = X_train_val.iloc[split_idx:]
            y_val = y_train_val[split_idx:]
            
            optimal_threshold = self.ensemble.find_optimal_threshold(X_val, y_val, metric='f1')
            logger.info(f"Using optimal threshold: {optimal_threshold:.4f}")
            
            # Generate predictions with optimal threshold
            logger.info("Generating ensemble predictions...")
            predictions, confidences = self.ensemble.predict_ensemble(X_test, threshold=optimal_threshold)
            
            # Store predictions (keep probabilities for output)
            self.predictions = pd.DataFrame({
                'account_id': account_ids,
                'is_mule': predictions,  # Keep probabilities
                'confidence': confidences
            })
            
            logger.info(f"Generated predictions for {len(self.predictions)} accounts")
            logger.info("Phase 7 complete")
            return True
            
        except Exception as e:
            logger.error(f"Phase 7 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_phase_9_output_generation(self) -> bool:
        """Phase 9: Output Generation"""
        logger.info("=" * 80)
        logger.info("PHASE 9: OUTPUT GENERATION")
        logger.info("=" * 80)
        
        try:
            if self.predictions is None:
                logger.error("No predictions available")
                return False
            
            # Add suspicious windows
            output_data = []
            
            for _, row in self.predictions.iterrows():
                account_id = row['account_id']
                features = self.test_features.get(account_id)
                
                if features:
                    suspicious_start = getattr(features, 'suspicious_start', None)
                    suspicious_end = getattr(features, 'suspicious_end', None)
                else:
                    suspicious_start = None
                    suspicious_end = None
                
                # Format timestamps
                if suspicious_start:
                    suspicious_start = suspicious_start.isoformat()
                else:
                    suspicious_start = ''
                
                if suspicious_end:
                    suspicious_end = suspicious_end.isoformat()
                else:
                    suspicious_end = ''
                
                output_data.append({
                    'account_id': account_id,
                    'is_mule': row['is_mule'],
                    'suspicious_start': suspicious_start,
                    'suspicious_end': suspicious_end
                })
            
            output_df = pd.DataFrame(output_data)
            
            # Save to CSV
            output_path = self.output_dir / 'predictions.csv'
            output_df.to_csv(output_path, index=False)
            
            logger.info(f"Predictions saved to {output_path}")
            logger.info("Phase 9 complete")
            return True
            
        except Exception as e:
            logger.error(f"Phase 9 failed: {e}")
            return False
    
    def _features_to_dict(self, features: AccountFeatures) -> Dict:
        """Convert AccountFeatures to dictionary for model input"""
        feature_dict = {
            'total_transactions': features.total_transactions,
            'inflow_transactions': features.inflow_transactions,
            'outflow_transactions': features.outflow_transactions,
            'transaction_frequency_daily': features.transaction_frequency_daily,
            'total_inflow': features.total_inflow,
            'total_outflow': features.total_outflow,
            'avg_inflow_amount': features.avg_inflow_amount,
            'avg_outflow_amount': features.avg_outflow_amount,
            'inflow_std': features.inflow_std,
            'outflow_std': features.outflow_std,
            'sub_threshold_ratio': features.sub_threshold_ratio,
            'round_amount_ratio': features.round_amount_ratio,
            'inflow_outflow_ratio': features.inflow_outflow_ratio,
            'avg_time_to_transfer_hours': features.avg_time_to_transfer_hours,
            'rapid_transfer_ratio_24h': features.rapid_transfer_ratio_24h,
            'unique_sources': features.unique_sources,
            'unique_destinations': features.unique_destinations,
            'source_concentration': features.source_concentration,
            'account_age_days': features.account_age_days,
            'dormancy_periods': features.dormancy_periods,
            'activity_spike_magnitude': features.activity_spike_magnitude,
            'degree_centrality': features.degree_centrality,
            'betweenness_centrality': features.betweenness_centrality,
            'clustering_coefficient': features.clustering_coefficient,
            'composite_signal': self.label_signal_integrator.get_composite_signal(features.account_id)
        }
        
        return feature_dict
    
    def run_full_pipeline(self) -> bool:
        """Run complete pipeline"""
        logger.info("Starting Mule Detection Pipeline")
        logger.info(f"Output directory: {self.output_dir}")
        
        phases = [
            ('Phase 1', self.run_phase_1_data_ingestion),
            ('Phase 1b', self.run_phase_1b_load_label_signals),
            ('Phase 2', self.run_phase_2_feature_engineering),
            ('Phase 3', self.run_phase_3_pattern_detection),
            ('Phase 4', self.run_phase_4_graph_analysis),
            ('Phase 5', self.run_phase_5_temporal_analysis),
            ('Phase 6', self.run_phase_6_model_training),
            ('Phase 7', self.run_phase_7_ensemble_prediction),
            ('Phase 9', self.run_phase_9_output_generation),
        ]
        
        for phase_name, phase_func in phases:
            logger.info(f"\nRunning {phase_name}...")
            if not phase_func():
                logger.error(f"{phase_name} failed")
                return False
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        
        return True


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    data_dir = '/Users/shivangisingh/Desktop/archive/'
    output_dir = 'output'
    
    # Run pipeline
    pipeline = MuleDetectionPipeline(data_dir, output_dir)
    success = pipeline.run_full_pipeline()
    
    return success


if __name__ == '__main__':
    main()
