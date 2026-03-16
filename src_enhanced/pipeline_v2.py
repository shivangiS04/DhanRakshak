"""
Enhanced Pipeline for Mule Account Detection

Uses advanced models and features:
- LightGBM & CatBoost boosting
- 50+ engineered features
- Enhanced graph analysis
- Velocity-based detection
"""
#pipeline_v2
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MuleDetectionPipelineV2:
    """Enhanced pipeline with advanced models and features"""
    
    def __init__(self, data_dir: str, output_dir: str = 'output_enhanced'):
        """
        Initialize enhanced pipeline.
        
        Args:
            data_dir: Path to data directory
            output_dir: Path to output directory
        """
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import enhanced modules
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src_enhanced.ensemble_models_v2 import EnsembleModelStackV2
        from src_enhanced.feature_engineering_v2 import FeatureExtractorV2, FeatureScalerV2
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2, GraphFeatureExtractorV2
        
        # Import original modules for data loading
        from src.data_loader import DataLoader
        from src.pattern_detection import PatternDetector, CompositeAnomalyScorer
        from src.temporal_analysis import TemporalAnalyzer
        from src.label_signal_integration import LabelSignalIntegratoryScorer
        from temporal_analysis import TemporalAnalyzer
        from label_signal_integration import LabelSignalIntegrator
        
        # Components
        self.data_loader = DataLoader(data_dir, str(self.output_dir / 'processed_data'))
        self.feature_extractor = None
        self.feature_scaler = FeatureScalerV2()
        self.pattern_detector = PatternDetector()
        self.anomaly_scorer = CompositeAnomalyScorer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.ensemble = EnsembleModelStackV2(str(self.output_dir / 'models_enhanced'))
        self.label_signal_integrator = LabelSignalIntegrator(str(self.output_dir / 'label_signals.csv'))
        self.graph = TransactionNetworkGraphV2()
        self.graph_feature_extractor = None
        
        # Data containers
        self.train_features = {}
        self.test_features = {}
        self.train_labels = None
        self.test_accounts = None
        self.predictions = None
        
        logger.info(f"MuleDetectionPipelineV2 initialized with data_dir: {data_dir}")
    
    def run_phase_1_data_ingestion(self) -> bool:
        """Phase 1: Data Ingestion & Preprocessing"""
        logger.info("=" * 80)
        logger.info("PHASE 1: DATA INGESTION & PREPROCESSING (ENHANCED)")
        logger.info("=" * 80)
        
        try:
            if not self.data_loader.load_all_data():
                logger.error("Failed to load data")
                return False
            
            if not self.data_loader.validate_temporal_consistency():
                logger.warning("Temporal consistency validation failed")
            
            report_path = self.data_loader.generate_quality_report(
                str(self.output_dir / 'data_quality_report.txt')
            )
            logger.info(f"Quality report: {report_path}")
            
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
        logger.info("PHASE 1B: LOAD LABEL SIGNALS (ENHANCED)")
        logger.info("=" * 80)
        
        try:
            if self.label_signal_integrator.load_signals():
                stats = self.label_signal_integrator.get_statistics()
                logger.info(f"Label signals loaded successfully")
                logger.info(f"  Total accounts: {stats.get('total_accounts', 0)}")
                logger.info(f"  Accounts with signals: {stats.get('accounts_with_signals', 0)}")
                logger.info("Phase 1b complete")
                return True
            else:
                logger.warning("Label signals not available, continuing without them")
                return True
            
        except Exception as e:
            logger.error(f"Phase 1b failed: {e}")
            return False
    
    def run_phase_2_feature_engineering(self) -> bool:
        """Phase 2: Enhanced Feature Engineering (50+ features)"""
        logger.info("=" * 80)
        logger.info("PHASE 2: ENHANCED FEATURE ENGINEERING (50+ FEATURES)")
        logger.info("=" * 80)
        
        try:
            from src_enhanced.feature_engineering_v2 import FeatureExtractorV2
            
            self.feature_extractor = FeatureExtractorV2(
                self.data_loader.accounts,
                self.data_loader.customers
            )
            
            # Extract features for training accounts
            logger.info("Extracting enhanced features for training accounts...")
            train_account_ids = self.train_labels['account_id'].tolist()
            
            for i, account_id in enumerate(train_account_ids):
                if i % 1000 == 0:
                    logger.info(f"Processing account {i}/{len(train_account_ids)}")
                
                transactions = self.data_loader.get_transactions_for_account(account_id)
                if transactions is None or transactions.empty:
                    transactions = pd.DataFrame()
                
                features = self.feature_extractor.extract_features_for_account(account_id, transactions)
                
                label = self.train_labels[self.train_labels['account_id'] == account_id]['is_mule'].values
                if len(label) > 0:
                    features.is_mule = int(label[0])
                
                self.train_features[account_id] = features
            
            logger.info(f"Extracted enhanced features for {len(self.train_features)} training accounts")
            
            # Extract features for test accounts
            logger.info("Extracting enhanced features for test accounts...")
            test_account_ids = self.test_accounts['account_id'].tolist()
            
            for i, account_id in enumerate(test_account_ids):
                if i % 1000 == 0:
                    logger.info(f"Processing account {i}/{len(test_account_ids)}")
                
                transactions = self.data_loader.get_transactions_for_account(account_id)
                if transactions is None or transactions.empty:
                    transactions = pd.DataFrame()
                
                features = self.feature_extractor.extract_features_for_account(account_id, transactions)
                self.test_features[account_id] = features
            
            logger.info(f"Extracted enhanced features for {len(self.test_features)} test accounts")
            
            # Initialize scaler
            self.feature_scaler.fit(list(self.train_features.values()))
            
            logger.info("Phase 2 complete")
            return True
            
        except Exception as e:
            logger.error(f"Phase 2 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_phase_4_graph_analysis(self) -> bool:
        """Phase 4: Enhanced Graph Analysis"""
        logger.info("=" * 80)
        logger.info("PHASE 4: ENHANCED GRAPH ANALYSIS")
        logger.info("=" * 80)
        
        try:
            logger.info("Building enhanced transaction network graph...")
            
            all_transactions = []
            for chunk in self.data_loader.iterate_transaction_chunks():
                all_transactions.append(chunk)
            
            if all_transactions:
                all_txns_df = pd.concat(all_transactions, ignore_index=True)
                
                # Build graph
                self.graph.build_graph(all_txns_df)
                
                # Extract graph features
                from src_enhanced.graph_analysis_v2 import GraphFeatureExtractorV2
                self.graph_feature_extractor = GraphFeatureExtractorV2(self.graph)
                self.graph_feature_extractor.compute_all_features()
                
                # Detect communities and suspicious patterns
                communities = self.graph.detect_communities()
                suspicious_comms = self.graph.identify_suspicious_communities(communities, all_txns_df)
                
                logger.info(f"Detected {len(communities)} communities")
                logger.info(f"Suspicious communities: {len(suspicious_comms)}")
                
                # Detect circular flows
                circular_flows = self.graph.detect_circular_flows()
                logger.info(f"Detected {len(circular_flows)} circular flows")
                
                # Detect layering patterns
                layering_patterns = self.graph.detect_layering_patterns()
                logger.info(f"Detected {len(layering_patterns)} layering patterns")
                
                # Store graph features in account features
                for account_id in self.train_features:
                    graph_features = self.graph_feature_extractor.get_features_for_account(account_id)
                    for key, value in graph_features.items():
                        setattr(self.train_features[account_id], key, value)
                
                for account_id in self.test_features:
                    graph_features = self.graph_feature_extractor.get_features_for_account(account_id)
                    for key, value in graph_features.items():
                        setattr(self.test_features[account_id], key, value)
            
            logger.info("Phase 4 complete")
            return True
            
        except Exception as e:
            logger.error(f"Phase 4 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_phase_6_model_training(self) -> bool:
        """Phase 6: Enhanced Model Training (LightGBM, CatBoost)"""
        logger.info("=" * 80)
        logger.info("PHASE 6: ENHANCED MODEL TRAINING (LightGBM, CatBoost)")
        logger.info("=" * 80)
        
        try:
            logger.info("Preparing training data...")
            
            X_train_list = []
            y_train_list = []
            
            for account_id, features in self.train_features.items():
                feature_dict = self._features_to_dict(features)
                X_train_list.append(feature_dict)
                y_train_list.append(features.is_mule if features.is_mule is not None else 0)
            
            X_train = pd.DataFrame(X_train_list)
            y_train = np.array(y_train_list)
            
            logger.info(f"Training data: {len(X_train)} samples, {len(X_train.columns)} features")
            logger.info(f"Class distribution: {np.bincount(y_train.astype(int))}")
            
            # Train enhanced ensemble
            logger.info("Training enhanced ensemble with LightGBM, CatBoost, XGBoost...")
            metrics = self.ensemble.train_all_models(X_train, y_train, use_smote=True)
            
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
        """Phase 7: Enhanced Ensemble Prediction"""
        logger.info("=" * 80)
        logger.info("PHASE 7: ENHANCED ENSEMBLE PREDICTION")
        logger.info("=" * 80)
        
        try:
            logger.info("Preparing test data...")
            
            X_test_list = []
            account_ids = []
            
            for account_id, features in self.test_features.items():
                feature_dict = self._features_to_dict(features)
                X_test_list.append(feature_dict)
                account_ids.append(account_id)
            
            X_test = pd.DataFrame(X_test_list)
            
            logger.info(f"Test data: {len(X_test)} samples, {len(X_test.columns)} features")
            
            # Find optimal threshold
            logger.info("Finding optimal prediction threshold...")
            X_train_list = []
            y_train_list = []
            
            for account_id, features in self.train_features.items():
                feature_dict = self._features_to_dict(features)
                X_train_list.append(feature_dict)
                y_train_list.append(features.is_mule if features.is_mule is not None else 0)
            
            X_train_val = pd.DataFrame(X_train_list)
            y_train_val = np.array(y_train_list)
            
            split_idx = int(len(X_train_val) * 0.8)
            X_val = X_train_val.iloc[split_idx:]
            y_val = y_train_val[split_idx:]
            
            optimal_threshold = self.ensemble.find_optimal_threshold(X_val, y_val, metric='f1')
            logger.info(f"Using optimal threshold: {optimal_threshold:.4f}")
            
            # Generate predictions
            logger.info("Generating enhanced ensemble predictions...")
            predictions, confidences = self.ensemble.predict_ensemble(X_test, threshold=optimal_threshold)
            
            self.predictions = pd.DataFrame({
                'account_id': account_ids,
                'is_mule': predictions,
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
        logger.info("PHASE 9: OUTPUT GENERATION (ENHANCED)")
        logger.info("=" * 80)
        
        try:
            if self.predictions is None:
                logger.error("No predictions available")
                return False
            
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
            
            output_path = self.output_dir / 'predictions_enhanced.csv'
            output_df.to_csv(output_path, index=False)
            
            logger.info(f"Predictions saved to {output_path}")
            logger.info("Phase 9 complete")
            return True
            
        except Exception as e:
            logger.error(f"Phase 9 failed: {e}")
            return False
    
    def _features_to_dict(self, features) -> Dict:
        """Convert features to dictionary for model input"""
        return {
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
            'inflow_cv': features.inflow_cv,
            'outflow_cv': features.outflow_cv,
            'amount_skewness': features.amount_skewness,
            'amount_kurtosis': features.amount_kurtosis,
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
            'transaction_time_entropy': features.transaction_time_entropy,
            'day_of_week_concentration': features.day_of_week_concentration,
            'degree_centrality': features.degree_centrality,
            'betweenness_centrality': features.betweenness_centrality,
            'clustering_coefficient': features.clustering_coefficient,
            'pagerank_score': features.pagerank_score,
            'community_size': features.community_size,
            'community_density': features.community_density,
            'velocity_inflow_1h': features.velocity_inflow_1h,
            'velocity_inflow_24h': features.velocity_inflow_24h,
            'velocity_outflow_1h': features.velocity_outflow_1h,
            'velocity_outflow_24h': features.velocity_outflow_24h,
            'velocity_spike_ratio': features.velocity_spike_ratio,
            'inter_transaction_time_mean': features.inter_transaction_time_mean,
            'inter_transaction_time_std': features.inter_transaction_time_std,
            'transaction_burst_score': features.transaction_burst_score,
            'pattern_anomaly_score': features.pattern_anomaly_score,
            'pattern_confidence': features.pattern_confidence,
            'composite_signal': self.label_signal_integrator.get_composite_signal(features.account_id),
            'risk_score': features.risk_score,
            'avg_counterparty_degree': features.avg_counterparty_degree,
            'counterparty_diversity': features.counterparty_diversity,
            'shared_counterparty_ratio': features.shared_counterparty_ratio,
            'network_risk_score': features.network_risk_score,
        }
    
    def run_full_pipeline(self) -> bool:
        """Run complete enhanced pipeline"""
        logger.info("Starting Enhanced Mule Detection Pipeline")
        logger.info(f"Output directory: {self.output_dir}")
        
        phases = [
            ('Phase 1', self.run_phase_1_data_ingestion),
            ('Phase 1b', self.run_phase_1b_load_label_signals),
            ('Phase 2', self.run_phase_2_feature_engineering),
            ('Phase 4', self.run_phase_4_graph_analysis),
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
        logger.info("ENHANCED PIPELINE COMPLETE")
        logger.info("=" * 80)
        
        return True


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    data_dir = '/Users/shivangisingh/Desktop/archive/'
    output_dir = 'output_enhanced'
    
    pipeline = MuleDetectionPipelineV2(data_dir, output_dir)
    success = pipeline.run_full_pipeline()
    
    return success


if __name__ == '__main__':
    main()
