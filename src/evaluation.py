"""
Evaluation & Reporting Module for Mule Account Detection

Calculates metrics and generates comprehensive reports:
- Accuracy, Precision, Recall, F1 Score
- AUC-ROC, Confusion Matrix
- ROC Curve, Precision-Recall Curve
- Feature Importance Analysis
- Model Comparison
- Temporal IoU Metrics
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # AUC-ROC if probabilities available
        if y_pred_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['auc_roc'] = 0.0
        
        return metrics
    
    @staticmethod
    def calculate_temporal_iou(predicted_windows: Dict[str, Tuple],
                              actual_windows: Dict[str, Tuple]) -> Dict:
        """
        Calculate Temporal IoU metrics.
        
        Args:
            predicted_windows: Dictionary of account_id -> (start, end)
            actual_windows: Dictionary of account_id -> (start, end)
            
        Returns:
            Dictionary of IoU metrics
        """
        iou_scores = []
        
        for account_id in predicted_windows:
            if account_id not in actual_windows:
                continue
            
            pred_start, pred_end = predicted_windows[account_id]
            actual_start, actual_end = actual_windows[account_id]
            
            # Calculate intersection
            intersection_start = max(pred_start, actual_start)
            intersection_end = min(pred_end, actual_end)
            
            if intersection_start > intersection_end:
                intersection_days = 0
            else:
                intersection_days = (intersection_end - intersection_start).days
            
            # Calculate union
            union_start = min(pred_start, actual_start)
            union_end = max(pred_end, actual_end)
            union_days = (union_end - union_start).days
            
            # Calculate IoU
            if union_days > 0:
                iou = intersection_days / union_days
                iou_scores.append(iou)
        
        if iou_scores:
            return {
                'mean_iou': np.mean(iou_scores),
                'median_iou': np.median(iou_scores),
                'std_iou': np.std(iou_scores),
                'min_iou': np.min(iou_scores),
                'max_iou': np.max(iou_scores)
            }
        else:
            return {}


class FeatureImportanceAnalyzer:
    """Analyze feature importance across models"""
    
    @staticmethod
    def analyze_importance(model_importances: Dict[str, Dict[str, float]]) -> Dict:
        """
        Analyze feature importance across models.
        
        Args:
            model_importances: Dictionary of model_name -> {feature_name: importance}
            
        Returns:
            Dictionary of importance analysis
        """
        # Aggregate importance across models
        feature_scores = {}
        
        for model_name, importances in model_importances.items():
            for feature, importance in importances.items():
                if feature not in feature_scores:
                    feature_scores[feature] = []
                feature_scores[feature].append(importance)
        
        # Calculate consensus importance
        consensus_importance = {}
        for feature, scores in feature_scores.items():
            consensus_importance[feature] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'model_count': len(scores)
            }
        
        # Sort by mean importance
        sorted_features = sorted(
            consensus_importance.items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        )
        
        return {
            'consensus_importance': dict(sorted_features),
            'top_10_features': [f[0] for f in sorted_features[:10]],
            'low_consensus_features': [
                f[0] for f in sorted_features 
                if f[1]['std'] > f[1]['mean'] * 0.5
            ]
        }


class RedHerringDetector:
    """Detect red-herring patterns in features"""
    
    @staticmethod
    def detect_perfect_separation(X_train: pd.DataFrame, y_train: np.ndarray) -> List[str]:
        """
        Detect features with perfect separation between classes.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            List of features with perfect separation
        """
        perfect_separation = []
        
        for feature in X_train.columns:
            values = X_train[feature].values
            
            # Skip if too few unique values
            if len(np.unique(values)) < 2:
                continue
            
            class_0_vals = values[y_train == 0]
            class_1_vals = values[y_train == 1]
            
            if len(class_0_vals) > 0 and len(class_1_vals) > 0:
                # Check for perfect separation
                if (max(class_0_vals) < min(class_1_vals) or 
                    max(class_1_vals) < min(class_0_vals)):
                    perfect_separation.append(feature)
        
        return perfect_separation
    
    @staticmethod
    def detect_synthetic_patterns(X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, List[str]]:
        """
        Detect synthetic patterns (too regular, artificially injected).
        
        Synthetic patterns are characterized by:
        1. Extreme uniformity (very few unique values)
        2. Perfect periodicity (regular intervals)
        3. Unrealistic distributions (too smooth, too peaked)
        4. Artificial clustering (distinct groups with gaps)
        5. Suspiciously high correlation with labels
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of synthetic pattern types and affected features
        """
        synthetic_patterns = {
            'extreme_uniformity': [],
            'perfect_periodicity': [],
            'unrealistic_distribution': [],
            'artificial_clustering': [],
            'suspicious_label_correlation': []
        }
        
        for feature in X_train.columns:
            values = X_train[feature].values
            
            # Skip if insufficient data
            if len(values) < 20:
                continue
            
            # 1. Extreme Uniformity: Very few unique values relative to sample size
            unique_ratio = len(np.unique(values)) / len(values)
            if unique_ratio < 0.05:  # Less than 5% unique values
                synthetic_patterns['extreme_uniformity'].append(feature)
                continue
            
            # 2. Perfect Periodicity: Regular intervals between values
            if len(np.unique(values)) > 5:
                sorted_unique = np.sort(np.unique(values))
                if len(sorted_unique) > 2:
                    diffs = np.diff(sorted_unique)
                    # Check if differences are suspiciously regular
                    if len(diffs) > 2:
                        diff_std = np.std(diffs)
                        diff_mean = np.mean(diffs)
                        if diff_mean > 0 and diff_std / diff_mean < 0.05:  # CV < 5%
                            synthetic_patterns['perfect_periodicity'].append(feature)
                            continue
            
            # 3. Unrealistic Distribution: Too smooth or too peaked
            # Check for bimodal or multimodal distributions that are too regular
            try:
                hist, _ = np.histogram(values[~np.isnan(values)], bins=20)
                # Count peaks (local maxima)
                peaks = 0
                for i in range(1, len(hist) - 1):
                    if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                        peaks += 1
                
                # Too many peaks or too few peaks can indicate synthetic data
                if peaks > 8 or (peaks > 0 and peaks < 2 and np.max(hist) > len(values) * 0.3):
                    synthetic_patterns['unrealistic_distribution'].append(feature)
                    continue
            except:
                pass
            
            # 4. Artificial Clustering: Distinct groups with gaps
            if len(np.unique(values)) > 10:
                sorted_vals = np.sort(values[~np.isnan(values)])
                gaps = np.diff(sorted_vals)
                
                # Look for suspiciously large gaps
                if len(gaps) > 0:
                    gap_threshold = np.percentile(gaps[gaps > 0], 90)
                    large_gaps = np.sum(gaps > gap_threshold * 5)
                    
                    # If there are many large gaps, suggests artificial clustering
                    if large_gaps > len(gaps) * 0.1:
                        synthetic_patterns['artificial_clustering'].append(feature)
                        continue
            
            # 5. Suspicious Label Correlation: Feature too perfectly aligned with labels
            # Calculate point-biserial correlation (correlation between continuous and binary)
            try:
                valid_mask = ~np.isnan(values)
                if np.sum(valid_mask) > 10:
                    valid_values = values[valid_mask]
                    valid_labels = y_train[valid_mask]
                    
                    # Normalize values for correlation
                    if np.std(valid_values) > 0:
                        normalized_values = (valid_values - np.mean(valid_values)) / np.std(valid_values)
                        correlation = np.abs(np.corrcoef(normalized_values, valid_labels)[0, 1])
                        
                        # Suspiciously high correlation (>0.7) suggests synthetic pattern
                        if correlation > 0.7:
                            synthetic_patterns['suspicious_label_correlation'].append(feature)
            except:
                pass
        
        return synthetic_patterns
    
    @staticmethod
    def detect_red_herrings(X_train: pd.DataFrame, y_train: np.ndarray,
                           feature_importances: Dict[str, float]) -> Dict:
        """
        Detect potential red-herring features.
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_importances: Feature importance scores
            
        Returns:
            Dictionary of red-herring analysis
        """
        red_herrings = {
            'perfect_separation': [],
            'synthetic_patterns': {},
            'low_consensus': [],
            'high_variance': []
        }
        
        # Detect perfect separation
        red_herrings['perfect_separation'] = RedHerringDetector.detect_perfect_separation(
            X_train, y_train
        )
        
        # Detect synthetic patterns
        red_herrings['synthetic_patterns'] = RedHerringDetector.detect_synthetic_patterns(
            X_train, y_train
        )
        
        # Check for high variance
        for feature in X_train.columns:
            if feature not in feature_importances:
                continue
            
            values = X_train[feature].values
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) > 0:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                
                # High variance relative to mean
                if mean_val != 0 and std_val > np.abs(mean_val) * 2:
                    red_herrings['high_variance'].append(feature)
        
        return red_herrings


class ReportGenerator:
    """Generate comprehensive reports"""
    
    def __init__(self, output_dir: str):
        """
        Initialize report generator.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_evaluation_report(self, metrics: Dict, output_file: str = 'evaluation_report.txt') -> str:
        """
        Generate evaluation report.
        
        Args:
            metrics: Dictionary of metrics
            output_file: Output file name
            
        Returns:
            Path to report file
        """
        report_path = self.output_dir / output_file
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EVALUATION REPORT - MULE ACCOUNT DETECTION\n")
            f.write("=" * 80 + "\n\n")
            
            # Performance Metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy:  {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:    {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1 Score:  {metrics.get('f1_score', 0):.4f}\n")
            f.write(f"AUC-ROC:   {metrics.get('auc_roc', 0):.4f}\n\n")
            
            # Confusion Matrix
            if 'confusion_matrix' in metrics:
                f.write("CONFUSION MATRIX\n")
                f.write("-" * 80 + "\n")
                cm = metrics['confusion_matrix']
                f.write(f"True Negatives:  {cm[0][0]}\n")
                f.write(f"False Positives: {cm[0][1]}\n")
                f.write(f"False Negatives: {cm[1][0]}\n")
                f.write(f"True Positives:  {cm[1][1]}\n\n")
            
            # Performance Targets
            f.write("PERFORMANCE TARGETS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Target AUC-ROC:  >0.90 | Achieved: {metrics.get('auc_roc', 0):.4f}\n")
            f.write(f"Target Precision: >0.85 | Achieved: {metrics.get('precision', 0):.4f}\n")
            f.write(f"Target Recall:    >0.80 | Achieved: {metrics.get('recall', 0):.4f}\n")
            f.write(f"Target F1 Score:  >0.82 | Achieved: {metrics.get('f1_score', 0):.4f}\n\n")
            
            f.write("=" * 80 + "\n")
        
        logger.info(f"Evaluation report generated: {report_path}")
        return str(report_path)
    
    def generate_feature_importance_report(self, importance_analysis: Dict,
                                          output_file: str = 'feature_importance_report.txt') -> str:
        """
        Generate feature importance report.
        
        Args:
            importance_analysis: Dictionary of importance analysis
            output_file: Output file name
            
        Returns:
            Path to report file
        """
        report_path = self.output_dir / output_file
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FEATURE IMPORTANCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Top 10 Features
            f.write("TOP 10 IMPORTANT FEATURES\n")
            f.write("-" * 80 + "\n")
            
            consensus = importance_analysis.get('consensus_importance', {})
            for i, (feature, scores) in enumerate(list(consensus.items())[:10], 1):
                f.write(f"{i:2d}. {feature:40s} | Mean: {scores['mean']:.4f} | Std: {scores['std']:.4f}\n")
            
            f.write("\n")
            
            # Low Consensus Features (potential red-herrings)
            f.write("LOW CONSENSUS FEATURES (Potential Red-Herrings)\n")
            f.write("-" * 80 + "\n")
            
            low_consensus = importance_analysis.get('low_consensus_features', [])
            if low_consensus:
                for feature in low_consensus[:10]:
                    scores = consensus.get(feature, {})
                    f.write(f"  - {feature:40s} | Mean: {scores.get('mean', 0):.4f} | Std: {scores.get('std', 0):.4f}\n")
            else:
                f.write("  None detected\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Feature importance report generated: {report_path}")
        return str(report_path)
    
    def generate_red_herring_report(self, red_herrings: Dict,
                                   output_file: str = 'red_herring_report.txt') -> str:
        """
        Generate red-herring detection report.
        
        Args:
            red_herrings: Dictionary of red-herring analysis
            output_file: Output file name
            
        Returns:
            Path to report file
        """
        report_path = self.output_dir / output_file
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RED-HERRING DETECTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Perfect Separation
            f.write("FEATURES WITH PERFECT SEPARATION\n")
            f.write("-" * 80 + "\n")
            f.write("These features perfectly separate mule from non-mule accounts,\n")
            f.write("which is unrealistic and suggests synthetic injection.\n\n")
            if red_herrings.get('perfect_separation'):
                for feature in red_herrings['perfect_separation']:
                    f.write(f"  - {feature}\n")
            else:
                f.write("  None detected\n")
            f.write("\n")
            
            # Synthetic Patterns
            f.write("FEATURES WITH SYNTHETIC PATTERNS (Too Regular)\n")
            f.write("-" * 80 + "\n")
            f.write("These features exhibit artificial regularity suggesting manual injection:\n\n")
            
            synthetic_patterns = red_herrings.get('synthetic_patterns', {})
            
            if synthetic_patterns.get('extreme_uniformity'):
                f.write("  EXTREME UNIFORMITY (< 5% unique values):\n")
                for feature in synthetic_patterns['extreme_uniformity']:
                    f.write(f"    - {feature}\n")
                f.write("\n")
            
            if synthetic_patterns.get('perfect_periodicity'):
                f.write("  PERFECT PERIODICITY (Regular intervals):\n")
                for feature in synthetic_patterns['perfect_periodicity']:
                    f.write(f"    - {feature}\n")
                f.write("\n")
            
            if synthetic_patterns.get('unrealistic_distribution'):
                f.write("  UNREALISTIC DISTRIBUTION (Too smooth or peaked):\n")
                for feature in synthetic_patterns['unrealistic_distribution']:
                    f.write(f"    - {feature}\n")
                f.write("\n")
            
            if synthetic_patterns.get('artificial_clustering'):
                f.write("  ARTIFICIAL CLUSTERING (Distinct groups with gaps):\n")
                for feature in synthetic_patterns['artificial_clustering']:
                    f.write(f"    - {feature}\n")
                f.write("\n")
            
            if synthetic_patterns.get('suspicious_label_correlation'):
                f.write("  SUSPICIOUS LABEL CORRELATION (> 0.7 correlation):\n")
                for feature in synthetic_patterns['suspicious_label_correlation']:
                    f.write(f"    - {feature}\n")
                f.write("\n")
            
            if not any(synthetic_patterns.values()):
                f.write("  None detected\n\n")
            
            # High Variance
            f.write("FEATURES WITH HIGH VARIANCE\n")
            f.write("-" * 80 + "\n")
            if red_herrings.get('high_variance'):
                for feature in red_herrings['high_variance']:
                    f.write(f"  - {feature}\n")
            else:
                f.write("  None detected\n")
            f.write("\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            total_red_herrings = (
                len(red_herrings.get('perfect_separation', [])) +
                sum(len(v) for v in red_herrings.get('synthetic_patterns', {}).values()) +
                len(red_herrings.get('high_variance', []))
            )
            f.write(f"Total Red-Herring Features Detected: {total_red_herrings}\n\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            if red_herrings.get('perfect_separation'):
                f.write("  - CRITICAL: Remove features with perfect separation\n")
            if synthetic_patterns.get('extreme_uniformity') or synthetic_patterns.get('perfect_periodicity'):
                f.write("  - HIGH PRIORITY: Review and remove features with extreme uniformity or periodicity\n")
            if synthetic_patterns.get('suspicious_label_correlation'):
                f.write("  - MEDIUM PRIORITY: Investigate features with suspiciously high label correlation\n")
            if not any(red_herrings.values()):
                f.write("  - No red-herring features detected. Feature set appears robust.\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Red-herring report generated: {report_path}")
        return str(report_path)
    
    def generate_summary_report(self, metrics: Dict, importance_analysis: Dict,
                               red_herrings: Dict, output_file: str = 'summary_report.txt') -> str:
        """
        Generate comprehensive summary report.
        
        Args:
            metrics: Evaluation metrics
            importance_analysis: Feature importance analysis
            red_herrings: Red-herring detection results
            output_file: Output file name
            
        Returns:
            Path to report file
        """
        report_path = self.output_dir / output_file
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MULE ACCOUNT DETECTION - COMPREHENSIVE SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Model Performance:\n")
            f.write(f"  - Accuracy:  {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"  - Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"  - Recall:    {metrics.get('recall', 0):.4f}\n")
            f.write(f"  - F1 Score:  {metrics.get('f1_score', 0):.4f}\n")
            f.write(f"  - AUC-ROC:   {metrics.get('auc_roc', 0):.4f}\n\n")
            
            # Key Findings
            f.write("KEY FINDINGS\n")
            f.write("-" * 80 + "\n")
            
            top_features = importance_analysis.get('top_10_features', [])
            f.write(f"Top 3 Important Features:\n")
            for i, feature in enumerate(top_features[:3], 1):
                f.write(f"  {i}. {feature}\n")
            f.write("\n")
            
            red_herring_count = sum(len(v) for v in red_herrings.values())
            f.write(f"Red-Herring Features Detected: {red_herring_count}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            
            if metrics.get('auc_roc', 0) < 0.90:
                f.write("  - Consider feature engineering improvements\n")
            
            if red_herring_count > 0:
                f.write("  - Review and potentially remove red-herring features\n")
            
            if metrics.get('recall', 0) < 0.80:
                f.write("  - Increase model sensitivity to catch more mules\n")
            
            if metrics.get('precision', 0) < 0.85:
                f.write("  - Reduce false positives through threshold optimization\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Summary report generated: {report_path}")
        return str(report_path)
