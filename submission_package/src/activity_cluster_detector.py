"""
activity_cluster_detector.py
Detect dense transaction clusters to find mule activity windows
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class ActivityClusterDetector:
    """Detect transaction clusters and find densest burst"""
    
    def __init__(self, gap_threshold_hours=12, min_cluster_size=2):
        """
        gap_threshold_hours: Hours between transactions to consider a gap
        min_cluster_size: Minimum transactions in a cluster
        """
        self.gap_threshold = timedelta(hours=gap_threshold_hours)
        self.min_cluster_size = min_cluster_size
    
    def detect_clusters(self, txn_times):
        """
        Detect transaction clusters from sorted timestamps
        
        Args:
            txn_times: List/array of transaction timestamps (datetime or numeric)
        
        Returns:
            List of clusters, each cluster is a list of timestamps
        """
        if len(txn_times) < self.min_cluster_size:
            return [txn_times]
        
        # Convert to datetime if needed
        times = self._to_datetime(txn_times)
        times = sorted(times)
        
        # Find gaps
        clusters = []
        current_cluster = [times[0]]
        
        for i in range(1, len(times)):
            gap = times[i] - times[i-1]
            
            if gap <= self.gap_threshold:
                current_cluster.append(times[i])
            else:
                # Gap found - save cluster and start new one
                if len(current_cluster) >= self.min_cluster_size:
                    clusters.append(current_cluster)
                current_cluster = [times[i]]
        
        # Add last cluster
        if len(current_cluster) >= self.min_cluster_size:
            clusters.append(current_cluster)
        
        return clusters if clusters else [times]
    
    def get_densest_cluster(self, txn_times):
        """
        Find the densest transaction cluster
        
        Returns:
            (cluster_start, cluster_end, density, txn_count)
        """
        clusters = self.detect_clusters(txn_times)
        
        if not clusters:
            return None, None, 0, 0
        
        best_cluster = None
        best_density = 0
        
        for cluster in clusters:
            if len(cluster) < self.min_cluster_size:
                continue
            
            time_span = (cluster[-1] - cluster[0]).total_seconds() / 3600  # hours
            if time_span == 0:
                time_span = 1
            
            density = len(cluster) / time_span
            
            if density > best_density:
                best_density = density
                best_cluster = cluster
        
        if best_cluster is None:
            return None, None, 0, 0
        
        return best_cluster[0], best_cluster[-1], best_density, len(best_cluster)
    
    def get_window_from_cluster(self, txn_times, pad_hours=24):
        """
        Get window around densest cluster with optional padding
        
        Args:
            txn_times: Transaction timestamps
            pad_hours: Hours to pad before/after cluster
        
        Returns:
            (window_start, window_end)
        """
        cluster_start, cluster_end, density, count = self.get_densest_cluster(txn_times)
        
        if cluster_start is None:
            return None, None
        
        # Add padding
        padding = timedelta(hours=pad_hours)
        window_start = cluster_start - padding
        window_end = cluster_end + padding
        
        return window_start, window_end
    
    @staticmethod
    def _to_datetime(times):
        """Convert timestamps to datetime if needed"""
        if not times:
            return []
        
        first = times[0]
        
        # Already datetime
        if isinstance(first, (datetime, pd.Timestamp)):
            return list(times)
        
        # Numeric (assume seconds since epoch)
        if isinstance(first, (int, float)):
            return [datetime.fromtimestamp(t) for t in times]
        
        # String
        if isinstance(first, str):
            return [pd.to_datetime(t) for t in times]
        
        return list(times)


def detect_activity_clusters_batch(transactions_df, account_ids, gap_threshold_hours=12):
    """
    Detect clusters for multiple accounts
    
    Args:
        transactions_df: DataFrame with columns [account_id, timestamp]
        account_ids: List of account IDs to process
        gap_threshold_hours: Gap threshold in hours
    
    Returns:
        Dict mapping account_id -> (window_start, window_end, density, txn_count)
    """
    detector = ActivityClusterDetector(gap_threshold_hours=gap_threshold_hours)
    results = {}
    
    for account_id in account_ids:
        account_txns = transactions_df[transactions_df['account_id'] == account_id]
        
        if len(account_txns) == 0:
            results[account_id] = (None, None, 0, 0)
            continue
        
        times = sorted(account_txns['timestamp'].values)
        cluster_start, cluster_end, density, count = detector.get_densest_cluster(times)
        
        results[account_id] = (cluster_start, cluster_end, density, count)
    
    return results
