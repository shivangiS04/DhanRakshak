"""
Graph Analysis Engine for Mule Account Detection

Analyzes transaction network structure to detect anomalies:
- Community detection (Louvain algorithm)
- Centrality analysis (degree, betweenness, closeness, eigenvector)
- Money flow path tracing
- Circular flow and layering detection
- Isolation Forest for graph anomalies
- Local Outlier Factor for neighborhood anomalies
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)


class TransactionNetworkGraph:
    """Build and analyze transaction network graph"""
    
    def __init__(self):
        """Initialize graph"""
        self.graph = nx.DiGraph()
        self.account_features = {}
        logger.info("TransactionNetworkGraph initialized")
    
    def build_graph(self, transactions: pd.DataFrame) -> None:
        """
        Build transaction network graph.
        
        Args:
            transactions: DataFrame with transaction data
        """
        if 'account_id' not in transactions.columns or 'counterparty_account_id' not in transactions.columns:
            logger.warning("Missing required columns for graph building")
            return
        
        # Add edges for each transaction
        for _, row in transactions.iterrows():
            source = row['account_id']
            dest = row['counterparty_account_id']
            amount = row.get('amount', 1)
            
            # Add edge with weight
            if self.graph.has_edge(source, dest):
                self.graph[source][dest]['weight'] += amount
                self.graph[source][dest]['count'] += 1
            else:
                self.graph.add_edge(source, dest, weight=amount, count=1)
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def calculate_degree_centrality(self) -> Dict[str, float]:
        """
        Calculate degree centrality for all accounts.
        
        Returns:
            Dictionary of account_id -> degree_centrality
        """
        centrality = nx.degree_centrality(self.graph)
        logger.info(f"Calculated degree centrality for {len(centrality)} accounts")
        return centrality
    
    def calculate_betweenness_centrality(self) -> Dict[str, float]:
        """
        Calculate betweenness centrality (bridge accounts).
        
        Returns:
            Dictionary of account_id -> betweenness_centrality
        """
        centrality = nx.betweenness_centrality(self.graph, weight='weight')
        logger.info(f"Calculated betweenness centrality for {len(centrality)} accounts")
        return centrality
    
    def calculate_closeness_centrality(self) -> Dict[str, float]:
        """
        Calculate closeness centrality.
        
        Returns:
            Dictionary of account_id -> closeness_centrality
        """
        # For directed graphs, use weakly connected components
        centrality = {}
        for node in self.graph.nodes():
            try:
                centrality[node] = nx.closeness_centrality(self.graph, node)
            except:
                centrality[node] = 0.0
        
        logger.info(f"Calculated closeness centrality for {len(centrality)} accounts")
        return centrality
    
    def calculate_clustering_coefficient(self) -> Dict[str, float]:
        """
        Calculate clustering coefficient (local network density).
        
        Returns:
            Dictionary of account_id -> clustering_coefficient
        """
        # Convert to undirected for clustering coefficient
        undirected = self.graph.to_undirected()
        clustering = nx.clustering(undirected)
        logger.info(f"Calculated clustering coefficient for {len(clustering)} accounts")
        return clustering
    
    def detect_communities(self) -> Dict[str, int]:
        """
        Detect communities using Louvain algorithm.
        
        Returns:
            Dictionary of account_id -> community_id
        """
        try:
            # Convert to undirected for community detection
            undirected = self.graph.to_undirected()
            
            # Use greedy modularity optimization (similar to Louvain)
            communities = nx.community.greedy_modularity_communities(undirected)
            
            community_map = {}
            for community_id, community in enumerate(communities):
                for node in community:
                    community_map[node] = community_id
            
            logger.info(f"Detected {len(communities)} communities")
            return community_map
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            return {}
    
    def identify_suspicious_communities(self, community_map: Dict[str, int], 
                                       mule_labels: Optional[Dict[str, int]] = None) -> Dict[int, Dict]:
        """
        Identify suspicious communities.
        
        Args:
            community_map: Dictionary of account_id -> community_id
            mule_labels: Optional dictionary of account_id -> is_mule label
            
        Returns:
            Dictionary of community_id -> community_info
        """
        communities = defaultdict(list)
        for account_id, community_id in community_map.items():
            communities[community_id].append(account_id)
        
        suspicious_communities = {}
        
        for community_id, members in communities.items():
            if len(members) < 3:
                continue
            
            # Calculate community statistics
            mule_count = 0
            if mule_labels:
                mule_count = sum(1 for m in members if mule_labels.get(m, 0) == 1)
            
            # Calculate internal edge density
            internal_edges = 0
            for source in members:
                for dest in members:
                    if source != dest and self.graph.has_edge(source, dest):
                        internal_edges += 1
            
            max_edges = len(members) * (len(members) - 1)
            density = internal_edges / max_edges if max_edges > 0 else 0
            
            # Flag as suspicious if high mule ratio or high density
            if (mule_labels and mule_count / len(members) > 0.5) or density > 0.3:
                suspicious_communities[community_id] = {
                    'members': members,
                    'size': len(members),
                    'mule_count': mule_count,
                    'density': density
                }
        
        logger.info(f"Identified {len(suspicious_communities)} suspicious communities")
        return suspicious_communities
    
    def trace_money_flow_paths(self, source_account: str, dest_account: str, 
                              max_depth: int = 5) -> List[List[str]]:
        """
        Trace money flow paths from source to destination.
        
        Args:
            source_account: Source account ID
            dest_account: Destination account ID
            max_depth: Maximum path depth
            
        Returns:
            List of paths (each path is a list of account IDs)
        """
        paths = []
        
        try:
            # Find all simple paths up to max_depth
            for path in nx.all_simple_paths(self.graph, source_account, dest_account, cutoff=max_depth):
                paths.append(path)
        except nx.NetworkXNoPath:
            pass
        except nx.NodeNotFound:
            pass
        
        return paths
    
    def detect_circular_flows(self) -> List[List[str]]:
        """
        Detect circular flows (money laundering loops).
        
        Returns:
            List of cycles (each cycle is a list of account IDs)
        """
        cycles = []
        
        try:
            # Find all simple cycles
            for cycle in nx.simple_cycles(self.graph):
                if len(cycle) >= 3:  # Only cycles with 3+ nodes
                    cycles.append(cycle)
        except Exception as e:
            logger.warning(f"Error detecting cycles: {e}")
        
        logger.info(f"Detected {len(cycles)} circular flows")
        return cycles
    
    def detect_layering_patterns(self) -> Dict[str, List[List[str]]]:
        """
        Detect layering patterns (multiple hops for obfuscation).
        
        Returns:
            Dictionary of pattern_type -> list_of_patterns
        """
        patterns = {
            'three_hop_paths': [],
            'four_hop_paths': [],
            'five_hop_paths': []
        }
        
        # Find paths of specific lengths
        for source in self.graph.nodes():
            for dest in self.graph.nodes():
                if source == dest:
                    continue
                
                try:
                    # Find shortest path
                    path = nx.shortest_path(self.graph, source, dest)
                    
                    if len(path) == 4:  # 3 hops
                        patterns['three_hop_paths'].append(path)
                    elif len(path) == 5:  # 4 hops
                        patterns['four_hop_paths'].append(path)
                    elif len(path) == 6:  # 5 hops
                        patterns['five_hop_paths'].append(path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        
        logger.info(f"Detected layering patterns: {sum(len(v) for v in patterns.values())} total")
        return patterns


class GraphAnomalyDetector:
    """Detect anomalies using graph-based methods"""
    
    def __init__(self):
        """Initialize detector"""
        logger.info("GraphAnomalyDetector initialized")
    
    def detect_with_isolation_forest(self, graph_features: pd.DataFrame, 
                                     contamination: float = 0.1) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            graph_features: DataFrame with graph-based features
            contamination: Expected proportion of anomalies
            
        Returns:
            Array of anomaly scores (-1 for anomaly, 1 for normal)
        """
        if graph_features.empty:
            return np.array([])
        
        # Select numeric columns
        numeric_cols = graph_features.select_dtypes(include=[np.number]).columns
        X = graph_features[numeric_cols].fillna(0)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(X)
        
        logger.info(f"Isolation Forest detected {(predictions == -1).sum()} anomalies")
        return predictions
    
    def detect_with_lof(self, graph_features: pd.DataFrame, 
                       n_neighbors: int = 20) -> np.ndarray:
        """
        Detect anomalies using Local Outlier Factor.
        
        Args:
            graph_features: DataFrame with graph-based features
            n_neighbors: Number of neighbors for LOF
            
        Returns:
            Array of LOF scores (higher = more anomalous)
        """
        if graph_features.empty:
            return np.array([])
        
        # Select numeric columns
        numeric_cols = graph_features.select_dtypes(include=[np.number]).columns
        X = graph_features[numeric_cols].fillna(0)
        
        # Fit LOF
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        lof_scores = lof.fit_predict(X)
        
        logger.info(f"LOF detected {(lof_scores == -1).sum()} anomalies")
        return lof_scores


class GraphFeatureExtractor:
    """Extract graph-based features for accounts"""
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize extractor.
        
        Args:
            graph: NetworkX directed graph
        """
        self.graph = graph
        self.degree_centrality = None
        self.betweenness_centrality = None
        self.clustering_coefficient = None
        self.community_map = None
    
    def compute_all_features(self) -> None:
        """Compute all graph features"""
        logger.info("Computing graph features...")
        
        network = TransactionNetworkGraph()
        network.graph = self.graph
        
        self.degree_centrality = network.calculate_degree_centrality()
        self.betweenness_centrality = network.calculate_betweenness_centrality()
        self.clustering_coefficient = network.calculate_clustering_coefficient()
        self.community_map = network.detect_communities()
        
        logger.info("Graph features computed")
    
    def get_features_for_account(self, account_id: str) -> Dict[str, float]:
        """
        Get graph features for a specific account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Dictionary of feature_name -> value
        """
        features = {
            'degree_centrality': self.degree_centrality.get(account_id, 0.0) if self.degree_centrality else 0.0,
            'betweenness_centrality': self.betweenness_centrality.get(account_id, 0.0) if self.betweenness_centrality else 0.0,
            'clustering_coefficient': self.clustering_coefficient.get(account_id, 0.0) if self.clustering_coefficient else 0.0,
            'community_id': self.community_map.get(account_id, -1) if self.community_map else -1
        }
        
        return features
