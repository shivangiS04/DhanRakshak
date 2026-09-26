"""
Enhanced Graph Analysis for Mule Account Detection

Advanced network analysis with:
- Community detection
- Suspicious cluster identification
- Money flow path tracing
- Circular flow detection
- Layering pattern detection
- PageRank scoring
"""
#graph analysis
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class TransactionNetworkGraphV2:
    """Enhanced transaction network graph analysis"""
    
    def __init__(self):
        """Initialize graph"""
        if not HAS_NETWORKX:
            logger.warning("NetworkX not available")
            self.graph = None
        else:
            self.graph = nx.DiGraph()
        
        self.account_features = {}
        logger.info("TransactionNetworkGraphV2 initialized")
    
    def build_graph(self, transactions: pd.DataFrame) -> None:
        """Build transaction network graph"""
        if not HAS_NETWORKX or self.graph is None:
            logger.warning("Cannot build graph without NetworkX")
            return
        
        logger.info(f"Building graph from {len(transactions)} transactions...")
        
        for _, txn in transactions.iterrows():
            source = txn.get('source_account')
            dest = txn.get('destination_account')
            amount = txn.get('amount', 0)
            
            if source and dest:
                if self.graph.has_edge(source, dest):
                    self.graph[source][dest]['weight'] += amount
                    self.graph[source][dest]['count'] += 1
                else:
                    self.graph.add_edge(source, dest, weight=amount, count=1)
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def calculate_degree_centrality(self) -> Dict[str, float]:
        """Calculate degree centrality"""
        if not HAS_NETWORKX or self.graph is None:
            return {}
        
        return nx.degree_centrality(self.graph)
    
    def calculate_betweenness_centrality(self) -> Dict[str, float]:
        """Calculate betweenness centrality"""
        if not HAS_NETWORKX or self.graph is None:
            return {}
        
        try:
            return nx.betweenness_centrality(self.graph)
        except:
            logger.warning("Betweenness centrality calculation failed")
            return {}
    
    def calculate_closeness_centrality(self) -> Dict[str, float]:
        """Calculate closeness centrality"""
        if not HAS_NETWORKX or self.graph is None:
            return {}
        
        try:
            return nx.closeness_centrality(self.graph)
        except:
            logger.warning("Closeness centrality calculation failed")
            return {}
    
    def calculate_clustering_coefficient(self) -> Dict[str, float]:
        """Calculate clustering coefficient"""
        if not HAS_NETWORKX or self.graph is None:
            return {}
        
        try:
            undirected = self.graph.to_undirected()
            return nx.clustering(undirected)
        except:
            logger.warning("Clustering coefficient calculation failed")
            return {}
    
    def calculate_pagerank(self) -> Dict[str, float]:
        """Calculate PageRank scores"""
        if not HAS_NETWORKX or self.graph is None:
            return {}
        
        try:
            return nx.pagerank(self.graph, alpha=0.85)
        except:
            logger.warning("PageRank calculation failed")
            return {}
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect communities using Louvain method"""
        if not HAS_NETWORKX or self.graph is None:
            return {}
        
        try:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(self.graph.to_undirected())
            
            community_map = {}
            for comm_id, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = comm_id
            
            logger.info(f"Detected {len(communities)} communities")
            return community_map
        except:
            logger.warning("Community detection failed")
            return {}
    
    def identify_suspicious_communities(self, community_map: Dict[str, int],
                                       transactions: pd.DataFrame) -> Dict[int, float]:
        """Identify suspicious communities based on transaction patterns"""
        if not community_map:
            return {}
        
        suspicious_scores = defaultdict(float)
        
        for comm_id in set(community_map.values()):
            members = [acc for acc, c in community_map.items() if c == comm_id]
            
            # Get transactions within community
            comm_txns = transactions[
                (transactions['source_account'].isin(members)) &
                (transactions['destination_account'].isin(members))
            ]
            
            if len(comm_txns) == 0:
                continue
            
            # Calculate suspicion factors
            # 1. High internal transaction ratio
            internal_ratio = len(comm_txns) / max(len(transactions), 1)
            
            # 2. Rapid cycling (money flowing back and forth)
            cycling_score = self._detect_cycling(members, transactions)
            
            # 3. Layering (many hops)
            layering_score = self._detect_layering(members, transactions)
            
            # 4. Concentration (few sources/destinations)
            concentration = len(set(comm_txns['source_account'])) / max(len(members), 1)
            
            suspicious_scores[comm_id] = (
                internal_ratio * 0.3 +
                cycling_score * 0.3 +
                layering_score * 0.2 +
                (1 - concentration) * 0.2
            )
        
        return dict(suspicious_scores)
    
    def trace_money_flow_paths(self, source_account: str, dest_account: str,
                              max_hops: int = 5) -> List[List[str]]:
        """Trace money flow paths between accounts"""
        if not HAS_NETWORKX or self.graph is None:
            return []
        
        try:
            paths = list(nx.all_simple_paths(
                self.graph, source_account, dest_account, cutoff=max_hops
            ))
            return paths
        except:
            return []
    
    def detect_circular_flows(self) -> List[List[str]]:
        """Detect circular money flows (cycles in graph)"""
        if not HAS_NETWORKX or self.graph is None:
            return []
        
        try:
            cycles = list(nx.simple_cycles(self.graph))
            # Filter to meaningful cycles (3+ nodes)
            return [c for c in cycles if len(c) >= 3]
        except:
            logger.warning("Cycle detection failed")
            return []
    
    def detect_layering_patterns(self) -> Dict[str, List[List[str]]]:
        """Detect layering patterns (money flowing through many intermediaries)"""
        if not HAS_NETWORKX or self.graph is None:
            return {}
        
        layering_patterns = defaultdict(list)
        
        # Find paths of length 4+ (indicating layering)
        for source in self.graph.nodes():
            for dest in self.graph.nodes():
                if source != dest:
                    try:
                        paths = list(nx.all_simple_paths(
                            self.graph, source, dest, cutoff=5
                        ))
                        for path in paths:
                            if len(path) >= 4:
                                layering_patterns[source].append(path)
                    except:
                        pass
        
        return dict(layering_patterns)
    
    def _detect_cycling(self, members: List[str], transactions: pd.DataFrame) -> float:
        """Detect rapid cycling within a group"""
        if len(members) < 2:
            return 0.0
        
        cycling_score = 0.0
        member_set = set(members)
        
        for source in members:
            for dest in members:
                if source != dest:
                    # Check if money flows both ways
                    forward = len(transactions[
                        (transactions['source_account'] == source) &
                        (transactions['destination_account'] == dest)
                    ])
                    backward = len(transactions[
                        (transactions['source_account'] == dest) &
                        (transactions['destination_account'] == source)
                    ])
                    
                    if forward > 0 and backward > 0:
                        cycling_score += 1.0
        
        max_pairs = len(members) * (len(members) - 1)
        return min(cycling_score / max(max_pairs, 1), 1.0)
    
    def _detect_layering(self, members: List[str], transactions: pd.DataFrame) -> float:
        """Detect layering patterns"""
        if not HAS_NETWORKX or self.graph is None:
            return 0.0
        
        # Count paths of length 3+ within the group
        long_paths = 0
        total_paths = 0
        
        for source in members:
            for dest in members:
                if source != dest:
                    try:
                        paths = list(nx.all_simple_paths(
                            self.graph, source, dest, cutoff=5
                        ))
                        for path in paths:
                            total_paths += 1
                            if len(path) >= 4:
                                long_paths += 1
                    except:
                        pass
        
        return long_paths / max(total_paths, 1)
    
    def get_account_network_features(self, account_id: str) -> Dict[str, float]:
        """Get network features for an account"""
        if not HAS_NETWORKX or self.graph is None:
            return {}
        
        features = {}
        
        # Centrality measures
        degree_cent = self.calculate_degree_centrality()
        between_cent = self.calculate_betweenness_centrality()
        close_cent = self.calculate_closeness_centrality()
        pagerank = self.calculate_pagerank()
        
        features['degree_centrality'] = degree_cent.get(account_id, 0.0)
        features['betweenness_centrality'] = between_cent.get(account_id, 0.0)
        features['closeness_centrality'] = close_cent.get(account_id, 0.0)
        features['pagerank_score'] = pagerank.get(account_id, 0.0)
        
        # Clustering
        clustering = self.calculate_clustering_coefficient()
        features['clustering_coefficient'] = clustering.get(account_id, 0.0)
        
        # Community info
        communities = self.detect_communities()
        if account_id in communities:
            comm_id = communities[account_id]
            members = [acc for acc, c in communities.items() if c == comm_id]
            features['community_size'] = len(members)
            
            # Community density
            if len(members) > 1:
                subgraph = self.graph.subgraph(members)
                possible_edges = len(members) * (len(members) - 1)
                features['community_density'] = subgraph.number_of_edges() / max(possible_edges, 1)
            else:
                features['community_density'] = 0.0
        else:
            features['community_size'] = 1
            features['community_density'] = 0.0
        
        # Counterparty features
        if account_id in self.graph:
            predecessors = list(self.graph.predecessors(account_id))
            successors = list(self.graph.successors(account_id))
            
            features['avg_counterparty_degree'] = np.mean([
                self.graph.degree(p) for p in predecessors + successors
            ]) if (predecessors or successors) else 0.0
            
            # Counterparty diversity
            all_counterparties = set(predecessors + successors)
            features['counterparty_diversity'] = len(all_counterparties) / max(
                self.graph.degree(account_id), 1
            )
            
            # Shared counterparty ratio
            if len(predecessors) > 0 and len(successors) > 0:
                shared = len(set(predecessors) & set(successors))
                features['shared_counterparty_ratio'] = shared / max(len(all_counterparties), 1)
            else:
                features['shared_counterparty_ratio'] = 0.0
        else:
            features['avg_counterparty_degree'] = 0.0
            features['counterparty_diversity'] = 0.0
            features['shared_counterparty_ratio'] = 0.0
        
        return features


class GraphAnomalyDetectorV2:
    """Detect anomalies using graph features"""
    
    def __init__(self):
        """Initialize detector"""
        self.isolation_forest = None
        self.lof = None
    
    def detect_with_isolation_forest(self, graph_features: pd.DataFrame,
                                     contamination: float = 0.1) -> np.ndarray:
        """Detect anomalies using Isolation Forest"""
        if not HAS_SKLEARN:
            logger.warning("Sklearn not available")
            return np.zeros(len(graph_features))
        
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = self.isolation_forest.fit_predict(graph_features)
        
        # Convert to anomaly scores (0-1)
        scores = -self.isolation_forest.score_samples(graph_features)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        return scores
    
    def detect_with_lof(self, graph_features: pd.DataFrame,
                       n_neighbors: int = 20) -> np.ndarray:
        """Detect anomalies using Local Outlier Factor"""
        if not HAS_SKLEARN:
            logger.warning("Sklearn not available")
            return np.zeros(len(graph_features))
        
        self.lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        self.lof.fit(graph_features)
        
        # Get LOF scores
        scores = -self.lof.negative_outlier_factor_
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        return scores


class GraphFeatureExtractorV2:
    """Extract features from transaction graph"""
    
    def __init__(self, graph: 'TransactionNetworkGraphV2'):
        """Initialize extractor"""
        self.graph = graph
        self.features = {}
    
    def compute_all_features(self) -> None:
        """Compute all graph features"""
        if not HAS_NETWORKX or self.graph.graph is None:
            logger.warning("Cannot compute features without graph")
            return
        
        logger.info("Computing graph features...")
        
        for account_id in self.graph.graph.nodes():
            self.features[account_id] = self.graph.get_account_network_features(account_id)
        
        logger.info(f"Computed features for {len(self.features)} accounts")
    
    def get_features_for_account(self, account_id: str) -> Dict[str, float]:
        """Get features for specific account"""
        return self.features.get(account_id, {})
