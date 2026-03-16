import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not installed")
class TestTransactionNetworkGraph:
    """Test transaction network graph"""
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data"""
        return pd.DataFrame({
            'source_account': ['ACC001', 'ACC001', 'ACC002', 'ACC002', 'ACC003'],
            'destination_account': ['ACC002', 'ACC003', 'ACC001', 'ACC003', 'ACC001'],
            'amount': [1000, 2000, 1500, 3000, 500],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='D')
        })
    
    def test_graph_initialization(self):
        """Test graph initialization"""
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2
        
        graph = TransactionNetworkGraphV2()
        assert graph.graph is not None
        assert isinstance(graph.graph, nx.DiGraph)
    
    def test_build_graph(self, sample_transactions):
        """Test building graph from transactions"""
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2
        
        graph = TransactionNetworkGraphV2()
        graph.build_graph(sample_transactions)
        
        assert graph.graph.number_of_nodes() > 0
        assert graph.graph.number_of_edges() > 0
    
    def test_degree_centrality(self, sample_transactions):
        """Test degree centrality calculation"""
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2
        
        graph = TransactionNetworkGraphV2()
        graph.build_graph(sample_transactions)
        
        centrality = graph.calculate_degree_centrality()
        
        assert isinstance(centrality, dict)
        assert len(centrality) > 0
    
    def test_betweenness_centrality(self, sample_transactions):
        """Test betweenness centrality calculation"""
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2
        
        graph = TransactionNetworkGraphV2()
        graph.build_graph(sample_transactions)
        
        centrality = graph.calculate_betweenness_centrality()
        
        assert isinstance(centrality, dict)
        assert len(centrality) > 0
    
    def test_closeness_centrality(self, sample_transactions):
        """Test closeness centrality calculation"""
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2
        
        graph = TransactionNetworkGraphV2()
        graph.build_graph(sample_transactions)
        
        centrality = graph.calculate_closeness_centrality()
        
        assert isinstance(centrality, dict)
        assert len(centrality) > 0
    
    def test_clustering_coefficient(self, sample_transactions):
        """Test clustering coefficient calculation"""
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2
        
        graph = TransactionNetworkGraphV2()
        graph.build_graph(sample_transactions)
        
        clustering = graph.calculate_clustering_coefficient()
        
        assert isinstance(clustering, dict)
        assert len(clustering) > 0
    
    def test_pagerank(self, sample_transactions):
        """Test PageRank calculation"""
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2
        
        graph = TransactionNetworkGraphV2()
        graph.build_graph(sample_transactions)
        
        pagerank = graph.calculate_pagerank()
        
        assert isinstance(pagerank, dict)
        assert len(pagerank) > 0
        assert all(v > 0 for v in pagerank.values())
    
    def test_community_detection(self, sample_transactions):
        """Test community detection"""
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2
        
        graph = TransactionNetworkGraphV2()
        graph.build_graph(sample_transactions)
        
        communities = graph.detect_communities()
        
        assert isinstance(communities, dict)
        assert len(communities) > 0
    
    def test_circular_flow_detection(self, sample_transactions):
        """Test circular flow detection"""
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2
        
        graph = TransactionNetworkGraphV2()
        graph.build_graph(sample_transactions)
        
        cycles = graph.detect_circular_flows()
        
        assert isinstance(cycles, list)
    
    def test_layering_pattern_detection(self, sample_transactions):
        """Test layering pattern detection"""
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2
        
        graph = TransactionNetworkGraphV2()
        graph.build_graph(sample_transactions)
        
        layering = graph.detect_layering_patterns()
        
        assert isinstance(layering, dict)
    
    def test_money_flow_paths(self, sample_transactions):
        """Test money flow path tracing"""
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2
        
        graph = TransactionNetworkGraphV2()
        graph.build_graph(sample_transactions)
        
        paths = graph.trace_money_flow_paths('ACC001', 'ACC003', max_hops=3)
        
        assert isinstance(paths, list)
    
    def test_suspicious_communities(self, sample_transactions):
        """Test suspicious community identification"""
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2
        
        graph = TransactionNetworkGraphV2()
        graph.build_graph(sample_transactions)
        
        communities = graph.detect_communities()
        suspicious = graph.identify_suspicious_communities(communities, sample_transactions)
        
        assert isinstance(suspicious, dict)


@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not installed")
class TestGraphFeatureExtractor:
    """Test graph feature extraction"""
    
    @pytest.fixture
    def sample_graph(self):
        """Create sample graph"""
        from src_enhanced.graph_analysis_v2 import TransactionNetworkGraphV2
        
        transactions = pd.DataFrame({
            'source_account': ['ACC001', 'ACC001', 'ACC002', 'ACC002', 'ACC003'],
            'destination_account': ['ACC002', 'ACC003', 'ACC001', 'ACC003', 'ACC001'],
            'amount': [1000, 2000, 1500, 3000, 500],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='D')
        })
        
        graph = TransactionNetworkGraphV2()
        graph.build_graph(transactions)
        return graph
    
    def test_feature_extractor_init(self, sample_graph):
        """Test feature extractor initialization"""
        from src_enhanced.graph_analysis_v2 import GraphFeatureExtractorV2
        
        extractor = GraphFeatureExtractorV2(sample_graph)
        assert extractor.graph is not None
        assert extractor.features == {}
    
    def test_compute_all_features(self, sample_graph):
        """Test computing all features"""
        from src_enhanced.graph_analysis_v2 import GraphFeatureExtractorV2
        
        extractor = GraphFeatureExtractorV2(sample_graph)
        extractor.compute_all_features()
        
        assert len(extractor.features) > 0
    
    def test_get_features_for_account(self, sample_graph):
        """Test getting features for specific account"""
        from src_enhanced.graph_analysis_v2 import GraphFeatureExtractorV2
        
        extractor = GraphFeatureExtractorV2(sample_graph)
        extractor.compute_all_features()
        
        features = extractor.get_features_for_account('ACC001')
        
        assert isinstance(features, dict)
        assert 'degree_centrality' in features
        assert 'betweenness_centrality' in features
        assert 'pagerank_score' in features
    
    def test_network_features_structure(self, sample_graph):
        """Test network features structure"""
        from src_enhanced.graph_analysis_v2 import GraphFeatureExtractorV2
        
        extractor = GraphFeatureExtractorV2(sample_graph)
        extractor.compute_all_features()
        
        features = extractor.get_features_for_account('ACC001')
        
        required_features = [
            'degree_centrality',
            'betweenness_centrality',
            'closeness_centrality',
            'pagerank_score',
            'clustering_coefficient',
            'community_size',
            'community_density',
            'avg_counterparty_degree',
            'counterparty_diversity',
            'shared_counterparty_ratio'
        ]
        
        for feature in required_features:
            assert feature in features


@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not installed")
class TestGraphAnomalyDetector:
    """Test graph anomaly detection"""
    
    @pytest.fixture
    def sample_graph_features(self):
        """Create sample graph features"""
        return pd.DataFrame({
            'degree_centrality': np.random.uniform(0, 1, 100),
            'betweenness_centrality': np.random.uniform(0, 1, 100),
            'pagerank_score': np.random.uniform(0, 1, 100),
            'clustering_coefficient': np.random.uniform(0, 1, 100),
        })
    
    def test_isolation_forest_detection(self, sample_graph_features):
        """Test Isolation Forest anomaly detection"""
        from src_enhanced.graph_analysis_v2 import GraphAnomalyDetectorV2
        
        detector = GraphAnomalyDetectorV2()
        scores = detector.detect_with_isolation_forest(sample_graph_features)
        
        assert len(scores) == len(sample_graph_features)
        assert np.all((scores >= 0) & (scores <= 1))
    
    def test_lof_detection(self, sample_graph_features):
        """Test Local Outlier Factor anomaly detection"""
        from src_enhanced.graph_analysis_v2 import GraphAnomalyDetectorV2
        
        detector = GraphAnomalyDetectorV2()
        scores = detector.detect_with_lof(sample_graph_features)
        
        assert len(scores) == len(sample_graph_features)
        assert np.all((scores >= 0) & (scores <= 1))
