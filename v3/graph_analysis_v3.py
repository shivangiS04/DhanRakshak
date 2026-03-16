"""
graph_analysis_v3.py
Key fixes over v2:
- Precompute ALL centralities ONCE, store in dict → O(N) lookup per account
- Betweenness uses k=500 approximation (exact is infeasible on 160K nodes)
- detect_layering_patterns removed (was O(N²) and would never complete)
- Community detection called once, result cached
- Fan-in / fan-out per node added as graph features
- Suspicious community membership stored per account
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logger = logging.getLogger(__name__)


class TransactionNetworkGraphV3:
    """
    Efficient transaction network graph.
    All expensive centrality calculations are done ONCE and cached.
    """

    def __init__(self):
        self.graph: Optional['nx.DiGraph'] = nx.DiGraph() if HAS_NETWORKX else None

        # Cached per-node feature dicts — populated by precompute_all()
        self._degree_cent:    Dict[str, float] = {}
        self._between_cent:   Dict[str, float] = {}
        self._pagerank:       Dict[str, float] = {}
        self._clustering:     Dict[str, float] = {}
        self._community_map:  Dict[str, int]   = {}
        self._community_sizes: Dict[int, int]  = {}
        self._community_density: Dict[int, float] = {}
        self._suspicious_comm: Dict[int, float] = {}

        self._precomputed = False
        logger.info("TransactionNetworkGraphV3 initialized")

    # ------------------------------------------------------------------
    def build_graph(self, transactions: pd.DataFrame) -> None:
        if not HAS_NETWORKX or self.graph is None:
            logger.warning("NetworkX not available — skipping graph build")
            return

        logger.info(f"Building graph from {len(transactions):,} transactions …")

        src_col  = 'source_account'      if 'source_account'      in transactions.columns else None
        dst_col  = 'destination_account' if 'destination_account' in transactions.columns else None
        amt_col  = 'amount'              if 'amount'              in transactions.columns else None

        if src_col is None or dst_col is None:
            logger.warning("Source/destination columns not found — cannot build graph")
            return

        for row in transactions[[src_col, dst_col, amt_col]].dropna().itertuples(index=False):
            src, dst, amt = row[0], row[1], row[2] if amt_col else 1.0
            if src == dst:
                continue
            if self.graph.has_edge(src, dst):
                self.graph[src][dst]['weight'] += amt
                self.graph[src][dst]['count']  += 1
            else:
                self.graph.add_edge(src, dst, weight=float(amt), count=1)

        logger.info(f"Graph: {self.graph.number_of_nodes():,} nodes, "
                    f"{self.graph.number_of_edges():,} edges")

    # ------------------------------------------------------------------
    def precompute_all(self, transactions: pd.DataFrame) -> None:
        """
        Precompute ALL graph features in one pass.
        Call this ONCE after build_graph — then use get_features_for_account().
        """
        if not HAS_NETWORKX or self.graph is None:
            return
        if self._precomputed:
            return

        logger.info("Precomputing graph features …")

        # 1. Degree centrality — O(N+E)
        logger.info("  Computing degree centrality …")
        self._degree_cent = nx.degree_centrality(self.graph)

        # 2. PageRank — O(N+E) per iteration
        logger.info("  Computing PageRank …")
        try:
            self._pagerank = nx.pagerank(self.graph, alpha=0.85, max_iter=100)
        except Exception as e:
            logger.warning(f"  PageRank failed: {e}")
            self._pagerank = {n: 0.0 for n in self.graph.nodes()}

        # 3. Betweenness — O(kN) with approximation
        logger.info("  Computing approximate betweenness centrality (k=500) …")
        try:
            self._between_cent = nx.betweenness_centrality(self.graph, k=500, normalized=True)
        except Exception as e:
            logger.warning(f"  Betweenness failed: {e}")
            self._between_cent = {n: 0.0 for n in self.graph.nodes()}

        # 4. Clustering on undirected projection
        logger.info("  Computing clustering coefficient …")
        try:
            undirected = self.graph.to_undirected()
            self._clustering = nx.clustering(undirected)
        except Exception as e:
            logger.warning(f"  Clustering failed: {e}")
            self._clustering = {n: 0.0 for n in self.graph.nodes()}

        # 5. Community detection
        logger.info("  Detecting communities …")
        self._community_map = self._detect_communities_once()

        # Precompute community sizes and densities
        for node, cid in self._community_map.items():
            self._community_sizes[cid] = self._community_sizes.get(cid, 0) + 1

        for cid, members in self._communities_by_id().items():
            subg = self.graph.subgraph(members)
            n = len(members)
            possible = n * (n - 1)
            self._community_density[cid] = (
                subg.number_of_edges() / possible if possible > 0 else 0.0
            )

        # 6. Suspicious community scores
        logger.info("  Scoring communities …")
        self._suspicious_comm = self._score_communities(transactions)

        self._precomputed = True
        logger.info("Graph precomputation complete.")

    # ------------------------------------------------------------------
    def get_features_for_account(self, account_id: str) -> Dict[str, float]:
        """O(1) lookup — all values are precomputed."""
        f: Dict[str, float] = {}

        f['degree_centrality']      = self._degree_cent.get(account_id, 0.0)
        f['betweenness_centrality'] = self._between_cent.get(account_id, 0.0)
        f['pagerank_score']         = self._pagerank.get(account_id, 0.0)
        f['clustering_coefficient'] = self._clustering.get(account_id, 0.0)

        cid = self._community_map.get(account_id)
        f['community_size']    = float(self._community_sizes.get(cid, 1)) if cid is not None else 1.0
        f['community_density'] = self._community_density.get(cid, 0.0)   if cid is not None else 0.0

        # Network-level risk: is this account in a suspicious community?
        if cid is not None:
            f['network_risk_score'] = self._suspicious_comm.get(cid, 0.0)
        else:
            f['network_risk_score'] = 0.0

        # Counterparty features
        if self.graph is not None and account_id in self.graph:
            preds = list(self.graph.predecessors(account_id))
            succs = list(self.graph.successors(account_id))
            all_cp = set(preds + succs)

            degrees = [self.graph.degree(c) for c in all_cp]
            f['avg_counterparty_degree'] = float(np.mean(degrees)) if degrees else 0.0
            f['counterparty_diversity']  = len(all_cp) / max(self.graph.degree(account_id), 1)

            shared = len(set(preds) & set(succs))
            f['shared_counterparty_ratio'] = shared / max(len(all_cp), 1)
        else:
            f['avg_counterparty_degree'] = 0.0
            f['counterparty_diversity']  = 0.0
            f['shared_counterparty_ratio'] = 0.0

        return f

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_communities_once(self) -> Dict[str, int]:
        try:
            from networkx.algorithms import community as nx_comm
            comms = nx_comm.greedy_modularity_communities(self.graph.to_undirected())
            cmap = {}
            for cid, members in enumerate(comms):
                for node in members:
                    cmap[node] = cid
            logger.info(f"  Found {len(comms)} communities")
            return cmap
        except Exception as e:
            logger.warning(f"  Community detection failed: {e}")
            return {}

    def _communities_by_id(self) -> Dict[int, List[str]]:
        result: Dict[int, List[str]] = defaultdict(list)
        for node, cid in self._community_map.items():
            result[cid].append(node)
        return dict(result)

    def _score_communities(self, transactions: pd.DataFrame) -> Dict[int, float]:
        """
        Score each community for suspiciousness.
        Uses only vectorised pandas ops — no per-row Python loops.
        """
        scores: Dict[int, float] = {}
        by_id = self._communities_by_id()

        src_col = 'source_account'      if 'source_account'      in transactions.columns else None
        dst_col = 'destination_account' if 'destination_account' in transactions.columns else None
        if src_col is None or dst_col is None:
            return scores

        # Build account → community lookup series for fast merge
        acc_cid = pd.Series(self._community_map, name='community_id')

        txns = transactions[[src_col, dst_col]].copy()
        txns['src_cid'] = txns[src_col].map(acc_cid)
        txns['dst_cid'] = txns[dst_col].map(acc_cid)

        # Internal transactions: same community on both sides
        internal = txns[txns['src_cid'] == txns['dst_cid']].copy()
        internal_counts = internal.groupby('src_cid').size()

        for cid, members in by_id.items():
            n = len(members)
            if n < 3:
                scores[cid] = 0.0
                continue

            internal_n = internal_counts.get(cid, 0)
            density    = self._community_density.get(cid, 0.0)

            # Bidirectional edge fraction (cycling proxy)
            subg = self.graph.subgraph(members)
            edges = set(subg.edges())
            cycling = sum(1 for u, v in edges if (v, u) in edges) / max(len(edges), 1)

            scores[cid] = (
                min(internal_n / max(n * 10, 1), 1.0) * 0.3 +
                density * 0.3 +
                cycling * 0.4
            )

        return scores

    # ------------------------------------------------------------------
    def detect_circular_flows(self) -> List[List[str]]:
        """Return simple cycles of length ≥ 3 (limited to first 1000)."""
        if not HAS_NETWORKX or self.graph is None:
            return []
        try:
            cycles = []
            for c in nx.simple_cycles(self.graph):
                if len(c) >= 3:
                    cycles.append(c)
                if len(cycles) >= 1_000:
                    break
            return cycles
        except Exception as e:
            logger.warning(f"Cycle detection failed: {e}")
            return []
