"""
Branch-Level Collusion Detection Module

Identifies suspicious patterns suggesting collusion between accounts at same branch:
- Circular money flows within branch
- Coordinated high-value transfers
- Unusual account clustering
- Shared counterparties
- Temporal coordination
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class BranchCollusionPattern:
    """Branch-level collusion pattern"""
    branch_id: str
    pattern_type: str  # 'circular_flow', 'coordinated_transfer', 'account_cluster', 'shared_counterparty'
    involved_accounts: List[str]
    involved_counterparties: List[str]
    total_amount: float
    transaction_count: int
    risk_score: float  # 0-1
    confidence: float
    supporting_evidence: List[str]


class BranchCollusionDetector:
    """Detect branch-level collusion patterns"""
    
    def __init__(self):
        """Initialize detector"""
        logger.info("BranchCollusionDetector initialized")
        self.branch_accounts = defaultdict(list)
        self.branch_transactions = defaultdict(list)
    
    def build_branch_graph(self, transactions: pd.DataFrame, accounts: pd.DataFrame) -> Dict[str, nx.DiGraph]:
        """
        Build transaction graphs for each branch.
        
        Args:
            transactions: Transaction dataframe
            accounts: Accounts dataframe with branch_id
            
        Returns:
            Dictionary of branch_id -> NetworkX DiGraph
        """
        branch_graphs = {}
        
        # Map accounts to branches
        if 'branch_id' in accounts.columns and 'account_id' in accounts.columns:
            for _, row in accounts.iterrows():
                branch_id = row['branch_id']
                account_id = row['account_id']
                self.branch_accounts[branch_id].append(account_id)
        
        # Build graph for each branch
        for branch_id, accounts_in_branch in self.branch_accounts.items():
            graph = nx.DiGraph()
            
            # Add transactions within branch
            if 'account_id' in transactions.columns and 'counterparty_account_id' in transactions.columns:
                branch_txns = transactions[
                    transactions['account_id'].isin(accounts_in_branch)
                ]
                
                for _, txn in branch_txns.iterrows():
                    source = txn['account_id']
                    dest = txn['counterparty_account_id']
                    amount = txn.get('amount', 1)
                    
                    if graph.has_edge(source, dest):
                        graph[source][dest]['weight'] += amount
                        graph[source][dest]['count'] += 1
                    else:
                        graph.add_edge(source, dest, weight=amount, count=1)
            
            branch_graphs[branch_id] = graph
        
        logger.info(f"Built graphs for {len(branch_graphs)} branches")
        return branch_graphs
    
    def detect_circular_flows(self, branch_graphs: Dict[str, nx.DiGraph]) -> List[BranchCollusionPattern]:
        """
        Detect circular money flows within branches.
        
        Args:
            branch_graphs: Dictionary of branch_id -> NetworkX DiGraph
            
        Returns:
            List of BranchCollusionPattern objects
        """
        patterns = []
        
        for branch_id, graph in branch_graphs.items():
            # Find all cycles
            try:
                cycles = list(nx.simple_cycles(graph))
            except:
                cycles = []
            
            # Analyze each cycle
            for cycle in cycles:
                if len(cycle) < 3:  # Only cycles with 3+ nodes
                    continue
                
                # Calculate total amount in cycle
                total_amount = 0
                for i in range(len(cycle)):
                    source = cycle[i]
                    dest = cycle[(i + 1) % len(cycle)]
                    
                    if graph.has_edge(source, dest):
                        total_amount += graph[source][dest]['weight']
                
                # Risk scoring
                risk_score = self._calculate_circular_flow_risk(cycle, graph, total_amount)
                
                if risk_score > 0.5:
                    patterns.append(BranchCollusionPattern(
                        branch_id=branch_id,
                        pattern_type='circular_flow',
                        involved_accounts=cycle,
                        involved_counterparties=[],
                        total_amount=total_amount,
                        transaction_count=len(cycle),
                        risk_score=risk_score,
                        confidence=0.8,
                        supporting_evidence=[
                            f"Circular flow detected: {' -> '.join(cycle)} -> {cycle[0]}",
                            f"Total amount: ${total_amount:,.0f}",
                            f"Cycle length: {len(cycle)} accounts"
                        ]
                    ))
        
        logger.info(f"Detected {len(patterns)} circular flow patterns")
        return patterns
    
    def detect_coordinated_transfers(self, transactions: pd.DataFrame, accounts: pd.DataFrame,
                                    time_window_hours: int = 24) -> List[BranchCollusionPattern]:
        """
        Detect coordinated high-value transfers within branches.
        
        Args:
            transactions: Transaction dataframe
            accounts: Accounts dataframe
            time_window_hours: Time window for coordination
            
        Returns:
            List of BranchCollusionPattern objects
        """
        patterns = []
        
        if 'transaction_date' not in transactions.columns or 'amount' not in transactions.columns:
            return patterns
        
        transactions = transactions.copy()
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        
        # Group by branch and time window
        for branch_id, accounts_in_branch in self.branch_accounts.items():
            branch_txns = transactions[
                transactions['account_id'].isin(accounts_in_branch)
            ].sort_values('transaction_date')
            
            if len(branch_txns) == 0:
                continue
            
            # Find coordinated transfers
            for i in range(len(branch_txns)):
                window_start = branch_txns.iloc[i]['transaction_date']
                window_end = window_start + timedelta(hours=time_window_hours)
                
                window_txns = branch_txns[
                    (branch_txns['transaction_date'] >= window_start) &
                    (branch_txns['transaction_date'] <= window_end)
                ]
                
                if len(window_txns) < 3:  # Need at least 3 coordinated transfers
                    continue
                
                # Check if transfers are to same counterparty
                counterparties = window_txns['counterparty_account_id'].unique()
                if len(counterparties) == 1:
                    total_amount = window_txns['amount'].sum()
                    
                    risk_score = self._calculate_coordination_risk(window_txns, total_amount)
                    
                    if risk_score > 0.6:
                        patterns.append(BranchCollusionPattern(
                            branch_id=branch_id,
                            pattern_type='coordinated_transfer',
                            involved_accounts=window_txns['account_id'].unique().tolist(),
                            involved_counterparties=counterparties.tolist(),
                            total_amount=total_amount,
                            transaction_count=len(window_txns),
                            risk_score=risk_score,
                            confidence=0.75,
                            supporting_evidence=[
                                f"Coordinated transfers to {counterparties[0]}",
                                f"Total amount: ${total_amount:,.0f}",
                                f"Number of accounts: {len(window_txns)}",
                                f"Time window: {time_window_hours} hours"
                            ]
                        ))
        
        logger.info(f"Detected {len(patterns)} coordinated transfer patterns")
        return patterns
    
    def detect_account_clusters(self, branch_graphs: Dict[str, nx.DiGraph],
                               min_cluster_size: int = 5) -> List[BranchCollusionPattern]:
        """
        Detect unusual account clustering within branches.
        
        Args:
            branch_graphs: Dictionary of branch_id -> NetworkX DiGraph
            min_cluster_size: Minimum cluster size
            
        Returns:
            List of BranchCollusionPattern objects
        """
        patterns = []
        
        for branch_id, graph in branch_graphs.items():
            if len(graph.nodes()) < min_cluster_size:
                continue
            
            # Find communities using greedy modularity
            try:
                undirected = graph.to_undirected()
                communities = list(nx.community.greedy_modularity_communities(undirected))
            except:
                communities = []
            
            # Analyze each community
            for community in communities:
                if len(community) < min_cluster_size:
                    continue
                
                # Calculate internal density
                internal_edges = 0
                total_weight = 0
                
                for source in community:
                    for dest in community:
                        if source != dest and graph.has_edge(source, dest):
                            internal_edges += 1
                            total_weight += graph[source][dest]['weight']
                
                max_edges = len(community) * (len(community) - 1)
                density = internal_edges / max_edges if max_edges > 0 else 0
                
                # High density = suspicious cluster
                if density > 0.3:
                    risk_score = min(density, 1.0)
                    
                    patterns.append(BranchCollusionPattern(
                        branch_id=branch_id,
                        pattern_type='account_cluster',
                        involved_accounts=list(community),
                        involved_counterparties=[],
                        total_amount=total_weight,
                        transaction_count=internal_edges,
                        risk_score=risk_score,
                        confidence=0.7,
                        supporting_evidence=[
                            f"Suspicious account cluster detected",
                            f"Cluster size: {len(community)} accounts",
                            f"Internal density: {density:.2%}",
                            f"Total internal transfers: ${total_weight:,.0f}"
                        ]
                    ))
        
        logger.info(f"Detected {len(patterns)} account cluster patterns")
        return patterns
    
    def detect_shared_counterparties(self, transactions: pd.DataFrame, accounts: pd.DataFrame,
                                    min_shared_count: int = 5) -> List[BranchCollusionPattern]:
        """
        Detect accounts sharing many counterparties (suspicious network).
        
        Args:
            transactions: Transaction dataframe
            accounts: Accounts dataframe
            min_shared_count: Minimum shared counterparties
            
        Returns:
            List of BranchCollusionPattern objects
        """
        patterns = []
        
        if 'counterparty_account_id' not in transactions.columns:
            return patterns
        
        # Build counterparty sets for each account
        account_counterparties = defaultdict(set)
        
        for _, txn in transactions.iterrows():
            account_id = txn['account_id']
            counterparty = txn['counterparty_account_id']
            account_counterparties[account_id].add(counterparty)
        
        # Find accounts with many shared counterparties
        for branch_id, accounts_in_branch in self.branch_accounts.items():
            branch_accounts_list = [a for a in accounts_in_branch if a in account_counterparties]
            
            if len(branch_accounts_list) < 2:
                continue
            
            # Compare counterparty sets
            for i in range(len(branch_accounts_list)):
                for j in range(i + 1, len(branch_accounts_list)):
                    account1 = branch_accounts_list[i]
                    account2 = branch_accounts_list[j]
                    
                    shared = account_counterparties[account1] & account_counterparties[account2]
                    
                    if len(shared) >= min_shared_count:
                        risk_score = min(len(shared) / 20, 1.0)
                        
                        patterns.append(BranchCollusionPattern(
                            branch_id=branch_id,
                            pattern_type='shared_counterparty',
                            involved_accounts=[account1, account2],
                            involved_counterparties=list(shared),
                            total_amount=0,
                            transaction_count=len(shared),
                            risk_score=risk_score,
                            confidence=0.65,
                            supporting_evidence=[
                                f"Accounts {account1} and {account2} share {len(shared)} counterparties",
                                f"Shared counterparties: {len(shared)}"
                            ]
                        ))
        
        logger.info(f"Detected {len(patterns)} shared counterparty patterns")
        return patterns
    
    def _calculate_circular_flow_risk(self, cycle: List[str], graph: nx.DiGraph, total_amount: float) -> float:
        """Calculate risk score for circular flow"""
        # Base risk from cycle length
        cycle_risk = min(len(cycle) / 10, 0.5)
        
        # Risk from amount
        amount_risk = min(total_amount / 1000000, 0.5)
        
        return cycle_risk + amount_risk
    
    def _calculate_coordination_risk(self, transactions: pd.DataFrame, total_amount: float) -> float:
        """Calculate risk score for coordinated transfers"""
        # Base risk from number of accounts
        account_risk = min(len(transactions) / 10, 0.5)
        
        # Risk from amount
        amount_risk = min(total_amount / 1000000, 0.5)
        
        return account_risk + amount_risk
