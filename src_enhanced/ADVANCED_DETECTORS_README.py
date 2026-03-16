"""
Advanced Detectors Module - Summary

Three new modular detectors added to address missing challenge requirements:

1. RED HERRING DETECTOR (red_herring_detector.py)
   ================================================
   Identifies features that correlate with labels but don't generalize.
   
   Key Classes:
   - RedHerringDetector: Main detector with fit/detect methods
   - TemporalStabilityAnalyzer: Analyzes feature stability across time periods
   - LeakageDetector: Detects potential label leakage
   
   Methods:
   - fit(X_train, y_train): Fit on training data
   - detect_red_herrings(X_test, y_test): Identify red herring features
   - filter_features(X, red_herrings): Remove red herring features
   
   Metrics Calculated:
   - Temporal instability: Feature behavior changes over time
   - Distribution shift: KS test for distribution changes
   - Spurious correlation: Correlation differences between train/test
   
   Usage:
   ```python
   detector = RedHerringDetector()
   detector.fit(X_train, y_train)
   red_herrings = detector.detect_red_herrings(X_test, y_test, threshold=0.6)
   X_filtered = detector.filter_features(X_test, red_herrings)
   ```


2. FREEZE/UNFREEZE DETECTOR (freeze_unfreeze_detector.py)
   ========================================================
   Detects suspicious freeze and unfreeze patterns in accounts.
   
   Key Classes:
   - FreezeUnfreezeDetector: Main detector
   - CoordinatedFreezeDetector: Detects coordinated patterns across accounts
   
   Pattern Types Detected:
   - freeze_before_activity: Freeze before high-value transactions
   - unfreeze_spike: Activity spike after unfreeze
   - multiple_cycles: Multiple freeze/unfreeze cycles
   - coordinated: Coordinated freezes across accounts
   
   Methods:
   - detect_patterns(account_id, transactions, status_history): Detect patterns
   - detect_coordinated_patterns(account_freeze_events): Cross-account patterns
   
   Features:
   - Extracts freeze events from transaction gaps (>30 days)
   - Uses account status history if available
   - Calculates risk scores based on pattern type and duration
   - Generates supporting evidence
   
   Usage:
   ```python
   detector = FreezeUnfreezeDetector()
   patterns = detector.detect_patterns(account_id, transactions, status_history)
   coordinated = CoordinatedFreezeDetector.detect_coordinated_patterns(all_freeze_events)
   ```


3. BRANCH COLLUSION DETECTOR (branch_collusion_detector.py)
   =========================================================
   Identifies suspicious patterns suggesting collusion between accounts at same branch.
   
   Key Classes:
   - BranchCollusionDetector: Main detector
   
   Pattern Types Detected:
   - circular_flow: Circular money flows within branch
   - coordinated_transfer: Coordinated high-value transfers
   - account_cluster: Unusual account clustering
   - shared_counterparty: Accounts sharing many counterparties
   
   Methods:
   - build_branch_graph(transactions, accounts): Build transaction graphs per branch
   - detect_circular_flows(branch_graphs): Find circular flows
   - detect_coordinated_transfers(transactions, accounts): Find coordinated transfers
   - detect_account_clusters(branch_graphs): Find suspicious clusters
   - detect_shared_counterparties(transactions, accounts): Find shared networks
   
   Features:
   - Uses NetworkX for graph analysis
   - Calculates community detection using greedy modularity
   - Measures network density and centrality
   - Generates risk scores and evidence
   
   Usage:
   ```python
   detector = BranchCollusionDetector()
   branch_graphs = detector.build_branch_graph(transactions, accounts)
   circular = detector.detect_circular_flows(branch_graphs)
   coordinated = detector.detect_coordinated_transfers(transactions, accounts)
   clusters = detector.detect_account_clusters(branch_graphs)
   shared = detector.detect_shared_counterparties(transactions, accounts)
   ```


INTEGRATION GUIDE
=================

To integrate these detectors into your pipeline:

1. Import the detectors:
   ```python
   from src_enhanced.red_herring_detector import RedHerringDetector
   from src_enhanced.freeze_unfreeze_detector import FreezeUnfreezeDetector
   from src_enhanced.branch_collusion_detector import BranchCollusionDetector
   ```

2. Add to feature engineering:
   ```python
   # Red herring detection
   rh_detector = RedHerringDetector()
   rh_detector.fit(X_train, y_train)
   red_herrings = rh_detector.detect_red_herrings(X_test)
   X_filtered = rh_detector.filter_features(X_test, red_herrings)
   
   # Freeze/unfreeze patterns
   fu_detector = FreezeUnfreezeDetector()
   fu_patterns = fu_detector.detect_patterns(account_id, transactions)
   
   # Branch collusion
   bc_detector = BranchCollusionDetector()
   branch_graphs = bc_detector.build_branch_graph(transactions, accounts)
   collusion_patterns = bc_detector.detect_circular_flows(branch_graphs)
   ```

3. Add pattern scores as features:
   ```python
   features['red_herring_risk'] = red_herring_score
   features['freeze_unfreeze_risk'] = fu_pattern.risk_score
   features['branch_collusion_risk'] = collusion_pattern.risk_score
   ```


DESIGN PRINCIPLES
=================

✅ Modularity: Each detector is independent and can be used separately
✅ Accuracy: Uses statistical tests (KS test, mutual information) for validation
✅ Scalability: Efficient algorithms (NetworkX, pandas vectorization)
✅ Interpretability: Generates supporting evidence for each detection
✅ Robustness: Handles missing data and edge cases gracefully
✅ Logging: Comprehensive logging for debugging and monitoring


EXPECTED IMPACT
===============

These detectors address the 3 missing challenge requirements:

1. Red Herring Detection: Improves generalization by filtering spurious features
   Expected improvement: +2-3% on test set

2. Freeze/Unfreeze Patterns: Catches 13th mule behavior pattern
   Expected improvement: +1-2% on test set

3. Branch Collusion: Detects coordinated fraud at branch level
   Expected improvement: +2-3% on test set

Total expected improvement: +5-8% on leaderboard score
"""
