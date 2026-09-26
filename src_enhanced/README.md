# src_enhanced — v2 Detectors

Additional modules developed during the competition. These are **not imported by the final scored
pipeline** (`v3/generate_submission_v21_fast_advanced.py`), but are complete, tested implementations
that extend the core feature set.

## Modules

### `red_herring_detector.py`
Identifies features that correlate with training labels but may not generalise.

- `RedHerringDetector` — fit/detect/filter interface
- `TemporalStabilityAnalyzer` — KS test for feature drift across time periods
- `LeakageDetector` — detects correlation patterns suggesting label leakage

### `freeze_unfreeze_detector.py`
Detects suspicious account-freeze/unfreeze patterns.

- `FreezeUnfreezeDetector` — identifies gaps >30 days as freeze events, scores by pattern type
- `CoordinatedFreezeDetector` — looks for synchronised freeze events across multiple accounts
- Pattern types: `freeze_before_activity`, `unfreeze_spike`, `multiple_cycles`, `coordinated`

### `branch_collusion_detector.py`
Identifies potential collusion between accounts at the same branch using graph analysis.

- `BranchCollusionDetector` — NetworkX-based graph per branch
- Detects: circular flows, coordinated transfers, dense account clusters, shared counterparties

### `feature_engineering_v2.py`
Extended feature set (50+ features) building on the base 13 in `src/transaction_data_v1.py`.
Adds coefficient of variation, skewness/kurtosis on amounts, dormancy periods, and activity spike magnitude.
Implements Herfindahl-index concentration (vs max-share in the scored module).

### `ensemble_models_v2.py`
Enhanced ensemble with additional model types and cross-validation utilities.

### `graph_analysis_v2.py`
PageRank, betweenness centrality, clustering coefficient, and community cycling score
computed on the transaction network. Dependency-optional (wraps `import networkx`).

### `pipeline_v2.py`
Orchestrator that wires up the enhanced feature set and detectors end-to-end.

## Usage

```python
from src_enhanced.branch_collusion_detector import BranchCollusionDetector
from src_enhanced.freeze_unfreeze_detector import FreezeUnfreezeDetector
from src_enhanced.red_herring_detector import RedHerringDetector

bc = BranchCollusionDetector()
branch_graphs = bc.build_branch_graph(transactions, accounts)
circular = bc.detect_circular_flows(branch_graphs)
```
