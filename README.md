# DhanRakshak — AML Mule Account Detection

[![CI](https://github.com/shivangiS04/DhanRakshak/actions/workflows/ci.yml/badge.svg)](https://github.com/shivangiS04/DhanRakshak/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Competition submission for the [RBIH-NFPC Phase 2 challenge](https://www.kaggle.com/competitions/rbih-nfpc-phase-2)
(Reserve Bank Innovation Hub / National Financial Protection Council).
The task: identify money-mule accounts in Indian retail banking and predict the time window of suspicious activity.

Dataset: ~159 K accounts, ~400 M transactions across ~88 batches of Parquet files.

---

## Results

These are the competition leaderboard scores from the original submission.
A subsequent reproducibility run (on Kaggle, 5-fold CV, held-out fold) gave a **validation AUC of 0.9203** —
see [`verified_metrics.json`](verified_metrics.json) for the exact run record.
The gap between 0.9203 and the leaderboard scores is discussed in the notes at the bottom.

| Metric | Public leaderboard | Private leaderboard |
|--------|-------------------:|--------------------:|
| AUC-ROC | 0.9773 | **0.9671** |
| F1 Score | 0.5879 | 0.5090 |
| Temporal IoU | 0.6850 | **0.6181** |
| Adversarial robustness avg (RH 1–6) | — | **0.9456** |

---

## What the Scored Pipeline Does

The final submission runs in two stages. Both stages are standalone scripts that read data directly
and do not import other modules from this repo.

**Stage 1 — `v3/generate_submission_v21_fast_advanced.py`**

1. Reads a pre-extracted feature CSV (`output/mega_transaction_features.csv`, not in repo — built by `src/transaction_data_v1.py`)
2. Trains four sklearn/XGBoost models on labelled accounts
3. Combines predictions using AUC-weighted averaging
4. Writes an initial submission with placeholder temporal windows
   (calendar-date ranges assigned by probability score, *not* real per-account timestamps)

**Stage 2 — `wide_window_attempt.py`**

Reads Stage 1's output and the raw transaction files, then replaces the placeholder windows
with the actual first–last transaction span for each predicted mule account.
This is what improved Temporal IoU from ~0.18 to ~0.62.

**Optional — `f1_calibration_postprocessor.py`**

Post-processing only; uses isotonic regression to recalibrate probabilities.
Not part of the final `submission_v21_wide_windows.csv`.

---

## Feature Engineering

Features are built by `src/transaction_data_v1.py`, which streams the raw Parquet batches
and produces one row per account.

| Feature | What it actually computes |
|---------|--------------------------|
| `structuring_ratio` | Fraction of transactions in bands just below round reporting thresholds (₹47.5k–50k, ₹95k–1L, ₹1.9L–2L, ₹4.75L–5L) |
| `round_ratio` | Fraction of transactions at commonly round amounts (₹990–1010, ₹4.95k–5.05k, ₹9.9k–10.1k, etc.) |
| `credit_concentration` | max(single-counterparty credit count) / total credits — largest single inflow source share |
| `debit_concentration` | max(single-counterparty debit count) / total debits — largest single outflow destination share |
| `counterparty_diversity` | unique counterparties / total transactions |
| `transaction_flow_anomaly` | \|total_credit − total_debit\| / (total_credit + total_debit) |
| `transaction_span_days` | days between first and last transaction |
| `total_transactions`, `total_credit_amount`, `total_debit_amount` | raw volume counts |
| `avg_credit_amount`, `avg_debit_amount` | per-transaction averages |
| `median_time_between_txns` | median inter-transaction gap in minutes |

The scored pipeline then computes ~30 interaction and log-transform derivatives from these at runtime
(`create_derived_features` in Stage 1). The exact column count depends on what's present in the CSV.

### What is in the repo but was not used in the final submission

The following modules are complete and tested but are **not imported by the two scored scripts**:

- `src/graph_analysis.py`, `src_enhanced/graph_analysis_v2.py` — NetworkX-based features (degree centrality, betweenness, community detection). Tested in `tests/test_enhanced_graph.py`.
- `src/feature_engineering.py`, `src_enhanced/feature_engineering_v2.py`, `v3/feature_engineering_v3.py` — alternate feature implementations, some with Herfindahl concentration and ₹9k–₹10k structuring band.
- `src/pipeline.py`, `src_enhanced/pipeline_v2.py` — orchestrator classes that wire up the above modules.
- Everything in `src_enhanced/` (`branch_collusion_detector.py`, `freeze_unfreeze_detector.py`, `red_herring_detector.py`) — built during development, not integrated into the final scored run.

These represent iterative development work. Some of the ideas from them informed the final features even if the modules themselves weren't imported.

---

## Ensemble

| Model | Val AUC (reproduced) | Key parameters |
|-------|---------------------:|----------------|
| XGBoost | 0.9118 | n_est=500, depth=8, lr=0.04, scale_pos_weight=60 |
| ExtraTrees | 0.8994 | n_est=400, depth=16, class_weight=balanced |
| GradientBoosting | 0.9185 | n_est=300, depth=6, lr=0.05, subsample=0.75 |
| Neural Network (MLP) | 0.9110 | 512→256→128, ReLU, Adam, early stopping |

Combined: `p = Σ (AUCᵢ / ΣAUC) × pᵢ`.
Class imbalance (≈2.8% mule rate) handled via `scale_pos_weight=60` (XGBoost),
`class_weight='balanced'` (tree models), stratified 80/20 split.

---

## What Worked / What Didn't

| Approach | Outcome |
|----------|---------|
| ✅ Stage 2 wide-window replacement | Temporal IoU 0.183 → 0.618 |
| ✅ 4-model ensemble over single XGBoost | Validation AUC +0.015 |
| ✅ `structuring_ratio` (round-boundary detection) | Highest single feature importance |
| ✅ Isotonic calibration (optional post-process) | F1 improvement without AUC loss |
| ❌ SMOTE oversampling | Synthetic minority samples didn't match real patterns |
| ❌ Platt scaling | Compressed probabilities into a narrow range, hurt F1 |
| ❌ Burst detection (rolling z-score) for windows | Consistently found the wrong peak period |
| ❌ CatBoost / LightGBM | Comparable or slightly worse AUC, longer training |

---

## Adversarial Robustness

The private test set contains 7 categories of accounts designed to look like mules but aren't
(or vice versa). Scores from the competition:

| Category | Pattern | Score |
|----------|---------|------:|
| RH_1 | High-volume businesses | 0.9904 |
| RH_2 | Seasonal spending spikes | 0.9510 |
| RH_3 | Large one-off transfers | 0.9040 |
| RH_4 | Dormant → reactivated accounts | 0.9789 |
| RH_5 | New account high activity | 0.9522 |
| RH_6 | Branch-level correlated patterns | 0.9968 |
| RH_7 | Post mobile-update spike (account takeover) | 0.1429 ⚠️ |

RH_7 failed because it requires device-fingerprint or IP-geolocation signals not present in the dataset.

---

## Repository Layout

```
DhanRakshak/
├── v3/
│   ├── generate_submission_v21_fast_advanced.py  ← Stage 1 (scored)
│   ├── generate_submission_v25_aggressive_threshold.py
│   ├── feature_engineering_v3.py
│   ├── ensemble_models_v3.py
│   └── graph_analysis_v3.py
├── wide_window_attempt.py                        ← Stage 2 (scored)
├── f1_calibration_postprocessor.py               ← optional post-process
├── src/                                          ← feature extraction + utilities
│   ├── transaction_data_v1.py                    ← builds the feature CSV Stage 1 reads
│   ├── feature_engineering.py                    ← alternate feature implementation
│   ├── graph_analysis.py                         ← NetworkX graph features (not used in final)
│   ├── ensemble_models.py
│   ├── pipeline.py
│   └── ...
├── src_enhanced/                                 ← v2 detectors (not used in final)
│   ├── branch_collusion_detector.py
│   ├── freeze_unfreeze_detector.py
│   ├── red_herring_detector.py
│   └── ...
├── experiments/                                  ← earlier iteration scripts
├── tests/                                        ← pytest unit tests
├── verified_metrics.json                         ← Kaggle reproduction run metrics
├── pyproject.toml
└── requirements.txt
```

---

## How to Run

```bash
pip install -e ".[dev]"

# Build the feature CSV from raw data (requires the Kaggle archive)
export DHANRAKSHAK_DATA_ROOT=/path/to/rbih-nfpc-phase-2
python src/transaction_data_v1.py

# Stage 1 — train ensemble and generate initial submission
python v3/generate_submission_v21_fast_advanced.py \
  --data-root $DHANRAKSHAK_DATA_ROOT \
  --features-path output/mega_transaction_features.csv

# Stage 2 — replace placeholder windows with real transaction spans
python wide_window_attempt.py \
  --data-root $DHANRAKSHAK_DATA_ROOT

# Run tests
pytest
```

### Data requirements

The raw dataset is not in this repo. It is the public `abhyudayrbih/rbih-nfpc-phase-2` Kaggle release.
`output/mega_transaction_features.csv` (the pre-extracted feature CSV Stage 1 reads) is also not committed
— regenerate it by running `src/transaction_data_v1.py` against the archive.

---

## Submission History

| Version | Public AUC | F1 | Temporal IoU | Change |
|---------|----------:|---:|-------------:|--------|
| V5 | 0.9643 | 0.5307 | 0.185 | Initial baseline |
| V15 | 0.9655 | 0.5549 | 0.225 | Ensemble diversity |
| V20 | 0.9742 | 0.5648 | 0.181 | Best single-metric AUC at the time |
| V21 | 0.9773 | 0.5879 | 0.183 | Fast 4-model ensemble |
| **V21 + wide windows** | **0.9773** | **0.5879** | **0.680** | Stage 2 IoU breakthrough |

27+ total iterations.

---

## Notes on Reproducibility

The competition scores (0.9773 public, 0.9671 private) are from the original submission
and cannot be reproduced from this repo alone — the pre-built feature CSV and original training
data environment are not retained here.

A subsequent held-out validation run on Kaggle (5-fold, fold 0) reproduced an ensemble AUC of **0.9203**
(see `verified_metrics.json`). The gap from 0.9203 to the leaderboard 0.9773 is most likely
a training-fold vs held-out AUC difference — tree ensembles with these hyperparameters on imbalanced
data typically show a gap in this range. That hypothesis hasn't been confirmed by running both
train and held-out AUC in the same notebook.

---

## License

MIT © 2024 Shivangi Singh — see [LICENSE](LICENSE).
