# output/

`label_signals.csv` is intentionally **not** committed to this repository.

It is a per-account derivative of the source dataset (`abhyudayrbih/rbih-nfpc-phase-2` on
Kaggle, licensed CC BY-NC-SA 4.0) and redistributing it here would redistribute a
derivative of licensed raw data outside the terms of that license.

## Regenerating it

1. Obtain a local copy of the Kaggle dataset `abhyudayrbih/rbih-nfpc-phase-2`
   (accounts, train labels, and transaction batches).
2. Point `label_signal.py` at that local copy, either by editing
   `LabelSignalGenerator(data_dir=...)` or via the `data_dir` constructor argument,
   to the directory containing `accounts.parquet`, `train_labels.parquet`, and the
   `transactions/batch-*` shards.
3. Run:
   ```bash
   python label_signal.py
   ```
   This writes `output/label_signals.csv`.
