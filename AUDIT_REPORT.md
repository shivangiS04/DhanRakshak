# DhanRakshak ICAIF 2026 Paper Audit — Final Report

## Summary Table: Claims vs. Code Reality

| # | Claim (from paper/README) | File:Line | Status | Finding |
|----|---|---|---|---|
| **1a** | Stage 2: `wide_window_attempt.py` exists and implements "wide window strategy" | README.md:33-37, 74; .gitignore:132 | **RECONSTRUCTED** | Script was never a tracked file; archived in deleted code.zip/submission_code_final.zip (commit 86f9aff^). Extracted, restored to repo root, and path-parameterized. The authentic script uses `MULE_THRESHOLD=0.20`, enforces 30-day min windows, and applies dataset-wide fallback (`2020-07-01`–`2025-06-30`) for accounts with no transaction match — behavior the README does not fully specify. |
| **1b** | Stage 1's `generate_window()` uses real per-account transaction timestamps | v3/generate_submission_v21_fast_advanced.py:174-217 | **CONTRADICTED** | `generate_window()` sets windows from hardcoded fixed 2025 calendar-date buckets (e.g., `prob > 0.8` → `datetime(2025,2,1)` to `datetime(2025,5,31)`), keyed only on probability score and a few heuristics (transaction_span, structuring_ratio, flow_anomaly). No per-account first/last timestamp is used. Stage 1 output `suspicious_start`/`suspicious_end` are fake placeholders; Stage 2 overwrites them with real spans. |
| **2** | Probability rescaling to [0.01, 0.99] is applied before writing submission | Final_Solution_Report_Word.pdf (narrative, results table, appendix); README.md (none) | **CONTRADICTED — ABSENT FROM CODE** | Searched entire repo and git history (`grep -r "0.01\|0.99\|rescale.*proba\|clip.*proba"`): no such rescaling exists in any submission-generation script. The claim appears only in the PDF report's narrative and appendix, not in code. The one `np.clip` for probabilities found (`f1_calibration_postprocessor.py:30`) uses different bounds (0.001, 0.999), applies power transform and renormalization, and is marked "Optional" post-processing. **Paper should be corrected; no code change needed.** |
| **3a** | Stage 1 validation AUCs: XGBoost 0.9061, ExtraTrees 0.8993, GradientBoosting 0.9080, NN 0.9112 | v3/generate_submission_v21_fast_advanced.py:93,105,118,142 | **UNVERIFIABLE FROM REPO** | Per-model AUCs are logged via `logger.info()` (lines 93/105/118/142) to both stdout and `submission_v21_fast_advanced.log`. No structured artifact (JSON, CSV) persists them. No output_v21_fast_advanced/ directory or log file exists in the repo; no git history contains these exact numbers. **Fix: added structured JSON metrics persistence** (output_dir/`metrics_v21_fast_advanced.json` with `model_auc`, `ensemble_val_auc`, `ensemble_weights`, `run_timestamp`). Exact AUC values remain unverified without data. |
| **3b** | Ensemble validation AUC: 0.9773 (v21) / 0.9671 (wide_windows baseline) | v3/generate_submission_v21_fast_advanced.py:286; README.md:16 | **UNVERIFIABLE FROM REPO** | Ensemble AUC computed at line 286 (`val_auc = roc_auc_score(y_val_fold, val_probs)`) and logged at lines 286, 347. No artifact persists it; no output directory/file exists in the repo. No git history contains these specific numbers (searched all blobs via `git rev-list --all` × `git grep`). **Fix: same as 3a — JSON metrics now saved.** Actual numbers remain unverified. |
| **4a** | Temporal IoU 0.6181 (after wide-window strategy) | README.md:269; Final_Solution_Report_Word.pdf; f1_calibration_postprocessor.py:195 | **BLOCKED — REQUIRES DATA + EVALUATION SCRIPT** | The wide-window script's own log only prints an **estimated** range ("IoU=0.15-0.35 estimated"). Computing ground-truth IoU requires running `src/evaluation.py` with real `train_labels.parquet` ground truth. Raw dataset not on this machine; Kaggle download in progress. Cannot verify without data. |
| **4b** | F1 score 0.5090 (wide-windows) | README.md:16 | **BLOCKED — REQUIRES DATA** | No artifact in repo. Cannot compute without running full pipeline on real data. |
| **5** | PDF Appendix describes functions `get_account_date_spans`, `find_peak_window`, `generate_smart_window` in `src/temporal_window_generator.py` | Final_Solution_Report_Word.pdf Appendix; src/temporal_window_generator.py (actual) | **CONTRADICTED** | Actual functions in the file: `find_suspicious_window`, `_find_peak_activity`, `_find_sustained_activity`, `_find_anomalous_period`, `generate_windows`. The PDF's claimed function names do not exist. This is a separate documentation divergence from code. |
| **6** | Hardcoded path `/Users/shivangisingh/Desktop/archive` is the sole data root used | v3/generate_submission_v21_fast_advanced.py:233 (+ 6+ other files) | **RECONSTRUCTED — PATH NOW PARAMETERIZED** | Confirmed present at line 233 and duplicated across 7 distinct source files (v3/generate_submission_v25*, src/data_loader.py:869, src/pipeline.py:581, src_enhanced/pipeline_v2.py:491, src/transaction_data_v1.py:15, src/transaction_data_enhanced.py:15, src/add_velocity_features.py:15) plus byte-identical copies under submission_code_final/ and submission_package/. **Fix: v3/generate_submission_v21_fast_advanced.py now accepts `--data-root` (default `$DHANRAKSHAK_DATA_ROOT` env var, fallback to hardcoded path).** Sibling files left untouched per user scope. |
| **7** | No evidence in repo history that headline numbers were ever computed | git history search (0.9671, 0.6181, 0.5090, 0.9773, 0.5879, 0.6850) | **CONFIRMED** | Searched all blobs in entire git history via `git rev-list --all | xargs git grep`: these exact numbers appear only in documentation files (README.md, deleted markdown checklists) and code comments. Never in a log file, JSON, CSV, or any computed artifact. An older, different pipeline's results file (`RESULTS_SUMMARY.txt`, deleted at commit 8b08a86, AUC 0.8724) exists but is from a separate feature set/model list (19 KYC/balance features, 64K accounts, not the v21 ensemble). **Conclusion: no artifact anywhere demonstrates the v21/wide-window headline numbers were computed by the shipped code.** |

---

## Code Changes Made (Implementation Complete)

### ✓ Stage 1: `v3/generate_submission_v21_fast_advanced.py`
- **Added imports**: `argparse`, `os`, `json` (lines 8-10)
- **Added `parse_args()` function** (lines 220-240): accepts `--data-root`, `--features-path`, `--output-dir` with sensible defaults
- **Updated `main(argv=None)`** (line 242): now calls `parse_args(argv)` first
- **Parameterized hardcoded paths**:
  - Line 247: `features_path = Path(args.features_path)`
  - Line 258: `data_root = Path(args.data_root)`
  - Line 370: `output_dir = Path(args.output_dir)`
- **Enhanced `train_fast_ensemble()` return** (line 177): now returns `(models, weights, auc_scores)` instead of `(models, weights)`
- **Updated call site** (line 304): captures third return value `models_dict, weights, auc_scores = ...`
- **Added structured metrics logging** (lines 373-378): writes `output_dir/metrics_v21_fast_advanced.json` with:
  ```json
  {
    "model_auc": {"xgb": float, "et": float, "gb": float, "nn": float},
    "ensemble_val_auc": float,
    "ensemble_weights": {...},
    "run_timestamp": "ISO-8601"
  }
  ```

### ✓ Stage 2: `wide_window_attempt.py` (repo root)
- **Restored from git history** (commit 86f9aff^, code.zip); 210 lines, byte-identical to original
- **Added parameterization**:
  - Imports: `argparse`, `os`
  - Function: `parse_args(argv=None)` with `--data-root` (env var `DHANRAKSHAK_DATA_ROOT`), `--input-csv`, `--output-csv`, `--mule-threshold` (default 0.20)
  - `main(argv=None)` now parses args
  - Replaced hardcoded `TXN_DIR`, `INPUT_SUBMISSION`, `OUTPUT_PATH` with arg-derived values
- **Added explicit failure on missing data**: `get_account_date_spans()` raises `FileNotFoundError` if no `*.parquet` files found under `--data-root/transactions/`; `main()` catches, logs, returns exit code 1 — **never writes a no-op output**

### ✓ `.gitignore` (line 132)
- Changed `wide_window_attempt.py` → `!wide_window_attempt.py` to un-ignore it
- File is now tracked (git check-ignore returns false)

### ✓ Syntax verification
- Both scripts parse cleanly (`ast.parse` OK)
- Both scripts accept `--help` without error
- `.gitignore` edit confirmed

---

## Task Status

| Task | Status | Details |
|------|--------|---------|
| **1. Confirm wide_window_attempt.py gap** | ✓ DONE | File recovered from git history, restored, and un-ignored. |
| **2. Reconstruct wide_window_attempt.py** | ✓ DONE | Authentic original restored; path-parameterized; fails loudly if data absent. |
| **3. Fix hardcoded personal path** | ✓ DONE | v3/generate_submission_v21_fast_advanced.py now uses `--data-root` arg. |
| **4. Run full pipeline, report real metrics** | ⏳ **IN PROGRESS** | Kaggle dataset download in progress. Once data arrives, will run: (a) Stage 1 → metrics_v21_fast_advanced.json, (b) Stage 2 → submission_v21_wide_windows.csv, (c) src/evaluation.py → real AUC/F1/Temporal IoU. |
| **5. Check [0.01, 0.99] rescaling claim** | ✓ DONE | Confirmed absent from all code. **Paper should be corrected, not code.** |
| **6. Capture per-model AUCs as artifact** | ✓ DONE | JSON metrics persistence added; per-model + ensemble AUCs now saved. |

---

## Key Findings for Your Paper

1. **Stage 2 wide-window logic exists** — but was never committed; recovered from git history. Use its authentic behavior (0.20 threshold, 30-day minimum, dataset-wide fallback) in your paper, not the README's simplified version.

2. **Stage 1 windows are fake** — hardcoded calendar-date buckets, not real per-account timestamps. This should be clarified in your paper: "Stage 1 generates placeholder windows; Stage 2 replaces them with real per-account transaction spans."

3. **Headline numbers unverifiable** — no artifact in repo history shows they were computed. This is honest to disclose: "The v21 pipeline was not run end-to-end during development; headline numbers reflect the original competition run. See Task 4 below for reproducibility."

4. **Probability rescaling claim should be removed** — it doesn't exist in code. Either delete it or qualify it as "considered but not implemented."

5. **PDF appendix has fabricated function names** — update it to match actual code (`find_suspicious_window`, `_find_peak_activity`, etc.) or remove the pseudo-code.

---

## Next Steps (Pending Data Availability)

Once the Kaggle dataset download completes:

```bash
export DHANRAKSHAK_DATA_ROOT=<path-to-kaggle-data>
python v3/generate_submission_v21_fast_advanced.py
python wide_window_attempt.py
python src/evaluation.py  # or your evaluation script
```

Then compare the real metrics against what you claim in the paper, and update the paper accordingly.

---

**Report generated**: 2026-09-06
**Status**: Code reconstruction and parameterization complete; awaiting data for Task 4 verification.
