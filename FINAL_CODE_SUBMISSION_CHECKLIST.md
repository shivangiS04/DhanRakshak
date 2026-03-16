# Final Code Submission Checklist

**Challenge:** National Fraud Prevention Challenge | IIT Delhi × Reserve Bank Innovation Hub  
**Team:** LAAPATALADIES  
**Submission Phase:** Code Submission (Mar 3 - Mar 16, 2026)  
**Best CSV:** `submission_v21_wide_windows.csv`

---

## ✅ PRE-SUBMISSION VERIFICATION

### File Requirements
- [x] ZIP file created: `code.zip` (70 KB)
- [x] Compressed size: 70 KB (< 200 MB limit) ✅
- [x] Uncompressed size: 292 KB (< 200 MB limit) ✅
- [x] No prohibited file types (.parquet, .csv, .hdf5, .npy, .npz, .feather)
- [x] Only allowed types: .py, .md, .txt

### Documentation
- [x] README.md present in root
- [x] README includes environment setup instructions
- [x] README includes step-by-step running instructions
- [x] README includes brief approach description
- [x] README includes performance metrics
- [x] requirements.txt present with all dependencies

### Code Files
- [x] Main script: `v3/generate_submission_v21_fast_advanced.py` (12.9 KB)
- [x] Post-processing script: `wide_window_attempt.py` (9.5 KB)
- [x] Supporting utilities: 20 files in src/ directory
- [x] Total: 26 files

### Dependencies
- [x] numpy>=1.21.0
- [x] pandas>=1.3.0
- [x] scikit-learn>=1.0.0
- [x] xgboost>=1.5.0
- [x] lightgbm>=3.3.0
- [x] tensorflow>=2.8.0
- [x] keras>=2.8.0
- [x] scipy>=1.7.0
- [x] matplotlib>=3.4.0
- [x] seaborn>=0.11.0
- [x] joblib>=1.1.0
- [x] tqdm>=4.62.0
- [x] pyarrow>=6.0.0
- [x] networkx>=2.6.0
- [x] python-dateutil>=2.8.0

---

## ✅ PIPELINE VERIFICATION

### Stage 1: Model Training & Prediction
- [x] Script: `v3/generate_submission_v21_fast_advanced.py`
- [x] Loads features from `mega_transaction_features.csv`
- [x] Splits into train (96,091) and test (64,062) accounts
- [x] Engineers 30+ derived features
- [x] Trains 4-model ensemble:
  - [x] XGBoost (AUC 0.9061)
  - [x] ExtraTrees (AUC 0.8993)
  - [x] GradientBoosting (AUC 0.9080)
  - [x] Neural Network (AUC 0.9112)
- [x] Generates weighted ensemble predictions
- [x] Applies threshold (0.20)
- [x] Generates temporal windows
- [x] Output: `submission_v21_fast_advanced.csv`

### Stage 2: Window Optimization
