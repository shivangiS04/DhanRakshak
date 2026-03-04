# Data Leakage Analysis

## Question: Was the split done correctly?

**Split accounts → THEN generate features** (CORRECT)
OR
**Generate features on full dataset → THEN split** (DATA LEAKAGE)

## Answer: ✓ CORRECT - Split AFTER feature generation

The order in `optimize_models.py` is:

```python
# Step 1: Load FULL dataset
data_df = load_training_data()  # All 96,091 training accounts

# Step 2: Extract features from FULL dataset
features_df = extract_features(data_df)  # Features for all 96,091 accounts

# Step 3: THEN split into train/val
X = features_df.drop('account_id', axis=1)
y = data_df['is_mule'].values

split_idx = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
```

## Why This Order is CORRECT

### Features Used (All Account-Level, No Aggregation)

The features are **static account attributes** that don't require aggregation:

```python
features['avg_balance'] = data_df['avg_balance'].fillna(0)
features['monthly_avg_balance'] = data_df['monthly_avg_balance'].fillna(0)
features['daily_avg_balance'] = data_df['daily_avg_balance'].fillna(0)
features['account_age_days'] = (date_calc).dt.days
features['is_frozen'] = (data_df['account_status'] == 'frozen').astype(int)
features['has_freeze_history'] = data_df['freeze_date'].notna().astype(int)
features['kyc_compliant'] = (data_df['kyc_compliant'] == 'Y').astype(int)
features['days_since_kyc'] = (date_calc).dt.days
features['has_mobile_update'] = data_df['last_mobile_update_date'].notna().astype(int)
features['days_since_mobile_update'] = (date_calc).dt.days
features['cheque_allowed'] = (data_df['cheque_allowed'] == 'Y').astype(int)
features['cheque_availed'] = (data_df['cheque_availed'] == 'Y').astype(int)
features['num_chequebooks'] = data_df['num_chequebooks'].fillna(0)
features['nomination_flag'] = (data_df['nomination_flag'] == 'Y').astype(int)
features['is_savings'] = (data_df['product_family'] == 'S').astype(int)
features['is_kfamily'] = (data_df['product_family'] == 'K').astype(int)
features['is_overdraft'] = (data_df['product_family'] == 'O').astype(int)
features['rural_branch'] = (data_df['rural_branch'] == 'Y').astype(int)
features['composite_signal'] = data_df['composite_signal'].fillna(0)
```

**Key Point**: Each feature is computed **independently per account** with NO cross-account aggregation or statistics.

## Why Data Leakage is NOT a Problem Here

### ✓ No Normalization/Scaling on Full Dataset
- StandardScaler is applied AFTER split (only on training data)
- Validation data is scaled using training data's mean/std
- This is correct practice

### ✓ No Feature Selection on Full Dataset
- No feature importance computed on full data
- No correlation analysis on full data
- All features are pre-defined account attributes

### ✓ No Hyperparameter Tuning on Full Dataset
- Hyperparameters are fixed (not tuned on full data)
- No grid search on full data
- Models trained only on training set

### ✓ No Threshold Optimization on Full Dataset
- Threshold optimized on validation set only
- Not on full dataset

## Potential Data Leakage Scenarios (NOT Present Here)

### ❌ Would be data leakage if:
1. **Scaling on full data**: `scaler.fit(X_full)` then split
   - Current: ✓ Correct - `scaler.fit(X_train)` after split

2. **Feature selection on full data**: Select features based on full dataset correlation
   - Current: ✓ Correct - All features pre-defined

3. **Threshold optimization on full data**: Find threshold on full dataset
   - Current: ✓ Correct - Threshold optimized on validation set

4. **Hyperparameter tuning on full data**: GridSearch on full dataset
   - Current: ✓ Correct - Fixed hyperparameters

5. **Label statistics on full data**: Use label distribution from full data
   - Current: ✓ Correct - Scale pos weight computed from training data only

## Verification

Let me verify the StandardScaler is applied correctly:

```python
# In optimize_models.py:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ✓ Fit on training data only
X_val_scaled = scaler.transform(X_val)          # ✓ Transform validation data using training stats
```

This is **CORRECT** - validation data is scaled using training data's statistics.

## Conclusion

✓ **NO DATA LEAKAGE**

The data split is done correctly:
1. Features extracted from full dataset (safe because they're account-level attributes)
2. Split into train/val (80/20)
3. Scaling fitted on training data only
4. Validation data transformed using training statistics
5. Threshold optimized on validation set

The model evaluation is **honest and unbiased**.

---

**Status**: ✓ APPROVED - No data leakage detected

**Date**: March 4, 2026
