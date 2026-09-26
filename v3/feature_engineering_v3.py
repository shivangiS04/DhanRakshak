"""
feature_engineering_v3.py
Key improvements over v2:
- Fixed account_age_days to use dataset end date, not datetime.now()
- Added structuring detection (txns in 9000-9999 band)
- Added fan-in / fan-out ratio
- Added dormant activation features
- Added pass-through speed (largest inflow → outflow gap)
- Fixed velocity windows (rolling, not last-N-hours-from-max)
- ADDED: suspicious_start / suspicious_end detection for Temporal IoU
- Added income-mismatch proxy features
- Added round-number burst detection
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Dataset time boundaries — use these instead of datetime.now()
DATASET_START = pd.Timestamp("2020-07-01")
DATASET_END   = pd.Timestamp("2025-06-30")


@dataclass
class AccountFeaturesV3:
    """Account features with Temporal IoU support"""
    account_id: str = ""

    # --- Volume (8) ---
    total_transactions: int = 0
    inflow_transactions: int = 0
    outflow_transactions: int = 0
    transaction_frequency_daily: float = 0.0
    total_inflow: float = 0.0
    total_outflow: float = 0.0
    avg_inflow_amount: float = 0.0
    avg_outflow_amount: float = 0.0

    # --- Variability (6) ---
    inflow_std: float = 0.0
    outflow_std: float = 0.0
    inflow_cv: float = 0.0
    outflow_cv: float = 0.0
    amount_skewness: float = 0.0
    amount_kurtosis: float = 0.0

    # --- Behavioural (10) ---
    sub_threshold_ratio: float = 0.0
    structuring_ratio: float = 0.0          # NEW: txns in [9000, 9999]
    round_amount_ratio: float = 0.0
    inflow_outflow_ratio: float = 0.0
    avg_time_to_transfer_hours: float = 0.0
    rapid_transfer_ratio_24h: float = 0.0
    unique_sources: int = 0
    unique_destinations: int = 0
    source_concentration: float = 0.0
    fan_in_out_ratio: float = 0.0           # NEW: unique_sources / unique_destinations

    # --- Temporal (12) ---
    account_age_days: int = 0
    dormancy_periods: int = 0
    dormancy_before_burst: int = 0          # NEW: days silent before biggest activity spike
    activity_spike_magnitude: float = 0.0
    suspicious_start: Optional[pd.Timestamp] = None
    suspicious_end: Optional[pd.Timestamp] = None
    suspicious_window_days: float = 0.0     # NEW: length of suspicious window
    temporal_anomaly_score: float = 0.0
    transaction_time_entropy: float = 0.0
    day_of_week_concentration: float = 0.0
    pass_through_speed_hours: float = 0.0   # NEW: time between peak inflow → peak outflow
    post_mobile_update_spike: float = 0.0   # NEW: placeholder, populate if mobile_update col exists

    # --- Graph (6) ---
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    pagerank_score: float = 0.0
    community_size: int = 0
    community_density: float = 0.0

    # --- Velocity (8) ---
    velocity_inflow_1d: float = 0.0         # renamed: day window, not 1h (more stable)
    velocity_inflow_7d: float = 0.0         # NEW: 7-day window
    velocity_outflow_1d: float = 0.0
    velocity_outflow_7d: float = 0.0        # NEW
    velocity_spike_ratio: float = 0.0
    inter_transaction_time_mean: float = 0.0
    inter_transaction_time_std: float = 0.0
    transaction_burst_score: float = 0.0

    # --- Risk / composite (6) ---
    pattern_anomaly_score: float = 0.0
    pattern_confidence: float = 0.0
    composite_signal: float = 0.0
    risk_score: float = 0.0

    # --- Network (4) ---
    avg_counterparty_degree: float = 0.0
    counterparty_diversity: float = 0.0
    shared_counterparty_ratio: float = 0.0
    network_risk_score: float = 0.0

    is_mule: Optional[int] = None


class FeatureExtractorV3:
    """
    Enhanced feature extractor — v3.
    Fixes all v2 bugs and adds Temporal IoU window detection.
    """

    def __init__(self,
                 accounts_df: pd.DataFrame,
                 customers_df: Optional[pd.DataFrame] = None,
                 dataset_end: pd.Timestamp = DATASET_END):
        self.accounts = accounts_df
        self.customers = customers_df
        self.dataset_end = dataset_end
        logger.info("FeatureExtractorV3 initialized")

    # ------------------------------------------------------------------
    def extract_features_for_account(self,
                                     account_id: str,
                                     transactions: pd.DataFrame) -> AccountFeaturesV3:
        """Extract all features for one account."""
        f = AccountFeaturesV3(account_id=account_id)

        if transactions.empty:
            return f

        # Ensure timestamp column is parsed
        if 'timestamp' in transactions.columns:
            transactions = transactions.copy()
            transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
            transactions = transactions.sort_values('timestamp')

        self._extract_volume_features(f, transactions)
        self._extract_variability_features(f, transactions)
        self._extract_behavioural_features(f, transactions)
        self._extract_temporal_features(f, transactions)
        self._extract_velocity_features(f, transactions)
        self._detect_suspicious_window(f, transactions)   # ← Temporal IoU
        self._compute_risk_scores(f, transactions)

        return f

    # ------------------------------------------------------------------
    def _extract_volume_features(self, f: AccountFeaturesV3, txns: pd.DataFrame):
        f.total_transactions = len(txns)

        inflow  = txns[txns['transaction_type'] == 'inflow']
        outflow = txns[txns['transaction_type'] == 'outflow']

        f.inflow_transactions  = len(inflow)
        f.outflow_transactions = len(outflow)

        if 'timestamp' in txns.columns and len(txns) > 1:
            date_range = (txns['timestamp'].max() - txns['timestamp'].min()).days
            f.transaction_frequency_daily = f.total_transactions / max(date_range, 1)

        f.total_inflow  = float(inflow['amount'].sum())  if not inflow.empty  else 0.0
        f.total_outflow = float(outflow['amount'].sum()) if not outflow.empty else 0.0
        f.avg_inflow_amount  = float(inflow['amount'].mean())  if not inflow.empty  else 0.0
        f.avg_outflow_amount = float(outflow['amount'].mean()) if not outflow.empty else 0.0

    # ------------------------------------------------------------------
    def _extract_variability_features(self, f: AccountFeaturesV3, txns: pd.DataFrame):
        inflow  = txns[txns['transaction_type'] == 'inflow']
        outflow = txns[txns['transaction_type'] == 'outflow']

        f.inflow_std  = float(inflow['amount'].std())  if not inflow.empty  else 0.0
        f.outflow_std = float(outflow['amount'].std()) if not outflow.empty else 0.0

        if f.avg_inflow_amount  > 0: f.inflow_cv  = f.inflow_std  / f.avg_inflow_amount
        if f.avg_outflow_amount > 0: f.outflow_cv = f.outflow_std / f.avg_outflow_amount

        f.amount_skewness = float(txns['amount'].skew())
        f.amount_kurtosis = float(txns['amount'].kurtosis())

    # ------------------------------------------------------------------
    def _extract_behavioural_features(self, f: AccountFeaturesV3, txns: pd.DataFrame):
        n = len(txns)

        # Sub-threshold (< 10 000)
        f.sub_threshold_ratio = (txns['amount'] < 10_000).sum() / n

        # Structuring: amounts in [9000, 9999] — a known red flag
        structuring = txns['amount'].between(9_000, 9_999)
        f.structuring_ratio = structuring.sum() / n

        # Round amounts (multiples of 1000)
        f.round_amount_ratio = (txns['amount'] % 1_000 == 0).sum() / n

        # Inflow / outflow ratio
        if f.total_outflow > 0:
            f.inflow_outflow_ratio = f.total_inflow / f.total_outflow

        # Average time between consecutive transactions (hours)
        if 'timestamp' in txns.columns and n > 1:
            diffs = txns['timestamp'].diff().dt.total_seconds().dropna() / 3600
            f.avg_time_to_transfer_hours = float(diffs.mean())

        # Rapid transfers: fraction of txns within 24 h of the PREVIOUS transaction
        if 'timestamp' in txns.columns and n > 1:
            diffs_h = txns['timestamp'].diff().dt.total_seconds().dropna() / 3600
            f.rapid_transfer_ratio_24h = (diffs_h <= 24).sum() / n

        # Counterparties
        if 'source_account' in txns.columns:
            f.unique_sources = int(txns['source_account'].nunique())
        if 'destination_account' in txns.columns:
            f.unique_destinations = int(txns['destination_account'].nunique())

        # Fan-in / fan-out ratio  (>1 = more sources than destinations = fan-in)
        if f.unique_destinations > 0:
            f.fan_in_out_ratio = f.unique_sources / f.unique_destinations

        # Source concentration (Herfindahl index)
        if 'source_account' in txns.columns and f.unique_sources > 0:
            src_share = txns['source_account'].value_counts(normalize=True)
            f.source_concentration = float((src_share ** 2).sum())

    # ------------------------------------------------------------------
    def _extract_temporal_features(self, f: AccountFeaturesV3, txns: pd.DataFrame):
        if 'timestamp' not in txns.columns:
            return

        # FIX: use dataset end date, not datetime.now()
        account_open = txns['timestamp'].min()
        f.account_age_days = int((self.dataset_end - account_open).days)

        # Dormancy periods (gaps > 30 days between consecutive transactions)
        if len(txns) > 1:
            gaps_days = txns['timestamp'].diff().dt.total_seconds().dropna() / 86_400
            f.dormancy_periods = int((gaps_days > 30).sum())

            # Dormant-before-burst: days of silence before the busiest 7-day window
            daily = txns.set_index('timestamp').resample('D').size()
            if len(daily) >= 7:
                rolling7 = daily.rolling(7).sum()
                peak_date = rolling7.idxmax()
                # Find the most recent gap > 30 days before peak_date
                before_peak = txns[txns['timestamp'] < peak_date]
                if len(before_peak) > 1:
                    gaps_before = before_peak['timestamp'].diff().dt.total_seconds().dropna() / 86_400
                    long_gaps = gaps_before[gaps_before > 30]
                    f.dormancy_before_burst = int(long_gaps.max()) if len(long_gaps) > 0 else 0

        # Activity spike magnitude (max daily count / mean daily count)
        if len(txns) > 1:
            daily_counts = txns.groupby(txns['timestamp'].dt.date).size()
            if daily_counts.mean() > 0:
                f.activity_spike_magnitude = float(daily_counts.max() / daily_counts.mean())

        # Transaction time entropy (how spread across 24 hours)
        if len(txns) > 1:
            hour_counts = txns['timestamp'].dt.hour.value_counts()
            p = hour_counts / len(txns)
            f.transaction_time_entropy = float(-(p * np.log2(p + 1e-10)).sum() / np.log2(24))

        # Day-of-week concentration
        if len(txns) > 1:
            dow = txns['timestamp'].dt.dayofweek.value_counts(normalize=True)
            f.day_of_week_concentration = float((dow ** 2).sum())

        # Pass-through speed: median hours between biggest inflow day and biggest outflow day
        inflow  = txns[txns['transaction_type'] == 'inflow']
        outflow = txns[txns['transaction_type'] == 'outflow']
        if not inflow.empty and not outflow.empty:
            peak_in  = inflow.loc[inflow['amount'].idxmax(), 'timestamp']
            peak_out = outflow.loc[outflow['amount'].idxmax(), 'timestamp']
            speed_h  = (peak_out - peak_in).total_seconds() / 3600
            # Rapid pass-through: positive and < 72 h is suspicious
            f.pass_through_speed_hours = float(speed_h)

    # ------------------------------------------------------------------
    def _extract_velocity_features(self, f: AccountFeaturesV3, txns: pd.DataFrame):
        """Velocity features using fixed time windows relative to account's last date."""
        if 'timestamp' not in txns.columns:
            return

        last_date = txns['timestamp'].max()

        inflow  = txns[txns['transaction_type'] == 'inflow']
        outflow = txns[txns['transaction_type'] == 'outflow']

        def window_sum(df, days):
            mask = df['timestamp'] >= last_date - pd.Timedelta(days=days)
            return float(df.loc[mask, 'amount'].sum())

        f.velocity_inflow_1d  = window_sum(inflow,  1)
        f.velocity_inflow_7d  = window_sum(inflow,  7)
        f.velocity_outflow_1d = window_sum(outflow, 1)
        f.velocity_outflow_7d = window_sum(outflow, 7)

        # Spike ratio: last-7d inflow vs average 7d inflow over whole history
        total_days = max((txns['timestamp'].max() - txns['timestamp'].min()).days, 7)
        n_weeks = total_days / 7
        avg_weekly_inflow = f.total_inflow / max(n_weeks, 1)
        if avg_weekly_inflow > 0:
            f.velocity_spike_ratio = f.velocity_inflow_7d / avg_weekly_inflow

        # Inter-transaction time stats
        if len(txns) > 1:
            diffs_min = txns['timestamp'].diff().dt.total_seconds().dropna() / 60
            f.inter_transaction_time_mean = float(diffs_min.mean())
            f.inter_transaction_time_std  = float(diffs_min.std())

        # Burst score: peak hourly count / mean hourly count
        if len(txns) > 1:
            hourly = txns.set_index('timestamp').resample('h').size()
            nonzero = hourly[hourly > 0]
            if len(nonzero) > 0 and nonzero.mean() > 0:
                f.transaction_burst_score = float(nonzero.max() / nonzero.mean())

    # ------------------------------------------------------------------
    def _detect_suspicious_window(self, f: AccountFeaturesV3, txns: pd.DataFrame):
        """
        Detect the suspicious activity window for Temporal IoU scoring.

        Strategy:
        1. Compute a 7-day rolling anomaly score combining:
           - volume spike (vs account baseline)
           - structuring signal (sub-threshold txns)
           - pass-through signal (inflow immediately followed by outflow)
        2. The window is the contiguous run of days with score > threshold.
        3. If no clear window, fall back to the busiest 30-day period.
        """
        if 'timestamp' not in txns.columns or len(txns) < 5:
            return

        txns = txns.copy()
        txns['date'] = txns['timestamp'].dt.normalize()

        # Daily metrics
        daily = txns.groupby('date').agg(
            n_txns=('amount', 'count'),
            total_amt=('amount', 'sum'),
            structuring=('amount', lambda x: x.between(9_000, 9_999).sum()),
        )

        if len(daily) < 3:
            # Too sparse — use first/last transaction dates
            f.suspicious_start = txns['timestamp'].min()
            f.suspicious_end   = txns['timestamp'].max()
            f.suspicious_window_days = float((f.suspicious_end - f.suspicious_start).days)
            return

        # 7-day rolling z-score for transaction count
        roll_mean = daily['n_txns'].rolling(7, min_periods=1).mean()
        roll_std  = daily['n_txns'].rolling(7, min_periods=1).std().fillna(1)
        z_volume  = (daily['n_txns'] - roll_mean) / roll_std.replace(0, 1)

        # Structuring signal: fraction of txns that are structuring on that day
        struct_signal = daily['structuring'] / daily['n_txns'].replace(0, 1)

        # Composite daily anomaly score (capped at 0)
        score = (z_volume.clip(lower=0) * 0.6 + struct_signal * 10 * 0.4)

        # Find contiguous window where score > 1.0
        threshold = 1.0
        high_days = score[score >= threshold].index

        if len(high_days) >= 3:
            # Find the longest contiguous run
            best_start, best_end, best_len = self._longest_run(high_days)
            # Expand window by ±3 days to capture surrounding activity
            f.suspicious_start = pd.Timestamp(best_start) - pd.Timedelta(days=3)
            f.suspicious_end   = pd.Timestamp(best_end)   + pd.Timedelta(days=3)
        else:
            # Fallback: 30-day window centred on the peak activity day
            peak_day = daily['n_txns'].idxmax()
            f.suspicious_start = pd.Timestamp(peak_day) - pd.Timedelta(days=15)
            f.suspicious_end   = pd.Timestamp(peak_day) + pd.Timedelta(days=15)

        # Clip to actual transaction range
        txn_start = txns['timestamp'].min()
        txn_end   = txns['timestamp'].max()
        f.suspicious_start = max(f.suspicious_start, txn_start)
        f.suspicious_end   = min(f.suspicious_end,   txn_end)

        f.suspicious_window_days = float(
            (f.suspicious_end - f.suspicious_start).total_seconds() / 86_400
        )

        # Temporal anomaly score = mean score in the suspicious window
        window_scores = score[
            (score.index >= f.suspicious_start.normalize()) &
            (score.index <= f.suspicious_end.normalize())
        ]
        f.temporal_anomaly_score = float(window_scores.mean()) if len(window_scores) > 0 else 0.0

    @staticmethod
    def _longest_run(dates) -> Tuple:
        """Find the longest contiguous run in a DatetimeIndex (by day)."""
        dates = sorted(dates)
        best_start = best_end = dates[0]
        cur_start  = dates[0]
        best_len   = 1

        for i in range(1, len(dates)):
            if (dates[i] - dates[i-1]).days <= 1:
                cur_len = (dates[i] - cur_start).days + 1
                if cur_len > best_len:
                    best_len  = cur_len
                    best_start = cur_start
                    best_end   = dates[i]
            else:
                cur_start = dates[i]

        return best_start, best_end, best_len

    # ------------------------------------------------------------------
    def _compute_risk_scores(self, f: AccountFeaturesV3, txns: pd.DataFrame):
        """Composite rule-based risk score (used as a feature, not final prediction)."""
        score = 0.0

        if f.velocity_spike_ratio > 2.0:       score += 0.25
        if f.rapid_transfer_ratio_24h > 0.3:   score += 0.20
        if f.source_concentration > 0.5:       score += 0.15
        if f.round_amount_ratio > 0.4:         score += 0.10
        if f.structuring_ratio > 0.05:         score += 0.20
        if f.dormancy_before_burst > 60:       score += 0.20
        if f.pass_through_speed_hours < 48 and f.pass_through_speed_hours > 0: score += 0.20
        if f.fan_in_out_ratio > 5:             score += 0.15
        if f.account_age_days < 90:            score += 0.15

        f.risk_score = min(score, 1.0)


# ---------------------------------------------------------------------------
# Feature dict helper (used by pipeline) — single source of truth
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    'total_transactions', 'inflow_transactions', 'outflow_transactions',
    'transaction_frequency_daily', 'total_inflow', 'total_outflow',
    'avg_inflow_amount', 'avg_outflow_amount',
    'inflow_std', 'outflow_std', 'inflow_cv', 'outflow_cv',
    'amount_skewness', 'amount_kurtosis',
    'sub_threshold_ratio', 'structuring_ratio', 'round_amount_ratio',
    'inflow_outflow_ratio', 'avg_time_to_transfer_hours',
    'rapid_transfer_ratio_24h', 'unique_sources', 'unique_destinations',
    'source_concentration', 'fan_in_out_ratio',
    'account_age_days', 'dormancy_periods', 'dormancy_before_burst',
    'activity_spike_magnitude', 'suspicious_window_days',
    'temporal_anomaly_score', 'transaction_time_entropy',
    'day_of_week_concentration', 'pass_through_speed_hours',
    'degree_centrality', 'betweenness_centrality', 'clustering_coefficient',
    'pagerank_score', 'community_size', 'community_density',
    'velocity_inflow_1d', 'velocity_inflow_7d',
    'velocity_outflow_1d', 'velocity_outflow_7d',
    'velocity_spike_ratio', 'inter_transaction_time_mean',
    'inter_transaction_time_std', 'transaction_burst_score',
    'pattern_anomaly_score', 'pattern_confidence', 'composite_signal',
    'risk_score',
    'avg_counterparty_degree', 'counterparty_diversity',
    'shared_counterparty_ratio', 'network_risk_score',
]


def features_to_dict(f: AccountFeaturesV3, composite_signal: float = 0.0) -> Dict:
    """Convert AccountFeaturesV3 to a flat dict for model input."""
    return {col: getattr(f, col, 0.0) or 0.0 for col in FEATURE_COLUMNS
            if col != 'composite_signal'} | {'composite_signal': composite_signal}
