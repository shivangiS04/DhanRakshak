import pandas as pd
import numpy as np
import logging
from glob import glob
from collections import defaultdict, Counter
import warnings
import gc
import os

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "../archive"
OUTPUT_DIR = "output"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ScalableFeatureExtractor:

    def __init__(self):
        self.account_stats = defaultdict(self._init_account_stats)
        self.accounts = None
        self.account_to_branch = {}

    # ✅ MEMORY SAFE STRUCTURE (NO LISTS)
    def _init_account_stats(self):
        return {
            "total_txns": 0,
            "total_credit": 0.0,
            "total_debit": 0.0,
            "credit_count": 0,
            "debit_count": 0,
            "structuring_count": 0,
            "round_count": 0,
            "credit_counterparties": Counter(),
            "debit_counterparties": Counter(),
            "first_timestamp": None,
            "last_timestamp": None,
            "time_diffs": [],
        }

    # ============================================================
    # LOAD STATIC DATA
    # ============================================================

    def load_static_data(self):
        logger.info("Loading accounts & branch info...")
        self.accounts = pd.read_parquet(f"{DATA_DIR}/accounts.parquet")
        self.account_to_branch = dict(
            zip(self.accounts["account_id"], self.accounts["branch_code"])
        )

    # ============================================================
    # STREAM TRANSACTIONS (TRUE CONSTANT MEMORY)
    # ============================================================

    def process_transactions(self):
        logger.info("=" * 80)
        logger.info("STREAMING TRANSACTIONS (8GB SAFE)")
        logger.info("=" * 80)

        parts = sorted(glob(f"{DATA_DIR}/transactions/batch-*/part_*.parquet"))
        logger.info(f"Found {len(parts)} transaction parts")

        for i, part in enumerate(parts):
            if i % 25 == 0:
                logger.info(f"Processing part {i+1}/{len(parts)}")

            # ✅ READ ONLY REQUIRED COLUMNS
            df = pd.read_parquet(
                part,
                columns=[
                    "account_id",
                    "transaction_timestamp",
                    "txn_type",
                    "amount",
                    "counterparty_id",
                ],
                engine="pyarrow",
            )

            logger.info(f"Loaded {len(df)} rows")

            df["timestamp"] = pd.to_datetime(df["transaction_timestamp"], format='mixed')

            for account_id, group in df.groupby("account_id"):
                stats = self.account_stats[account_id]

                stats["total_txns"] += len(group)

                credits = group[group["txn_type"] == "C"]
                debits = group[group["txn_type"] == "D"]

                credit_sum = credits["amount"].sum()
                debit_sum = debits["amount"].sum()

                stats["total_credit"] += credit_sum
                stats["total_debit"] += debit_sum

                stats["credit_count"] += len(credits)
                stats["debit_count"] += len(debits)

                # Timestamp range only (no storing lists)
                min_ts = group["timestamp"].min()
                max_ts = group["timestamp"].max()

                if stats["first_timestamp"] is None or min_ts < stats["first_timestamp"]:
                    stats["first_timestamp"] = min_ts

                if stats["last_timestamp"] is None or max_ts > stats["last_timestamp"]:
                    stats["last_timestamp"] = max_ts

                # Structuring detection (vectorized)
                amounts = group["amount"].values

                stats["structuring_count"] += (
                    ((amounts > 47500) & (amounts < 50000)) |
                    ((amounts > 95000) & (amounts < 100000)) |
                    ((amounts > 190000) & (amounts < 200000)) |
                    ((amounts > 475000) & (amounts < 500000))
                ).sum()

                # Round amounts detection
                stats["round_count"] += (
                    ((amounts >= 990) & (amounts <= 1010)) |
                    ((amounts >= 4950) & (amounts <= 5050)) |
                    ((amounts >= 9900) & (amounts <= 10100)) |
                    ((amounts >= 49500) & (amounts <= 50500)) |
                    ((amounts >= 99000) & (amounts <= 101000))
                ).sum()

                # Counterparties
                stats["credit_counterparties"].update(
                    credits["counterparty_id"].value_counts().to_dict()
                )
                stats["debit_counterparties"].update(
                    debits["counterparty_id"].value_counts().to_dict()
                )

                # VELOCITY FEATURES: Calculate time differences incrementally
                if len(group) > 1:
                    sorted_group = group.sort_values("timestamp")
                    ts_values = sorted_group["timestamp"].values
                    time_diffs = np.diff(ts_values).astype('timedelta64[m]').astype(float)
                    stats["time_diffs"].extend(time_diffs.tolist())

            del df
            gc.collect()

    # ============================================================
    # FINAL FEATURE CALCULATION (NO DBSCAN)
    # ============================================================

    def build_features(self):
        logger.info("=" * 80)
        logger.info("BUILDING FINAL FEATURES")
        logger.info("=" * 80)

        features = []

        for idx, (account_id, stats) in enumerate(self.account_stats.items()):

            if idx % 10000 == 0:
                logger.info(f"Finalizing account {idx}")

            total_txns = stats["total_txns"]
            total_credit = stats["total_credit"]
            total_debit = stats["total_debit"]

            avg_credit = (
                total_credit / stats["credit_count"]
                if stats["credit_count"] > 0 else 0
            )

            avg_debit = (
                total_debit / stats["debit_count"]
                if stats["debit_count"] > 0 else 0
            )

            structuring_ratio = (
                stats["structuring_count"] / total_txns if total_txns else 0
            )

            round_ratio = (
                stats["round_count"] / total_txns if total_txns else 0
            )

            credit_concentration = (
                max(stats["credit_counterparties"].values()) / stats["credit_count"]
                if stats["credit_count"] > 0 else 0
            )

            debit_concentration = (
                max(stats["debit_counterparties"].values()) / stats["debit_count"]
                if stats["debit_count"] > 0 else 0
            )

            counterparty_diversity = (
                len(
                    set(stats["credit_counterparties"].keys()).union(
                        stats["debit_counterparties"].keys()
                    )
                ) / total_txns
                if total_txns else 0
            )

            flow_imbalance = (
                abs(total_credit - total_debit) / (total_credit + total_debit)
                if (total_credit + total_debit) > 0 else 0
            )

            # Simple temporal span instead of DBSCAN
            transaction_span_days = (
                (stats["last_timestamp"] - stats["first_timestamp"]).days
                if stats["first_timestamp"] and stats["last_timestamp"]
                else 0
            )

            # VELOCITY FEATURE: Median time between transactions (in minutes)
            median_time_between_txns = 0
            if len(stats["time_diffs"]) > 0:
                median_time_between_txns = float(np.median(stats["time_diffs"]))

            features.append({
                "account_id": account_id,
                "structuring_ratio": structuring_ratio,
                "round_ratio": round_ratio,
                "total_transactions": total_txns,
                "total_credit_amount": total_credit,
                "total_debit_amount": total_debit,
                "avg_credit_amount": avg_credit,
                "avg_debit_amount": avg_debit,
                "credit_concentration": credit_concentration,
                "debit_concentration": debit_concentration,
                "counterparty_diversity": counterparty_diversity,
                "transaction_flow_anomaly": flow_imbalance,
                "transaction_span_days": transaction_span_days,
                "median_time_between_txns": median_time_between_txns,
            })

        return pd.DataFrame(features)

    # ============================================================
    # RUN
    # ============================================================

    def run(self):
        self.load_static_data()
        self.process_transactions()
        return self.build_features()


def main():
    extractor = ScalableFeatureExtractor()
    features = extractor.run()

    output_path = os.path.join(OUTPUT_DIR, "mega_transaction_features.csv")
    features.to_csv(output_path, index=False)
    logger.info(f"Saved features to {output_path}")


if __name__ == "__main__":
    main()
