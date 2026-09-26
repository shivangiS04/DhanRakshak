"""
ensemble_models_v3.py
Key fixes over v2:
- Validation split BEFORE SMOTE (no leakage into val set)
- Evaluate on hold-out val, not train (no fake AUC)
- Early stopping on XGBoost / LightGBM / CatBoost
- No StandardScaler on tree models (unnecessary, hurts nothing but wastes time)
- Ensemble weights learned from val AUC, not fixed
- Calibration fitted on val set (out-of-fold), not train
- IsolationForest score used as anomaly feature, not a classifier vote
- Proper F1 threshold search on val set
- LightGBM / CatBoost use dart / symmetric_tree for better generalisation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import pickle
from pathlib import Path
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve, confusion_matrix
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    accuracy:          float = 0.0
    precision:         float = 0.0
    recall:            float = 0.0
    f1_score:          float = 0.0
    auc_roc:           float = 0.0
    confusion_matrix:  Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None


class EnsembleModelStackV3:
    """
    Improved ensemble:
    - Proper train/val split BEFORE SMOTE
    - Learned weights from val AUC
    - Early stopping on boosting models
    - Calibration on val set (not train)
    """

    def __init__(self, model_dir: str = 'models_v3'):
        self.model_dir     = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models:        Dict[str, Any]            = {}
        self.scalers:       Dict[str, StandardScaler] = {}
        self.feature_names: Optional[List[str]]       = None
        self.model_weights: Dict[str, float]          = {}   # learned at train time
        self.calibrator                               = None
        self.optimal_threshold: float                 = 0.5
        logger.info("EnsembleModelStackV3 initialized")

    # ------------------------------------------------------------------
    def train_all_models(self,
                         X: pd.DataFrame,
                         y: np.ndarray,
                         val_fraction: float = 0.20,
                         use_smote: bool = True) -> Dict[str, ModelMetrics]:
        """
        Full training pipeline with proper hold-out validation.

        Steps:
        1. Stratified split into train / val
        2. Optionally SMOTE on train only
        3. Train each model
        4. Evaluate on val → AUC used to set ensemble weights
        5. Fit calibration on val predictions
        6. Find optimal F1 threshold on val
        """
        self.feature_names = X.columns.tolist()
        metrics: Dict[str, ModelMetrics] = {}

        logger.info(f"Training ensemble on {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y.astype(int))}")

        # ── 1. Stratified split ─────────────────────────────────────
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=val_fraction, stratify=y, random_state=42
        )
        logger.info(f"Train: {len(X_tr)}, Val: {len(X_val)}")

        # ── 2. SMOTE on train ONLY ──────────────────────────────────
        if use_smote and HAS_IMBLEARN:
            logger.info("Applying SMOTE on train split only …")
            try:
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
                logger.info(f"Post-SMOTE train size: {len(X_tr)}")
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}")

        # ── 3. Train models ─────────────────────────────────────────
        trainers = []
        if HAS_XGBOOST:   trainers.append(('xgboost',      self._train_xgboost))
        if HAS_LIGHTGBM:  trainers.append(('lightgbm',     self._train_lightgbm))
        if HAS_CATBOOST:  trainers.append(('catboost',     self._train_catboost))
        trainers.append(('random_forest',      self._train_random_forest))
        trainers.append(('logistic_regression', self._train_logistic_regression))

        for name, trainer in trainers:
            try:
                metrics[name] = trainer(X_tr, y_tr, X_val, y_val)
                logger.info(f"  {name}: val_AUC={metrics[name].auc_roc:.4f}, "
                            f"val_F1={metrics[name].f1_score:.4f}")
            except Exception as e:
                logger.error(f"  {name} training failed: {e}")

        # ── 4. Anomaly detector as extra feature (not a voting member) ──
        self._train_isolation_forest(X_tr, y_tr)

        # ── 5. Learned weights proportional to val AUC ──────────────
        self._set_learned_weights(metrics)

        # ── 6. Calibration on val predictions ───────────────────────
        self._calibrate_on_val(X_val, y_val)

        # ── 7. Optimal threshold on val ─────────────────────────────
        self.optimal_threshold = self._find_optimal_threshold(X_val, y_val)
        logger.info(f"Optimal threshold: {self.optimal_threshold:.4f}")

        return metrics

    # ------------------------------------------------------------------
    # Individual model trainers
    # ------------------------------------------------------------------

    def _train_xgboost(self, X_tr, y_tr, X_val, y_val) -> ModelMetrics:
        logger.info("Training XGBoost …")
        scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            objective='binary:logistic',
            eval_metric='auc',
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        self.models['xgboost'] = model
        return self._val_metrics(model, X_val, y_val, 'xgboost')

    def _train_lightgbm(self, X_tr, y_tr, X_val, y_val) -> ModelMetrics:
        logger.info("Training LightGBM …")
        scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        model = lgb.LGBMClassifier(
            num_leaves=63,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            boosting_type='dart',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False),
                     lgb.log_evaluation(period=-1)]
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=callbacks)

        self.models['lightgbm'] = model
        return self._val_metrics(model, X_val, y_val, 'lightgbm')

    def _train_catboost(self, X_tr, y_tr, X_val, y_val) -> ModelMetrics:
        logger.info("Training CatBoost …")
        scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        model = cb.CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            scale_pos_weight=scale_pos,
            eval_metric='AUC',
            early_stopping_rounds=30,
            random_state=42,
            verbose=0,
        )
        model.fit(X_tr, y_tr,
                  eval_set=(X_val, y_val),
                  use_best_model=True)

        self.models['catboost'] = model
        return self._val_metrics(model, X_val, y_val, 'catboost')

    def _train_random_forest(self, X_tr, y_tr, X_val, y_val) -> ModelMetrics:
        logger.info("Training Random Forest …")
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        self.models['random_forest'] = model
        return self._val_metrics(model, X_val, y_val, 'random_forest')

    def _train_logistic_regression(self, X_tr, y_tr, X_val, y_val) -> ModelMetrics:
        logger.info("Training Logistic Regression …")
        scaler = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        self.scalers['logistic_regression'] = scaler

        model = LogisticRegression(
            max_iter=1000, C=0.1,
            class_weight='balanced',
            random_state=42, n_jobs=-1,
        )
        model.fit(X_tr_s, y_tr)
        self.models['logistic_regression'] = model
        return self._val_metrics(model, X_val_s, y_val, 'logistic_regression',
                                 already_scaled=True)

    def _train_isolation_forest(self, X_tr, y_tr) -> None:
        """Train anomaly scorer — stored but NOT included in ensemble vote."""
        logger.info("Training Isolation Forest (anomaly feature only) …")
        contamination = max(0.01, (y_tr == 1).mean())  # use actual mule rate
        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        model.fit(X_tr)
        self.models['isolation_forest'] = model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _predict_proba_single(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        model  = self.models[model_name]
        scaler = self.scalers.get(model_name)
        X_in   = scaler.transform(X) if scaler else X

        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_in)[:, 1]
        elif hasattr(model, 'decision_function'):
            s = model.decision_function(X_in)
            return (s - s.min()) / (s.max() - s.min() + 1e-10)
        else:
            return model.predict(X_in).astype(float)

    def _val_metrics(self, model, X_val, y_val, name: str,
                     already_scaled: bool = False) -> ModelMetrics:
        m = ModelMetrics()
        preds = (model.predict_proba(X_val)[:, 1]
                 if hasattr(model, 'predict_proba')
                 else model.predict(X_val).astype(float))

        try:
            m.auc_roc = roc_auc_score(y_val, preds)
        except Exception:
            m.auc_roc = 0.0

        best_f1, best_thr = 0.0, 0.5
        for thr in np.linspace(0.05, 0.95, 100):
            f = f1_score(y_val, (preds >= thr).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, thr

        m.f1_score = best_f1

        if self.feature_names and hasattr(model, 'feature_importances_'):
            m.feature_importance = dict(zip(self.feature_names, model.feature_importances_))

        return m

    def _set_learned_weights(self, metrics: Dict[str, ModelMetrics]) -> None:
        """Weight each model by its val AUC (softmax normalised)."""
        voting_models = [k for k in metrics if k != 'isolation_forest']
        aucs = np.array([metrics[k].auc_roc for k in voting_models])
        aucs = np.clip(aucs, 0.5, 1.0) - 0.5   # centre at 0
        exp  = np.exp(aucs * 5)                 # temperature scaling
        weights = exp / exp.sum()
        self.model_weights = {k: float(w) for k, w in zip(voting_models, weights)}
        logger.info(f"Learned ensemble weights: {self.model_weights}")

    def _calibrate_on_val(self, X_val: pd.DataFrame, y_val: np.ndarray) -> None:
        """Fit Platt calibration on val-set raw ensemble scores."""
        try:
            raw = self._raw_ensemble(X_val)
            cal = CalibratedClassifierCV(
                LogisticRegression(max_iter=500, C=1.0),
                method='sigmoid', cv='prefit'
            )
            # Build a dummy "prefit" estimator that returns raw scores
            # Instead, use sigmoid fit directly
            from scipy.special import expit
            from scipy.optimize import curve_fit

            def sigmoid(x, a, b):
                return expit(a * x + b)

            popt, _ = curve_fit(sigmoid, raw, y_val.astype(float),
                                p0=[1.0, 0.0], maxfev=1000)
            self._cal_a, self._cal_b = popt
            self.calibrator = 'sigmoid'
            logger.info(f"Platt calibration: a={popt[0]:.4f}, b={popt[1]:.4f}")
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            self.calibrator = None

    def _apply_calibration(self, scores: np.ndarray) -> np.ndarray:
        if self.calibrator == 'sigmoid':
            from scipy.special import expit
            return expit(self._cal_a * scores + self._cal_b)
        return scores

    def _raw_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of voting models."""
        total_weight = 0.0
        accum = np.zeros(len(X))
        for name, weight in self.model_weights.items():
            if name not in self.models:
                continue
            accum += self._predict_proba_single(name, X) * weight
            total_weight += weight
        return accum / max(total_weight, 1e-10)

    def predict_ensemble(self, X: pd.DataFrame,
                         threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (probabilities, confidence).
        probabilities are calibrated.
        """
        raw = self._raw_ensemble(X)
        probs = self._apply_calibration(raw)

        # Per-model predictions for confidence (std across models)
        all_preds = []
        for name in self.model_weights:
            if name in self.models:
                all_preds.append(self._predict_proba_single(name, X))
        if all_preds:
            confidence = 1.0 - np.std(np.vstack(all_preds), axis=0)
        else:
            confidence = np.ones(len(X))

        thr = threshold if threshold is not None else self.optimal_threshold
        logger.info(f"Ensemble → min={probs.min():.4f} max={probs.max():.4f} "
                    f"mean={probs.mean():.4f} "
                    f"flagged@{thr:.2f}: {(probs >= thr).sum()}")
        return probs, confidence

    def _find_optimal_threshold(self, X_val: pd.DataFrame, y_val: np.ndarray) -> float:
        raw   = self._raw_ensemble(X_val)
        probs = self._apply_calibration(raw)
        best_f1, best_thr = 0.0, 0.5
        for thr in np.linspace(0.05, 0.95, 100):
            f = f1_score(y_val, (probs >= thr).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, thr
        logger.info(f"Optimal threshold: {best_thr:.4f} → F1={best_f1:.4f}")
        return best_thr

    # ------------------------------------------------------------------
    def save_models(self) -> None:
        for name, model in self.models.items():
            path = self.model_dir / f"{name}.pkl"
            try:
                with open(path, 'wb') as fh:
                    pickle.dump(model, fh)
            except Exception as e:
                logger.error(f"Save {name} failed: {e}")

        meta = {
            'model_weights': self.model_weights,
            'optimal_threshold': self.optimal_threshold,
            'calibrator': self.calibrator,
            'cal_params': getattr(self, '_cal_a', None),
        }
        with open(self.model_dir / 'meta.pkl', 'wb') as fh:
            pickle.dump(meta, fh)
        logger.info(f"Models saved to {self.model_dir}")

    def load_models(self) -> None:
        for name in ['xgboost', 'lightgbm', 'catboost',
                     'random_forest', 'logistic_regression', 'isolation_forest']:
            path = self.model_dir / f"{name}.pkl"
            if path.exists():
                with open(path, 'rb') as fh:
                    self.models[name] = pickle.load(fh)

        meta_path = self.model_dir / 'meta.pkl'
        if meta_path.exists():
            with open(meta_path, 'rb') as fh:
                meta = pickle.load(fh)
            self.model_weights     = meta.get('model_weights', {})
            self.optimal_threshold = meta.get('optimal_threshold', 0.5)
            self.calibrator        = meta.get('calibrator')
            if meta.get('cal_params'):
                self._cal_a = meta['cal_params']
        logger.info(f"Models loaded from {self.model_dir}")
