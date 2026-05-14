import numpy as np
import pandas as pd
import logging
import os
import json
from typing import Tuple, Optional, Dict, List, Union
from datetime import datetime
from collections import defaultdict

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    log_loss, confusion_matrix, classification_report,
    accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from features.features import MomentumFeatureEngine

# ═══════════════════════════════════════════════════════════════════════════════
# PURGED TIME SERIES SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

class PurgedTimeSeriesSplit:
    """
    Purged Walk-Forward Cross-Validation.

    Key properties:
        - Growing training window (standard walk-forward)
        - Purge gap between train end and val start (prevents leakage)
        - Embargo applied AFTER validation set (not inside it)
        - Minimum training size enforcement
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 20,
        embargo: int = 0,
        min_train_size: Optional[int] = None
    ):
        """
        Args:
            n_splits:       Number of CV folds
            purge_gap:      Bars to skip between train end and val start
                            Should equal the label look-forward period
            embargo:        Additional bars to skip after val end
                            Guards against serial correlation spill-over
            min_train_size: Minimum bars required in training set
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if purge_gap < 0:
            raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")
        if embargo < 0:
            raise ValueError(f"embargo must be >= 0, got {embargo}")

        self.n_splits      = n_splits
        self.purge_gap     = purge_gap
        self.embargo       = embargo
        self.min_train_size = min_train_size

    # ──────────────────────────────────────────────────────────────────────────
    def split(self, X, y=None, groups=None):
        """
        Yield (train_indices, val_indices) for each fold.

        Walk-forward structure (example with n_splits=3, gap=G, test_size=T):

            Fold 1:  train=[0 .. T-1]         gap   val=[T+G .. 2T+G-1]
            Fold 2:  train=[0 .. 2T-1]        gap   val=[2T+G .. 3T+G-1]
            Fold 3:  train=[0 .. 3T-1]        gap   val=[3T+G .. 4T+G-1]
        """
        n        = len(X)
        indices  = np.arange(n)

        # Each fold contributes one validation block of this size
        test_size = n // (self.n_splits + 1)

        if test_size < 1:
            raise ValueError(
                f"Dataset too small for {self.n_splits} splits "
                f"(n={n}, test_size={test_size})"
            )

        for i in range(self.n_splits):
            # ── Training window (grows each fold) ──────────────────────────
            train_end = test_size * (i + 1)          # exclusive upper bound

            # ── Validation window ──────────────────────────────────────────
            val_start = train_end + self.purge_gap
            val_end   = val_start + test_size

            # Hard boundary
            if val_end > n:
                break

            # Minimum training size guard
            if self.min_train_size and train_end < self.min_train_size:
                continue

            if val_start >= val_end:
                continue

            yield indices[:train_end], indices[val_start:val_end]

    # ──────────────────────────────────────────────────────────────────────────
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    # ──────────────────────────────────────────────────────────────────────────
    def visualize_splits(self, n_samples: int) -> None:
        """Print a text diagram of the fold structure for debugging."""
        print(f"\nPurgedTimeSeriesSplit — n={n_samples}, "
              f"gap={self.purge_gap}, embargo={self.embargo}")
        print(f"{'Fold':>5} | {'Train range':>22} | {'Gap':>5} | {'Val range':>22}")
        print("─" * 65)

        X_dummy = np.arange(n_samples)
        for fold, (tr, val) in enumerate(self.split(X_dummy), 1):
            gap_actual = val[0] - tr[-1] - 1
            print(
                f"{fold:>5} | "
                f"[{tr[0]:>6} .. {tr[-1]:>6}] ({len(tr):>6} bars) | "
                f"{gap_actual:>5} | "
                f"[{val[0]:>6} .. {val[-1]:>6}] ({len(val):>6} bars)"
            )




# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILITY CALIBRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ProbabilityCalibrator:
    """
    Post-hoc probability calibration for multi-class XGBoost output.

    Why calibrate?
        XGBoost probabilities are often overconfident (peaked distributions).
        Downstream ensemble models that weight signals by probability need
        well-calibrated probabilities — otherwise the momentum model will
        dominate inappropriately in high-confidence but wrong predictions.

    Method:
        One-vs-rest isotonic regression (non-parametric, no shape assumptions).
        Outputs are renormalised to sum to 1 after calibration.
    """

    def __init__(self, method: str = 'isotonic'):
        """
        Args:
            method: 'isotonic' (recommended) or 'sigmoid' (Platt scaling)
        """
        if method not in ('isotonic', 'sigmoid'):
            raise ValueError(f"method must be 'isotonic' or 'sigmoid', got '{method}'")
        self.method      = method
        self.calibrators = {}
        self.is_fitted   = False
        self.n_classes   = 0

    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        y_prob: np.ndarray,   # (n_samples, n_classes)
        y_true: np.ndarray    # (n_samples,) integer class indices
    ) -> 'ProbabilityCalibrator':
        """
        Fit calibrator on a dedicated calibration set.

        IMPORTANT: This set must be SEPARATE from both training and validation.
        A three-way split is required:
            train → fit XGBoost
            calib → fit this calibrator
            test  → final evaluation
        """
        n_samples, self.n_classes = y_prob.shape

        for c in range(self.n_classes):
            binary = (y_true == c).astype(int)
            probs  = y_prob[:, c]

            if self.method == 'isotonic':
                cal = IsotonicRegression(out_of_bounds='clip')
                cal.fit(probs, binary)
            else:
                cal = LogisticRegression(C=1.0, max_iter=1000)
                cal.fit(probs.reshape(-1, 1), binary)

            self.calibrators[c] = cal

        self.is_fitted = True
        return self

    # ──────────────────────────────────────────────────────────────────────────

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply calibration and renormalise rows to sum to 1.

        Args:
            y_prob: Raw XGBoost probabilities, shape (n_samples, n_classes)

        Returns:
            Calibrated probabilities, same shape
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted — call fit() first")

        calibrated = np.zeros_like(y_prob, dtype=np.float64)

        for c, cal in self.calibrators.items():
            probs = y_prob[:, c]
            if self.method == 'isotonic':
                calibrated[:, c] = cal.predict(probs)
            else:
                calibrated[:, c] = cal.predict_proba(
                    probs.reshape(-1, 1)
                )[:, 1]

        # Renormalise: rows must sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return calibrated / row_sums

    # ──────────────────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> np.ndarray:
        return self.fit(y_prob, y_true).transform(y_prob)

    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        joblib.dump({'calibrators': self.calibrators,
                     'method': self.method,
                     'n_classes': self.n_classes,
                     'is_fitted': self.is_fitted}, path)

    def load(self, path: str) -> 'ProbabilityCalibrator':
        data             = joblib.load(path)
        self.calibrators = data['calibrators']
        self.method      = data['method']
        self.n_classes   = data['n_classes']
        self.is_fitted   = data['is_fitted']
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class XGBoostMomentumDetector:
    """
    XGBoost Momentum Detector V4.0 — 3-Class Ensemble-Ready

    Predicts market momentum direction:
        BEARISH (−1) | NEUTRAL (0) | BULLISH (+1)

    Designed as a momentum component for an ensemble system
    alongside separate trend and volatility models.

    Key improvements over V3.0:
        ✓ MomentumFeatureEngine with 12 feature blocks
        ✓ Fixed triple barrier (symmetric, entry on next Open)
        ✓ Single canonical label conversion path
        ✓ Corrected PurgedTimeSeriesSplit (growing window + proper embargo)
        ✓ Post-hoc probability calibration
        ✓ Ensemble-ready output (continuous signal strength + abstain logic)
        ✓ Stronger regularisation defaults for noisy financial data
    """

    # ── Class-level constants ─────────────────────────────────────────────────
    CLASS_LABELS  = {-1: 'BEARISH',  0: 'NEUTRAL',  1: 'BULLISH'}
    CLASS_TO_IDX  = {-1: 0,          0: 1,           1: 2}
    IDX_TO_CLASS  = { 0: -1,         1: 0,           2: 1}

    # ── Recommended hyperparameters for momentum (noisy signal) ──────────────
    DEFAULT_PARAMS = dict(
        max_depth          = 4,     # Shallow: momentum is a relatively simple signal
        learning_rate      = 0.02,  # Slow learning → better generalisation
        n_estimators       = 2000,  # Large pool; early stopping finds optimum
        subsample          = 0.6,   # Aggressive row subsampling
        colsample_bytree   = 0.6,   # Aggressive column subsampling
        colsample_bylevel  = 0.8,   # Additional regularisation (per depth level)
        colsample_bynode   = 0.8,   # Additional regularisation (per split node)
        gamma              = 0.2,   # Minimum loss reduction to split
        reg_alpha          = 0.3,   # L1 → feature sparsity
        reg_lambda         = 5.0,   # L2 → weight smoothing
        min_child_weight   = 20,    # Large leaves → avoids overfit on finance data
        max_delta_step     = 1,     # Helps multiclass with imbalanced classes
        random_state       = 42
    )

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        max_depth:         int   = DEFAULT_PARAMS['max_depth'],
        learning_rate:     float = DEFAULT_PARAMS['learning_rate'],
        n_estimators:      int   = DEFAULT_PARAMS['n_estimators'],
        subsample:         float = DEFAULT_PARAMS['subsample'],
        colsample_bytree:  float = DEFAULT_PARAMS['colsample_bytree'],
        colsample_bylevel: float = DEFAULT_PARAMS['colsample_bylevel'],
        colsample_bynode:  float = DEFAULT_PARAMS['colsample_bynode'],
        gamma:             float = DEFAULT_PARAMS['gamma'],
        reg_alpha:         float = DEFAULT_PARAMS['reg_alpha'],
        reg_lambda:        float = DEFAULT_PARAMS['reg_lambda'],
        min_child_weight:  int   = DEFAULT_PARAMS['min_child_weight'],
        max_delta_step:    int   = DEFAULT_PARAMS['max_delta_step'],
        random_state:      int   = DEFAULT_PARAMS['random_state']
    ):
        # ── Input validation ──────────────────────────────────────────────
        if max_depth <= 0:
            raise ValueError(f"max_depth must be > 0, got {max_depth}")
        if not 0 < learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0,1], got {learning_rate}")
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be > 0, got {n_estimators}")
        if not 0 < subsample <= 1:
            raise ValueError(f"subsample must be in (0,1], got {subsample}")
        if not 0 < colsample_bytree <= 1:
            raise ValueError(f"colsample_bytree must be in (0,1], got {colsample_bytree}")

        # ── Hyperparameters ───────────────────────────────────────────────
        self.max_depth          = max_depth
        self.learning_rate      = learning_rate
        self.n_estimators       = n_estimators
        self.subsample          = subsample
        self.colsample_bytree   = colsample_bytree
        self.colsample_bylevel  = colsample_bylevel
        self.colsample_bynode   = colsample_bynode
        self.gamma              = gamma
        self.reg_alpha          = reg_alpha
        self.reg_lambda         = reg_lambda
        self.min_child_weight   = min_child_weight
        self.max_delta_step     = max_delta_step
        self.random_state       = random_state

        # ── Sub-components ────────────────────────────────────────────────
        self.feature_engine = MomentumFeatureEngine()
        self.calibrator     = ProbabilityCalibrator(method='isotonic')

        # ── Model state ───────────────────────────────────────────────────
        self.model            = None
        self.feature_names    = None
        self.n_features       = None
        self.feature_importance = None
        self.feature_stability  = None
        self.cv_feature_importances: List[Dict] = []

        # ── Training metadata ─────────────────────────────────────────────
        self.training_history = {}
        self.best_iteration   = None

        # ── Directory layout ──────────────────────────────────────────────
        self.checkpoint_dir = "models/momentum_model/checkpoints"
        self.log_dir        = "models/momentum_model/logs"
        self.trained_dir    = "models/momentum_model/trained"
        self.plot_dir       = "models/momentum_model/plots"

        for d in [self.checkpoint_dir, self.log_dir, self.trained_dir, self.plot_dir]:
            os.makedirs(d, exist_ok=True)

        # ── Logger ────────────────────────────────────────────────────────
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            ))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)

        self.logger.info("XGBoost Momentum Detector V4.0 initialised (3-CLASS)")
        self.logger.info(
            f"  depth={max_depth}, lr={learning_rate}, "
            f"trees={n_estimators}, min_child_weight={min_child_weight}"
        )


    # ═════════════════════════════════════════════════════════════════════════
    # LABEL CREATION
    # ═════════════════════════════════════════════════════════════════════════

    def create_triple_barrier_labels(
        self,
        df: pd.DataFrame,
        time_barrier:       int   = 30,
        atr_multiplier_upper: float = 1.5,
        atr_multiplier_lower: float = 1.5,   # Symmetric → avoids class bias
        atr_period:         int   = 14,
        entry_on_open:      bool  = True ,     # Realistic: enter on next Open
    ) -> pd.Series:
        """
        Triple Barrier Labeling — corrected implementation.

        Fixes vs V3.0:
            1. Symmetric barriers (was 2.0 upper / 1.5 lower → biased bearish)
            2. Entry price = next bar Open (not current Close)
            3. ATR is EWM-based with shift(1) — no look-ahead
            4. Upper barrier tie-break goes to bullish (conservative)

        Args:
            df:                   OHLCV DataFrame with DatetimeIndex
            time_barrier:         Max holding period in bars
            atr_multiplier_upper: ATR multiple for take-profit barrier
            atr_multiplier_lower: ATR multiple for stop-loss barrier
            atr_period:           Lookback for ATR computation
            entry_on_open:        If True, enter at Open of bar i+1

        Returns:
            pd.Series of labels {−1, 0, 1} with NaN for unusable rows
        """
        required = ['Open', 'High', 'Low', 'Close']
        if not all(c in df.columns for c in required):
            raise ValueError(f"DataFrame must contain {required}")

        self.logger.info("Creating triple barrier labels (symmetric ATR barriers)...")

        # ATR with shifted previous close — zero lookahead
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low']  - prev_close).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(span=atr_period, min_periods=atr_period).mean()

        labels = np.full(len(df), np.nan)

        for i in range(atr_period, len(df) - time_barrier):
            # ── Entry price ───────────────────────────────────────────
            if entry_on_open and (i + 1) < len(df):
                entry_price = df['Open'].iloc[i + 1]
                future_start = i + 1
            else:
                entry_price = df['Close'].iloc[i]
                future_start = i + 1

            if entry_price <= 0:
                continue

            current_atr = atr.iloc[i]
            if pd.isna(current_atr) or current_atr <= 0:
                continue

            # ── Barrier levels ────────────────────────────────────────
            upper_price = entry_price + current_atr * atr_multiplier_upper
            lower_price = entry_price - current_atr * atr_multiplier_lower

            # ── Future price path ─────────────────────────────────────
            future_end  = min(future_start + time_barrier, len(df))
            future_high = df['High'].iloc[future_start:future_end].values
            future_low  = df['Low'].iloc[future_start:future_end].values
            future_close  = df['Close'].iloc[future_start:future_end].values

            upper_hits = np.where(future_close >= upper_price)[0]
            lower_hits = np.where(future_close  <= lower_price)[0]

            # ── Label assignment ──────────────────────────────────────
            if len(upper_hits) > 2 and len(lower_hits) > 2:
                # Both barriers hit: whichever comes first wins
                # Tie goes to upper (bullish) — conservative
                labels[i] = 1 if upper_hits[0] < lower_hits[0] else -1

            elif len(upper_hits) > 2:
                labels[i] = 1

            elif len(lower_hits) > 2:
                labels[i] = -1

            else:
                labels[i] = 0   # Time barrier: neutral

        labels_series = pd.Series(labels, index=df.index)

        # ── Statistics ────────────────────────────────────────────────
        valid        = labels_series.dropna()
        total        = len(valid)
        class_counts = valid.value_counts().sort_index()

        self.logger.info(f"  ATR multiplier upper: {atr_multiplier_upper}x")
        self.logger.info(f"  ATR multiplier lower: {atr_multiplier_lower}x")
        self.logger.info(f"  Time barrier:         {time_barrier} bars")
        self.logger.info(f"  Entry on:             {'next Open' if entry_on_open else 'current Close'}")
        self.logger.info("  Class distribution:")
        for v, name in [(-1, 'BEARISH'), (0, 'NEUTRAL'), (1, 'BULLISH')]:
            n   = class_counts.get(v, 0)
            pct = n / total * 100 if total > 0 else 0
            self.logger.info(f"    {v:>2} {name:>10}: {n:>6} ({pct:>5.1f}%)")
        self.logger.info(f"  NaN rows: {labels_series.isna().sum()}")

        return labels_series


    # ═════════════════════════════════════════════════════════════════════════
    # LABEL CONVERSION (single canonical path)
    # ═════════════════════════════════════════════════════════════════════════

    def _convert_labels_to_indices(self, labels: pd.Series) -> np.ndarray:
        """
        Convert raw labels {−1, 0, 1} (int or float) → class indices {0, 1, 2}.

        This is the SINGLE label conversion method.
        Both train() and cross_validate() call this — eliminates the
        inconsistency between +1 shift and dict-map that existed in V3.0.

        Handles float labels (e.g. −1.0) by rounding before mapping.
        Raises ValueError if any label cannot be mapped.
        """
        # Round first: handles -1.0, 0.0, 1.0 from float Series
        labels_int = labels.round().astype(int)

        result  = np.full(len(labels_int), -99, dtype=np.int32)
        for raw_val, idx in self.CLASS_TO_IDX.items():
            result[labels_int == int(raw_val)] = idx

        unmapped = result == -99
        if unmapped.any():
            bad = np.unique(labels_int.values[unmapped])
            raise ValueError(
                f"Cannot map labels {bad} to class indices. "
                f"Expected: {list(self.CLASS_TO_IDX.keys())}"
            )
        return result


    # ═════════════════════════════════════════════════════════════════════════
    # SAMPLE WEIGHTS
    # ═════════════════════════════════════════════════════════════════════════

    def compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Balanced sample weights for class imbalance.

        weight[i] = n_samples / (n_classes × count_of_class[y[i]])

        Args:
            y: Integer class indices {0, 1, 2} — no NaN expected here
               (call after _convert_labels_to_indices)

        Returns:
            Float array of per-sample weights
        """
        y = y.astype(int)
        unique_classes = np.unique(y)
        n_samples  = len(y)
        n_classes  = len(unique_classes)
        class_counts = np.bincount(y, minlength=3)

        class_weights = np.zeros(3)
        for c in unique_classes:
            if class_counts[c] > 0:
                class_weights[c] = n_samples / (n_classes * class_counts[c])

        sample_weights = class_weights[y]

        names = {0: 'BEARISH', 1: 'NEUTRAL', 2: 'BULLISH'}
        self.logger.info("Sample weights (balanced):")
        for c in range(3):
            if class_counts[c] > 0:
                pct = class_counts[c] / n_samples * 100
                self.logger.info(
                    f"  Class {c} ({names[c]:>10}): "
                    f"n={class_counts[c]:>6} ({pct:>5.1f}%), "
                    f"weight={class_weights[c]:.4f}"
                )

        return sample_weights

    
    # ═════════════════════════════════════════════════════════════════════════
    # XGBOOST PARAMETER DICT (centralised)
    # ═════════════════════════════════════════════════════════════════════════

    def _build_xgb_params(self, verbosity: int = 0) -> Dict:
        """
        Single source of truth for XGBoost parameters.
        Called by both train() and cross_validate().
        """
        return {
            'objective':         'multi:softprob',
            'num_class':         3,
            'eval_metric':       'mlogloss',
            'max_depth':         self.max_depth,
            'learning_rate':     self.learning_rate,
            'subsample':         self.subsample,
            'colsample_bytree':  self.colsample_bytree,
            'colsample_bylevel': self.colsample_bylevel,
            'colsample_bynode':  self.colsample_bynode,
            'gamma':             self.gamma,
            'reg_alpha':         self.reg_alpha,
            'reg_lambda':        self.reg_lambda,
            'min_child_weight':  self.min_child_weight,
            'max_delta_step':    self.max_delta_step,
            'seed':              self.random_state,
            'tree_method':       'hist',
            'grow_policy':       'lossguide',
            'verbosity':         verbosity
        }


    # ═════════════════════════════════════════════════════════════════════════
    # PREPARE DATA
    # ═════════════════════════════════════════════════════════════════════════

    def _prepare_X(
        self,
        data: pd.DataFrame,
        fit_feature_names: bool = False
    ) -> np.ndarray:
        """
        Prepare feature matrix.

        Args:
            data:              Feature DataFrame
            fit_feature_names: If True, store feature names (first call only)

        Returns:
            np.float32 array of shape (n_samples, n_features)
        """
        if data.empty:
            raise ValueError("Feature DataFrame is empty")

        # Fill NaN
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"  {nan_count} NaN values found — forward/back filling")
            data = data.ffill().bfill().fillna(0.0)

        if fit_feature_names:
            self.feature_names = list(data.columns)
            self.n_features    = len(self.feature_names)
            self.logger.info(f"  Stored {self.n_features} feature names")
        else:
            if list(data.columns) != self.feature_names:
                raise ValueError(
                    "Feature mismatch between train and predict sets.\n"
                    f"  Expected: {self.feature_names[:5]} ...\n"
                    f"  Got:      {list(data.columns)[:5]} ..."
                )

        return data.values.astype(np.float32)


    # ═════════════════════════════════════════════════════════════════════════
    # EVALUATION HELPER
    # ═════════════════════════════════════════════════════════════════════════

    def _evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        prefix: str = ''
    ) -> Dict:
        """
        Compute all evaluation metrics for a set of predictions.
        Returns a flat dict with optional prefix for easy logging.
        """
        y_pred = np.argmax(y_pred_proba, axis=1)

        return {
            f'{prefix}log_loss':        log_loss(y_true, y_pred_proba),
            f'{prefix}accuracy':        accuracy_score(y_true, y_pred),
            f'{prefix}balanced_acc':    balanced_accuracy_score(y_true, y_pred),
            f'{prefix}f1_macro':        f1_score(y_true, y_pred, average='macro', zero_division=0),
            f'{prefix}f1_weighted':     f1_score(y_true, y_pred, average='weighted', zero_division=0),
            f'{prefix}mcc':             matthews_corrcoef(y_true, y_pred),
        }


    # ═════════════════════════════════════════════════════════════════════════
    # TRAIN
    # ═════════════════════════════════════════════════════════════════════════

    def train(
        self,
        df: pd.DataFrame,
        validation_split:       float = 0.2,
        calibration_split:      float = 0.1,    # NEW: separate calibration set
        purge_gap:              Optional[int] = None,
        use_triple_barrier:     bool  = True,
        use_sample_weights:     bool  = True,
        early_stopping_rounds:  int   = 50,
        verbose_eval:           int   = 25,
        fit_calibrator:         bool  = True,   # NEW: post-hoc calibration
        num_boost_round:        Optional[int] = None,
        **label_kwargs
    ) -> Dict:
        """
        Train XGBoost model with zero data leakage.

        Data split order (chronological):
            [─────── TRAIN ───────][── CALIB ──][── VAL ──]
               (1 − val − calib)      (calib)     (val)
            purge gaps are applied between each adjacent pair.

        Args:
            df:                   OHLCV DataFrame with DatetimeIndex
            validation_split:     Fraction for validation set
            calibration_split:    Fraction for calibration set
            purge_gap:            Bars between train/calib/val (auto = time_barrier)
            use_triple_barrier:   Recommended. If False, uses simple threshold labels
            use_sample_weights:   Apply balanced class weights during training
            early_stopping_rounds:Patience for early stopping
            verbose_eval:         Logging frequency (every N rounds)
            fit_calibrator:       Fit ProbabilityCalibrator on calibration set
            num_boost_round:      Override n_estimators if set
            **label_kwargs:       Passed to create_triple_barrier_labels()

        Returns:
            Dict of training results and metrics
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        self.logger.info("=" * 70)
        self.logger.info(f"Training XGBoost Momentum Detector V4.0 | rows={len(df)}")
        self.logger.info("=" * 70)

        # ─── Step 1: Feature Engineering ────────────────────────────────
        self.logger.info("Step 1/9 | Feature engineering...")
        df_feat = self.feature_engine.compute_features(df)

        # ─── Step 2: Labels ──────────────────────────────────────────────
        self.logger.info("Step 2/9 | Creating labels...")
        if use_triple_barrier:
            labels = self.create_triple_barrier_labels(df, **label_kwargs)
        else:
            self.logger.warning("  Using simple return-based labels (not recommended)")
            horizon = label_kwargs.get('time_barrier', 10)
            ret     = df['Close'].shift(-horizon) / df['Close'] - 1
            ret     = ret.replace([np.inf, -np.inf], np.nan)
            labels  = pd.Series(np.nan, index=df.index)
            labels[ret >  0.002] =  1
            labels[ret < -0.002] = -1
            labels[(ret >= -0.002) & (ret <= 0.002)] = 0
            labels.iloc[-horizon:] = np.nan

        # ─── Step 3: Remove invalid rows ────────────────────────────────
        self.logger.info("Step 3/9 | Removing invalid rows...")
        valid_mask  = ~labels.isna()
        df_feat     = df_feat[valid_mask].copy()
        labels      = labels[valid_mask].copy()
        self.logger.info(f"  Valid rows: {len(df_feat)}")

        # ─── Step 4: Feature columns ─────────────────────────────────────
        self.logger.info("Step 4/9 | Selecting feature columns...")
        exclude_cols = {'Open', 'High', 'Low', 'Close', 'Volume',
                        'Time', 'Date', 'Datetime'}
        feature_cols = [c for c in df_feat.columns if c not in exclude_cols]
        if not feature_cols:
            raise ValueError("No feature columns found after exclusion")
        df_model = df_feat[feature_cols].copy()
        self.logger.info(f"  Features: {len(feature_cols)}")

        # ─── Step 5: Purged three-way split ──────────────────────────────
        self.logger.info("Step 5/9 | Purged three-way chronological split...")

        if purge_gap is None:
            purge_gap = label_kwargs.get('time_barrier', 20)

        n = len(df_model)
        val_size   = int(n * validation_split)
        calib_size = int(n * calibration_split)
        train_size = n - val_size - calib_size - 2 * purge_gap

        if train_size < 200:
            raise ValueError(
                f"Insufficient training data: {train_size} rows. "
                f"Reduce validation_split, calibration_split, or purge_gap."
            )

        train_end  = train_size
        calib_start = train_end  + purge_gap
        calib_end  = calib_start + calib_size
        val_start  = calib_end   + purge_gap

        train_data  = df_model.iloc[:train_end]
        calib_data  = df_model.iloc[calib_start:calib_end]
        val_data    = df_model.iloc[val_start:]

        train_labels = labels.iloc[:train_end]
        calib_labels = labels.iloc[calib_start:calib_end]
        val_labels   = labels.iloc[val_start:]

        self.logger.info(
            f"  Train:  {len(train_data):>6} rows  [0 .. {train_end}]"
        )
        self.logger.info(
            f"  Gap:    {purge_gap:>6} bars  (purged)"
        )
        self.logger.info(
            f"  Calib:  {len(calib_data):>6} rows  [{calib_start} .. {calib_end}]"
        )
        self.logger.info(
            f"  Gap:    {purge_gap:>6} bars  (purged)"
        )
        self.logger.info(
            f"  Val:    {len(val_data):>6} rows  [{val_start} .. {n}]"
        )

        # ─── Step 6: Prepare arrays ──────────────────────────────────────
        self.logger.info("Step 6/9 | Preparing arrays...")
        self.feature_names = None  # Reset before fit
        self.n_features    = None

        X_train = self._prepare_X(train_data.copy(), fit_feature_names=True)
        X_calib = self._prepare_X(calib_data.copy())
        X_val   = self._prepare_X(val_data.copy())

        y_train = self._convert_labels_to_indices(train_labels.reset_index(drop=True))
        y_calib = self._convert_labels_to_indices(calib_labels.reset_index(drop=True))
        y_val   = self._convert_labels_to_indices(val_labels.reset_index(drop=True))

        # ─── Step 7: Sample weights ──────────────────────────────────────
        self.logger.info("Step 7/9 | Computing sample weights...")
        sample_weights = self.compute_sample_weights(y_train) if use_sample_weights else None

        # ─── Step 8: Build DMatrix & Train ───────────────────────────────
        self.logger.info("Step 8/9 | Building DMatrix and training...")
        dtrain = xgb.DMatrix(
            X_train, label=y_train,
            weight=sample_weights,
            feature_names=self.feature_names
        )
        dcalib = xgb.DMatrix(X_calib, label=y_calib, feature_names=self.feature_names)
        dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=self.feature_names)

        params      = self._build_xgb_params(verbosity=1)
        evals       = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round       = num_boost_round or self.n_estimators,
            evals                 = evals,
            early_stopping_rounds = early_stopping_rounds,
            evals_result          = evals_result,
            verbose_eval          = verbose_eval
        )

        self.best_iteration   = self.model.best_iteration
        self.training_history = evals_result

        # ─── Step 9: Calibration ────────────────────────────────────────
        if fit_calibrator:
            self.logger.info("Step 9/9 | Fitting probability calibrator...")
            calib_proba = self.model.predict(dcalib)   # (n_calib, 3)
            self.calibrator.fit(calib_proba, y_calib)
            self.logger.info("  Calibrator fitted (isotonic regression, one-vs-rest)")
        else:
            self.logger.info("Step 9/9 | Skipping calibration (fit_calibrator=False)")

        # ─── Feature importance ──────────────────────────────────────────
        self.feature_importance = self.model.get_score(importance_type='gain')
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importance.items()
        ]).sort_values('importance', ascending=False)

        imp_path = os.path.join(self.log_dir, 'feature_importance.csv')
        importance_df.to_csv(imp_path, index=False)

        self.logger.info("\nTop 15 features (by gain):")
        for _, row in importance_df.head(15).iterrows():
            self.logger.info(f"  {row['feature']:35s}: {row['importance']:.2f}")

        # ─── Evaluation ──────────────────────────────────────────────────
        train_proba = self.model.predict(dtrain)
        val_proba   = self.model.predict(dval)

        # Calibrate validation probabilities for reporting
        if fit_calibrator:
            val_proba_cal = self.calibrator.transform(val_proba)
        else:
            val_proba_cal = val_proba

        train_metrics = self._evaluate(y_train, train_proba,     prefix='train_')
        val_metrics   = self._evaluate(y_val,   val_proba_cal,   prefix='val_')

        # ─── Confusion matrix plot ────────────────────────────────────────
        target_names = [self.CLASS_LABELS[self.IDX_TO_CLASS[i]] for i in range(3)]
        cm = confusion_matrix(y_val, np.argmax(val_proba_cal, axis=1))

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix — Validation Set (V4.0)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'confusion_matrix.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # ─── Summary log ─────────────────────────────────────────────────
        self.logger.info("=" * 70)
        self.logger.info("TRAINING COMPLETE (3-CLASS)")
        self.logger.info("=" * 70)
        self.logger.info(f"  Best iteration: {self.best_iteration}")
        self.logger.info("")
        self.logger.info("  ✅ PRIMARY METRICS (focus here):")
        self.logger.info(f"     MCC       — train: {train_metrics['train_mcc']:.4f}  "
                         f"val: {val_metrics['val_mcc']:.4f}")
        self.logger.info(f"     F1 macro  — train: {train_metrics['train_f1_macro']:.4f}  "
                         f"val: {val_metrics['val_f1_macro']:.4f}")
        self.logger.info(f"     Bal. Acc  — train: {train_metrics['train_balanced_acc']:.4f}  "
                         f"val: {val_metrics['val_balanced_acc']:.4f}")
        self.logger.info("")
        self.logger.info("  Secondary:")
        self.logger.info(f"     Log loss  — train: {train_metrics['train_log_loss']:.4f}  "
                         f"val: {val_metrics['val_log_loss']:.4f}")
        self.logger.info(f"     Accuracy  — train: {train_metrics['train_accuracy']:.4f}  "
                         f"val: {val_metrics['val_accuracy']:.4f}")

        mcc_gap = train_metrics['train_mcc'] - val_metrics['val_mcc']
        status  = "⚠ Possible overfitting" if mcc_gap > 0.15 else "✓ Healthy"
        self.logger.info(f"\n  Overfit check (MCC gap = {mcc_gap:.4f}): {status}")
        self.logger.info("=" * 70)

        # Classification report
        class_report = classification_report(
            y_val,
            np.argmax(val_proba_cal, axis=1),
            target_names=target_names,
            output_dict=True
        )

        results = {
            'best_iteration':       self.best_iteration,
            **train_metrics,
            **val_metrics,
            'confusion_matrix':     cm,
            'classification_report': class_report,
            'training_history':     evals_result,
            'feature_importance':   importance_df
        }
        return results


    # ═════════════════════════════════════════════════════════════════════════
    # CROSS-VALIDATE
    # ═════════════════════════════════════════════════════════════════════════

    def cross_validate(
        self,
        df: pd.DataFrame,
        n_splits:           int   = 5,
        purge_gap:          int   = 20,
        embargo:            int   = 0,
        use_triple_barrier: bool  = True,
        use_sample_weights: bool  = True,
        early_stopping_rounds: int = 50,
        num_boost_round:    Optional[int] = None,
        **label_kwargs
    ) -> Dict:
        """
        Purged walk-forward cross-validation with feature stability tracking.

        Args:
            df:                   OHLCV DataFrame with DatetimeIndex
            n_splits:             Number of CV folds
            purge_gap:            Bars purged between train and val per fold
            embargo:              Additional bars excluded after val per fold
            use_triple_barrier:   Recommended label method
            use_sample_weights:   Balanced class weighting
            early_stopping_rounds:Early stopping patience
            num_boost_round:      Override n_estimators if set
            **label_kwargs:       Passed to label creation

        Returns:
            Summary dict with per-fold metrics and feature stability DataFrame
        """
        self.logger.info("=" * 70)
        self.logger.info(f"Purged {n_splits}-fold cross-validation (V4.0)")
        self.logger.info(f"  purge_gap={purge_gap}, embargo={embargo}")
        self.logger.info("=" * 70)

        # ─── Feature engineering & labelling (done ONCE for all folds) ───
        self.logger.info("Engineering features (all folds)...")
        df_feat = self.feature_engine.compute_features(df)

        self.logger.info("Creating labels (all folds)...")
        if use_triple_barrier:
            labels = self.create_triple_barrier_labels(df, **label_kwargs)
        else:
            horizon = label_kwargs.get('time_barrier', 10)
            ret     = df['Close'].shift(-horizon) / df['Close'] - 1
            ret     = ret.replace([np.inf, -np.inf], np.nan)
            labels  = pd.Series(np.nan, index=df.index)
            labels[ret >  0.002] =  1
            labels[ret < -0.002] = -1
            labels[(ret >= -0.002) & (ret <= 0.002)] = 0
            labels.iloc[-horizon:] = np.nan

        # Clean
        valid_mask  = ~labels.isna()
        df_feat     = df_feat[valid_mask].copy()
        labels      = labels[valid_mask].copy()

        exclude_cols = {'Open', 'High', 'Low', 'Close', 'Volume',
                        'Time', 'Date', 'Datetime'}
        feature_cols = [c for c in df_feat.columns if c not in exclude_cols]
        df_model     = df_feat[feature_cols].copy()

        self.logger.info(f"  Valid rows: {len(df_model)}, Features: {len(feature_cols)}")

        # ─── CV splitter ─────────────────────────────────────────────────
        tscv = PurgedTimeSeriesSplit(
            n_splits=n_splits,
            purge_gap=purge_gap,
            embargo=embargo
        )
        tscv.visualize_splits(len(df_model))

        # ─── Storage ─────────────────────────────────────────────────────
        cv_results: Dict[str, List] = defaultdict(list)
        fold_importances: List[Dict] = []

        params = self._build_xgb_params(verbosity=0)

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(df_model), 1):
            self.logger.info(f"\n{'─'*70}")
            self.logger.info(f"FOLD {fold}/{n_splits} | "
                             f"train={len(tr_idx)}, val={len(val_idx)}")

            # Reset state
            self.model         = None
            self.feature_names = None
            self.n_features    = None

            # Arrays
            X_tr  = df_model.iloc[tr_idx].copy()
            X_val = df_model.iloc[val_idx].copy()

            # NaN fill
            X_tr  = X_tr.ffill().bfill().fillna(0.0)
            X_val = X_val.ffill().bfill().fillna(0.0)

            X_tr_arr  = X_tr.values.astype(np.float32)
            X_val_arr = X_val.values.astype(np.float32)

            self.feature_names = feature_cols
            self.n_features    = len(feature_cols)

            # Labels — use single canonical converter
            y_tr_raw  = labels.iloc[tr_idx].reset_index(drop=True)
            y_val_raw = labels.iloc[val_idx].reset_index(drop=True)
            y_tr  = self._convert_labels_to_indices(y_tr_raw)
            y_val = self._convert_labels_to_indices(y_val_raw)

            # Weights
            sw = self.compute_sample_weights(y_tr) if use_sample_weights else None

            # DMatrix
            dtrain = xgb.DMatrix(
                X_tr_arr, label=y_tr,
                weight=sw,
                feature_names=self.feature_names
            )
            dval = xgb.DMatrix(
                X_val_arr, label=y_val,
                feature_names=self.feature_names
            )

            # Train
            evals_result = {}
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round       = num_boost_round or self.n_estimators,
                evals                 = [(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds = early_stopping_rounds,
                evals_result          = evals_result,
                verbose_eval          = False
            )

            # Feature importance
            fold_importances.append(self.model.get_score(importance_type='gain'))

            # Metrics
            val_proba  = self.model.predict(dval)
            fold_metrics = self._evaluate(y_val, val_proba, prefix='val_')

            for k, v in fold_metrics.items():
                cv_results[k].append(v)
            cv_results['best_iterations'].append(self.model.best_iteration)

            self.logger.info(
                f"  MCC={fold_metrics['val_mcc']:.4f}  "
                f"F1(macro)={fold_metrics['val_f1_macro']:.4f}  "
                f"BalAcc={fold_metrics['val_balanced_acc']:.4f}  "
                f"BestIter={self.model.best_iteration}"
            )

        # ─── Feature stability ────────────────────────────────────────────
        self.logger.info("\nComputing feature stability across folds...")

        all_features = set()
        for fi in fold_importances:
            all_features.update(fi.keys())

        rank_store: Dict[str, List[int]] = defaultdict(list)

        for fi in fold_importances:
            sorted_feats = sorted(fi, key=fi.get, reverse=True)
            rank_map     = {f: r for r, f in enumerate(sorted_feats)}
            max_rank     = len(sorted_feats)
            for feat in all_features:
                rank_store[feat].append(rank_map.get(feat, max_rank))

        stability_rows = []
        for feat, ranks in rank_store.items():
            r_arr = np.array(ranks)
            stability_rows.append({
                'feature':    feat,
                'mean_rank':  r_arr.mean(),
                'std_rank':   r_arr.std(),
                'stability':  1.0 / (1.0 + r_arr.std())
            })

        stability_df = pd.DataFrame(stability_rows).sort_values(
            'stability', ascending=False
        ).reset_index(drop=True)

        stability_path = os.path.join(self.log_dir, 'feature_stability.csv')
        stability_df.to_csv(stability_path, index=False)
        self.feature_stability = stability_df

        self.logger.info("Top 10 most stable features:")
        for _, row in stability_df.head(10).iterrows():
            self.logger.info(
                f"  {row['feature']:35s}: "
                f"stability={row['stability']:.3f}, "
                f"mean_rank={row['mean_rank']:.1f}, "
                f"std_rank={row['std_rank']:.1f}"
            )

        # ─── Summary statistics ───────────────────────────────────────────
        metric_keys = [
            'val_log_loss', 'val_accuracy', 'val_balanced_acc',
            'val_f1_macro', 'val_f1_weighted', 'val_mcc'
        ]

        summary = {}
        for k in metric_keys:
            vals = cv_results[k]
            summary[f'mean_{k}'] = float(np.mean(vals))
            summary[f'std_{k}']  = float(np.std(vals))

        summary['mean_best_iteration'] = float(np.mean(cv_results['best_iterations']))
        summary['fold_results']        = dict(cv_results)
        summary['feature_stability']   = stability_df

        self.logger.info(f"\n{'='*70}")
        self.logger.info("CROSS-VALIDATION SUMMARY (3-class)")
        self.logger.info(f"{'='*70}")
        self.logger.info("  ✅ PRIMARY METRICS:")
        self.logger.info(
            f"     MCC:         {summary['mean_val_mcc']:.4f}"
            f" ± {summary['std_val_mcc']:.4f}"
        )
        self.logger.info(
            f"     F1 macro:    {summary['mean_val_f1_macro']:.4f}"
            f" ± {summary['std_val_f1_macro']:.4f}"
        )
        self.logger.info(
            f"     Bal. Acc:    {summary['mean_val_balanced_acc']:.4f}"
            f" ± {summary['std_val_balanced_acc']:.4f}"
        )
        self.logger.info("  Secondary:")
        self.logger.info(
            f"     Log loss:    {summary['mean_val_log_loss']:.4f}"
            f" ± {summary['std_val_log_loss']:.4f}"
        )
        self.logger.info(
            f"     Accuracy:    {summary['mean_val_accuracy']:.4f}"
            f" ± {summary['std_val_accuracy']:.4f}"
        )
        self.logger.info(
            f"     Best iter:   {summary['mean_best_iteration']:.1f}"
        )
        self.logger.info(f"{'='*70}")

        return summary


    # ═════════════════════════════════════════════════════════════════════════
    # PREDICT (single-bar, for live use)
    # ═════════════════════════════════════════════════════════════════════════

    def predict(
        self,
        df: pd.DataFrame,
        return_proba:   bool  = True
    ) -> Dict:
        """
        Raw prediction on latest bar (uses last row of df).

        Returns the full probability distribution and discrete prediction.
        For ensemble consumption, use predict_for_ensemble() instead.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        if self.feature_names is None:
            raise ValueError("Feature names not stored — retrain or reload model")
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty")

        # Features
        features = self.feature_engine.compute_features(df)

        # Align to training features
        for feat in self.feature_names:
            if feat not in features.columns:
                features[feat] = 0.0
        features = features[self.feature_names]
        features = features.ffill().bfill().fillna(0.0)

        X_latest  = features.iloc[-1:].values.astype(np.float32)
        dmatrix   = xgb.DMatrix(X_latest, feature_names=self.feature_names)
        raw_proba = self.model.predict(dmatrix)[0]   # (3,)

        # Calibrate if available
        if self.calibrator.is_fitted:
            proba = self.calibrator.transform(raw_proba.reshape(1, -1))[0]
        else:
            proba = raw_proba

        pred_idx   = int(np.argmax(proba))
        pred_state = self.IDX_TO_CLASS[pred_idx]
        pred_label = self.CLASS_LABELS[pred_state]
        confidence = float(proba[pred_idx])

        result = {
            'predicted_state': pred_state,
            'predicted_label': pred_label,
            'confidence':      confidence
        }

        if return_proba:
            result['probabilities'] = {
                f'{self.IDX_TO_CLASS[i]}_{self.CLASS_LABELS[self.IDX_TO_CLASS[i]]}':
                    float(proba[i])
                for i in range(3)
            }

        # Log
        self.logger.info("=" * 70)
        self.logger.info("PREDICTION (3-CLASS)")
        self.logger.info(f"  State:      {pred_state} ({pred_label})")
        self.logger.info(f"  Confidence: {confidence:.2%}")
        if return_proba:
            self.logger.info("  Distribution:")
            for i in range(3):
                s = self.IDX_TO_CLASS[i]
                p = proba[i]
                bar = '█' * int(p * 40)
                self.logger.info(
                    f"    {s:>2} {self.CLASS_LABELS[s]:>10}: "
                    f"{p:>6.2%}  {bar}"
                )
        self.logger.info("=" * 70)

        return result


    # ═════════════════════════════════════════════════════════════════════════
    # PREDICT FOR ENSEMBLE (primary output for production)
    # ═════════════════════════════════════════════════════════════════════════

    def predict_for_ensemble(
        self,
        df: pd.DataFrame,
        min_confidence: float = 0.45,
        abstain_label:  int   = 0       # Return NEUTRAL when uncertain
    ) -> Dict:
        """
        Ensemble-ready prediction with continuous signal strength.

        Designed to be consumed directly by the ensemble layer that
        combines trend, volatility, and momentum signals.

        Output contract:
            signal_strength  float [-1, +1]  Continuous momentum score.
                             Positive = bullish, negative = bearish.
                             Magnitude encodes conviction level.
                             Formula: (P_bull − P_bear) × (1 − P_neutral)
                             Removes neutral uncertainty from the score.

            direction        int {-1, 0, 1}  Discrete momentum call.
                             Set to abstain_label when confidence < min_confidence.

            confidence       float [0,1]     Max class probability (calibrated).

            is_abstaining    bool            True when model is uncertain.
                             Ensemble should reduce weight for this component.

            probabilities    dict            Calibrated class probabilities.

            regime_context   dict            Key feature values for ensemble
                             to use in meta-decisions (e.g. reduce position
                             when RSI is diverging despite bullish signal).

            signal_strength is the most important field for the ensemble layer.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Features
        features = self.feature_engine.compute_features(df)
        for feat in self.feature_names:
            if feat not in features.columns:
                features[feat] = 0.0
        features = features[self.feature_names]
        features = features.ffill().bfill().fillna(0.0)

        X_latest  = features.iloc[-1:].values.astype(np.float32)
        dmatrix   = xgb.DMatrix(X_latest, feature_names=self.feature_names)
        raw_proba = self.model.predict(dmatrix)[0]

        # Calibrate
        if self.calibrator.is_fitted:
            proba = self.calibrator.transform(raw_proba.reshape(1, -1))[0]
        else:
            proba = raw_proba

        p_bear, p_neut, p_bull = float(proba[0]), float(proba[1]), float(proba[2])

        # Continuous signal strength: removes neutral uncertainty
        signal_strength = (p_bull - p_bear) * (1.0 - p_neut)

        pred_idx    = int(np.argmax(proba))
        raw_dir     = self.IDX_TO_CLASS[pred_idx]
        confidence  = float(proba[pred_idx])

        is_abstaining = confidence < min_confidence
        direction     = abstain_label if is_abstaining else raw_dir

        # Regime context: key features for ensemble meta-decisions
        last = features.iloc[-1]

        def _safe(col: str, default: float = 0.0) -> float:
            v = last.get(col, default)
            return float(v) if not pd.isna(v) else default

        regime_context = {
            'rsi_14':               _safe('rsi_14', 50.0),
            'rsi_divergence':       _safe('rsi_divergence'),
            'macd_histogram':       _safe('macd_histogram'),
            'macd_hist_slope':      _safe('macd_hist_slope'),
            'ma_alignment':         _safe('ma_alignment'),
            'signed_streak':        _safe('signed_streak'),
            'volume_ratio':         _safe('volume_ratio', 1.0),
            'atr_normalised':       _safe('atr_normalised'),
            'momentum_atr_ratio_1': _safe('momentum_atr_ratio_1'),
            'variance_ratio_4':     _safe('variance_ratio_4'),
            'obv_momentum_5':       _safe('obv_momentum_5'),
        }

        timestamp = (
            df.index[-1].isoformat()
            if hasattr(df.index[-1], 'isoformat')
            else str(df.index[-1])
        )

        result = {
            # ── Core ensemble inputs ──────────────────────────────────────
            'signal_strength':   signal_strength,   # Primary: use this in ensemble
            'direction':         direction,          # After abstain filter
            'confidence':        confidence,
            'is_abstaining':     is_abstaining,

            # ── Raw probabilities (calibrated) ────────────────────────────
            'probabilities': {
                'bearish': p_bear,
                'neutral': p_neut,
                'bullish': p_bull
            },

            # ── Context for ensemble meta-logic ───────────────────────────
            'regime_context': regime_context,

            # ── Metadata ──────────────────────────────────────────────────
            'predicted_label': self.CLASS_LABELS[direction],
            'raw_direction':   raw_dir,             # Before abstain filter
            'timestamp':       timestamp
        }

        self.logger.info("=" * 70)
        self.logger.info("ENSEMBLE PREDICTION (V4.0)")
        self.logger.info(f"  Signal strength:  {signal_strength:+.4f}")
        self.logger.info(f"  Direction:        {direction} ({self.CLASS_LABELS[direction]})")
        self.logger.info(f"  Confidence:       {confidence:.2%}")
        self.logger.info(f"  Abstaining:       {is_abstaining}")
        self.logger.info(f"  P(bear/neut/bull):{p_bear:.3f} / {p_neut:.3f} / {p_bull:.3f}")
        self.logger.info("=" * 70)

        return result


    # ═════════════════════════════════════════════════════════════════════════
    # SAVE
    # ═════════════════════════════════════════════════════════════════════════

    def save(
        self,
        model_path:    Optional[str] = None,
        metadata_path: Optional[str] = None,
        version:       Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save model + calibrator + metadata.

        File layout:
            trained/xgb_momentum_v{version}.json   ← XGBoost native model
            trained/calibrator_v{version}.pkl      ← ProbabilityCalibrator
            trained/metadata_v{version}.pkl        ← Feature names, hyperparams, etc.
        """
        if self.model is None:
            raise ValueError("No trained model to save")

        version = version or datetime.now().strftime('%Y%m%d_%H%M%S')

        model_path    = model_path or os.path.join(
            self.trained_dir, f'xgb_momentum_v{version}.json'
        )
        calibrator_path = os.path.join(
            self.trained_dir, f'calibrator_v{version}.pkl'
        )
        metadata_path = metadata_path or os.path.join(
            self.trained_dir, f'metadata_v{version}.pkl'
        )

        # XGBoost model
        self.model.save_model(model_path)
        self.logger.info(f"✓ XGBoost model saved: {model_path}")

        # Calibrator
        if self.calibrator.is_fitted:
            self.calibrator.save(calibrator_path)
            self.logger.info(f"✓ Calibrator saved:    {calibrator_path}")

        # Metadata
        metadata = {
            'version':          version,
            'saved_at':         datetime.now().isoformat(),
            'num_classes':      3,
            'feature_names':    self.feature_names,
            'n_features':       self.n_features,
            'feature_importance': self.feature_importance,
            'feature_stability': (
                self.feature_stability.to_dict()
                if self.feature_stability is not None else None
            ),
            'best_iteration':   self.best_iteration,
            'training_history': self.training_history,
            'calibrator_path':  calibrator_path if self.calibrator.is_fitted else None,
            'hyperparameters': {
                'max_depth':          self.max_depth,
                'learning_rate':      self.learning_rate,
                'n_estimators':       self.n_estimators,
                'subsample':          self.subsample,
                'colsample_bytree':   self.colsample_bytree,
                'colsample_bylevel':  self.colsample_bylevel,
                'colsample_bynode':   self.colsample_bynode,
                'gamma':              self.gamma,
                'reg_alpha':          self.reg_alpha,
                'reg_lambda':         self.reg_lambda,
                'min_child_weight':   self.min_child_weight,
                'max_delta_step':     self.max_delta_step,
                'random_state':       self.random_state
            },
            'class_mappings': {
                'CLASS_LABELS': self.CLASS_LABELS,
                'CLASS_TO_IDX': self.CLASS_TO_IDX,
                'IDX_TO_CLASS': self.IDX_TO_CLASS
            }
        }

        joblib.dump(metadata, metadata_path)
        self.logger.info(f"✓ Metadata saved:      {metadata_path}")

        self.logger.info("=" * 70)
        self.logger.info("SAVE SUMMARY")
        self.logger.info(f"  Version:    {version}")
        self.logger.info(f"  Features:   {self.n_features}")
        self.logger.info(f"  Classes:    3 (BEARISH | NEUTRAL | BULLISH)")
        self.logger.info(f"  Calibrated: {self.calibrator.is_fitted}")
        self.logger.info("=" * 70)

        return {
            'model_path':      model_path,
            'calibrator_path': calibrator_path,
            'metadata_path':   metadata_path,
            'version':         version
        }


    # ═════════════════════════════════════════════════════════════════════════
    # LOAD
    # ═════════════════════════════════════════════════════════════════════════

    def load(self, model_path: str, metadata_path: str) -> None:
        """
        Load model + calibrator + metadata.

        The calibrator is loaded automatically from the path stored in metadata
        (if it was saved alongside the model).
        """
        for p in [model_path, metadata_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"File not found: {p}")

        # XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.logger.info(f"✓ XGBoost model loaded: {model_path}")

        # Metadata
        metadata = joblib.load(metadata_path)

        self.feature_names    = metadata.get('feature_names')
        self.n_features       = metadata.get('n_features')
        self.feature_importance = metadata.get('feature_importance')
        self.best_iteration   = metadata.get('best_iteration')
        self.training_history = metadata.get('training_history', {})

        stability_dict = metadata.get('feature_stability')
        if stability_dict:
            self.feature_stability = pd.DataFrame(stability_dict)

        hyper = metadata.get('hyperparameters', {})
        for attr, val in hyper.items():
            setattr(self, attr, val)

        num_classes = metadata.get('num_classes', 3)
        if num_classes != 3:
            self.logger.warning(
                f"⚠ Loaded model has {num_classes} classes, expected 3"
            )

        # Calibrator (optional)
        cal_path = metadata.get('calibrator_path')
        if cal_path and os.path.exists(cal_path):
            self.calibrator = ProbabilityCalibrator()
            self.calibrator.load(cal_path)
            self.logger.info(f"✓ Calibrator loaded:    {cal_path}")
        else:
            self.logger.warning("  No calibrator found — predictions will be uncalibrated")

        self.logger.info("=" * 70)
        self.logger.info("LOAD SUMMARY")
        self.logger.info(f"  Model:      {model_path}")
        self.logger.info(f"  Features:   {self.n_features}")
        self.logger.info(f"  Classes:    {num_classes}")
        self.logger.info(f"  Best iter:  {self.best_iteration}")
        self.logger.info(f"  Calibrated: {self.calibrator.is_fitted}")
        self.logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT / SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )

    print("=" * 70)
    print("XGBoost Momentum Detector V4.0 — 3-Class Ensemble Component")
    print("=" * 70)
    print()
    print("Class mapping:")
    print("  −1  →  BEARISH")
    print("   0  →  NEUTRAL")
    print("  +1  →  BULLISH")
    print()
    print("Architecture:")
    print("  ├── PurgedTimeSeriesSplit   (walk-forward, growing window)")
    print("  ├── MomentumFeatureEngine   (12 blocks, zero lookahead)")
    print("  │     Block  1 : Rate of Change (6 horizons)")
    print("  │     Block  2 : Momentum Acceleration & Jerk")
    print("  │     Block  3 : RSI Family + Divergence")
    print("  │     Block  4 : MACD Momentum")
    print("  │     Block  5 : ATR-Normalised Momentum")
    print("  │     Block  6 : Volume-Price Momentum (OBV, Force, VWAP)")
    print("  │     Block  7 : Consecutive Bar Streak")
    print("  │     Block  8 : Candle Body & Shadow Analysis")
    print("  │     Block  9 : Moving Average Structure & Alignment")
    print("  │     Block 10 : Momentum Regime (Variance Ratio, Autocorr)")
    print("  │     Block 11 : Circular Session & Time Features")
    print("  │     Block 12 : Cross-Timeframe Momentum Context")
    print("  ├── Triple Barrier Labels   (symmetric ATR, entry on next Open)")
    print("  ├── ProbabilityCalibrator   (isotonic regression, one-vs-rest)")
    print("  └── predict_for_ensemble()  (signal_strength + abstain logic)")
    print()
    print("Key fixes vs V3.0:")
    print("  ✓ Feature engineering implemented (was empty)")
    print("  ✓ get_momentum_features() → MomentumFeatureEngine.compute_features()")
    print("  ✓ Symmetric barriers (was 2.0/1.5 → bearish-biased)")
    print("  ✓ Entry on next Open (not current Close)")
    print("  ✓ Single _convert_labels_to_indices() (eliminated dual-path bug)")
    print("  ✓ PurgedTimeSeriesSplit: growing window + embargo after val")
    print("  ✓ Three-way split: train | calib | val (not two-way)")
    print("  ✓ Post-hoc probability calibration")
    print("  ✓ predict_for_ensemble() with continuous signal_strength")
    print("  ✓ Stronger regularisation defaults (min_child_weight=20)")
    print("  ✓ colsample_bylevel + colsample_bynode added")
    print("  ✓ Centralised _build_xgb_params() — no param drift between methods")
    print("=" * 70)