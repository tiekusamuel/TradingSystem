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
    accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score,
    precision_score, recall_score, roc_auc_score, average_precision_score
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from features.features import VolatilityVolumeFeatureEngine


# ═══════════════════════════════════════════════════════════════════════════════
# PURGED TIME SERIES SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

class PurgedTimeSeriesSplit:
   
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
    Post-hoc probability calibration for binary XGBoost output.

    Why calibrate?
        XGBoost probabilities are often overconfident (peaked distributions).
        Downstream ensemble models that weight signals by probability need
        well-calibrated probabilities — otherwise the regime model will
        dominate inappropriately in high-confidence but wrong predictions.

    Method:
        Isotonic regression (non-parametric, no shape assumptions) or
        Platt scaling (sigmoid).
    """

    def __init__(self, method: str = 'isotonic'):
        """
        Args:
            method: 'isotonic' (recommended) or 'sigmoid' (Platt scaling)
        """
        if method not in ('isotonic', 'sigmoid'):
            raise ValueError(f"method must be 'isotonic' or 'sigmoid', got '{method}'")
        self.method      = method
        self.calibrator  = None
        self.is_fitted   = False

    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        y_prob: np.ndarray,   # (n_samples,) binary probabilities
        y_true: np.ndarray    # (n_samples,) binary labels {0, 1}
    ) -> 'ProbabilityCalibrator':
        """
        Fit calibrator on a dedicated calibration set.

        IMPORTANT: This set must be SEPARATE from both training and validation.
        A three-way split is required:
            train → fit XGBoost
            calib → fit this calibrator
            test  → final evaluation
        """
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_prob, y_true)
        else:
            self.calibrator = LogisticRegression(C=1.0, max_iter=1000)
            self.calibrator.fit(y_prob.reshape(-1, 1), y_true)

        self.is_fitted = True
        return self

    # ──────────────────────────────────────────────────────────────────────────

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply calibration.

        Args:
            y_prob: Raw XGBoost probabilities, shape (n_samples,)

        Returns:
            Calibrated probabilities, same shape
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted — call fit() first")

        if self.method == 'isotonic':
            return self.calibrator.predict(y_prob)
        else:
            return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]

    # ──────────────────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> np.ndarray:
        return self.fit(y_prob, y_true).transform(y_prob)

    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        joblib.dump({
            'calibrator': self.calibrator,
            'method': self.method,
            'is_fitted': self.is_fitted
        }, path)

    def load(self, path: str) -> 'ProbabilityCalibrator':
        data             = joblib.load(path)
        self.calibrator  = data['calibrator']
        self.method      = data['method']
        self.is_fitted   = data['is_fitted']
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class XGBoostRegimeDetector:
    """
    XGBoost Volatility-Volume Regime Detector V1.0 — Binary Classifier

    Predicts whether current market environment supports tradable expansion:
        0 = UNFAVORABLE (noisy, low-quality, weak participation)
        1 = FAVORABLE   (clean expansion, strong participation, sustainable)

    This is NOT a directional predictor.
    It's a market quality filter for momentum strategies.

    Purpose:
        ✓ Detect healthy expansion regimes
        ✓ Filter out noisy ranging markets
        ✓ Validate liquidity and participation
        ✓ Identify explosive continuation conditions
        ✓ Reduce false breakout entries

    Key components:
        ✓ VolatilityVolumeFeatureEngine (volatility + volume + interactions)
        ✓ Advanced expansion regime labels (not simple volatility spikes)
        ✓ Post-hoc probability calibration
        ✓ Ensemble-ready output (expansion_quality_prob + regime_strength)
        ✓ Strong regularization for noisy financial data
        ✓ Label quality diagnostics
        ✓ Per-fold calibration in CV
        ✓ SHAP analysis with label-feature coupling detection
        ✓ Out-of-time validation
    """

    # ── Class-level constants ─────────────────────────────────────────────────
    CLASS_LABELS  = {0: 'UNFAVORABLE', 1: 'FAVORABLE'}

    # ── Recommended hyperparameters for regime detection ─────────────────────
    DEFAULT_PARAMS = dict(
        max_depth          = 5,     # Slightly deeper: regime has complex interactions
        learning_rate      = 0.015, # Very slow learning
        n_estimators       = 2500,  # Large pool; early stopping finds optimum
        subsample          = 0.65,  # Aggressive row subsampling
        colsample_bytree   = 0.65,  # Aggressive column subsampling
        colsample_bylevel  = 0.8,   # Additional regularization
        colsample_bynode   = 0.8,   # Additional regularization
        gamma              = 0.3,   # Minimum loss reduction to split
        reg_alpha          = 0.5,   # L1 → feature sparsity
        reg_lambda         = 6.0,   # L2 → weight smoothing
        min_child_weight   = 25,    # Large leaves → avoid overfit
        max_delta_step     = 0,     # Not needed for binary
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
        self.feature_engine = VolatilityVolumeFeatureEngine()
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
       
        self.log_dir        = "models/regime_model/logs"
        self.trained_dir    = "models/regime_model/trained"
        self.plot_dir       = "models/regime_model/plots"

        for d in [ self.log_dir, self.trained_dir, self.plot_dir]:
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

        self.logger.info("XGBoost Regime Detector V1.0 initialised (BINARY)")
        self.logger.info(
            f"  depth={max_depth}, lr={learning_rate}, "
            f"trees={n_estimators}, min_child_weight={min_child_weight}"
        )


    # ═════════════════════════════════════════════════════════════════════════
    # LABEL CREATION — TRADABLE EXPANSION REGIME
    # ═════════════════════════════════════════════════════════════════════════

    def create_percentile_expansion_labels(
    self,
    df: pd.DataFrame,
    lookforward: int = 30,
    top_percentile: float = 0.25,  # Top 15% = favorable
    atr_period: int = 14,
    volume_window: int = 20,
    score_components: Dict[str, float] = None,
    ) -> pd.Series:
        """
        Percentile-based expansion regime labeling (ROBUST).
        
        Instead of hard thresholds, this labels the top N% of bars by
        expansion quality score. This guarantees balanced classes and
        adapts to market conditions.
        
        Expansion quality score combines:
            - Future ATR expansion (how much volatility increases)
            - Current volume strength (participation)
            - Future price movement (magnitude)
            - Movement sustainability (not immediately reversed)
        
        Args:
            df: OHLCV DataFrame with DatetimeIndex
            lookforward: Bars to look ahead for expansion measurement
            top_percentile: Fraction of top-scoring bars to label as FAVORABLE
                            (e.g., 0.15 = top 15%)
            atr_period: ATR calculation period
            volume_window: Window for volume normalization
            score_components: Custom weights for score components
                            Default: {'atr_exp': 0.4, 'volume': 0.3, 
                                    'price_move': 0.2, 'sustain': 0.1}
        
        Returns:
            pd.Series of labels {0, 1} with NaN for unusable rows
            
        Example:
            labels = model.create_percentile_expansion_labels(
                df,
                lookforward=20,
                top_percentile=0.15  # Top 15% favorable
            )
        """
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(c in df.columns for c in required):
            raise ValueError(f"DataFrame must contain {required}")
        
        self.logger.info("=" * 70)
        self.logger.info("Creating PERCENTILE-based expansion regime labels")
        self.logger.info("=" * 70)
        self.logger.info(f"  Lookforward:     {lookforward} bars")
        self.logger.info(f"  Top percentile:  {top_percentile:.1%}")
        self.logger.info(f"  ATR period:      {atr_period}")
        
        # Default score weights
        if score_components is None:
            score_components = {
                'atr_expansion': 0.40,    # Primary: volatility expansion
                'volume_strength': 0.30,  # Secondary: participation
                'price_movement': 0.20,   # Tertiary: actual price change
                'sustainability': 0.10,   # Quality: move sustains
            }
        
        self.logger.info("  Score weights:")
        for component, weight in score_components.items():
            self.logger.info(f"    {component:20s}: {weight:.2f}")
        
        # ══════════════════════════════════════════════════════════════════════
        # STEP 1: Compute base indicators
        # ══════════════════════════════════════════════════════════════════════
        
        # ATR (zero lookahead)
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(span=atr_period, min_periods=atr_period).mean()
        
        # Volume ratio (current vs average)
        volume_ma = df['Volume'].rolling(window=volume_window, min_periods=10).mean()
        volume_ratio = df['Volume'] / volume_ma.replace(0, np.nan)
        
        # Volume z-score
        volume_std = df['Volume'].rolling(window=volume_window, min_periods=10).std()
        volume_zscore = (df['Volume'] - volume_ma) / volume_std.replace(0, np.nan)
        
        # ══════════════════════════════════════════════════════════════════════
        # STEP 2: Compute expansion quality scores
        # ══════════════════════════════════════════════════════════════════════
        
        expansion_scores = []
        component_scores = {
            'atr_expansion': [],
            'volume_strength': [],
            'price_movement': [],
            'sustainability': [],
        }
        
        n_valid = 0
        
        for i in range(atr_period + volume_window, len(df) - lookforward):
            current_atr = atr.iloc[i]
            current_close = df['Close'].iloc[i]
            
            # Validation
            if pd.isna(current_atr) or current_atr <= 0:
                expansion_scores.append(np.nan)
                for key in component_scores:
                    component_scores[key].append(np.nan)
                continue
            
            if pd.isna(current_close) or current_close <= 0:
                expansion_scores.append(np.nan)
                for key in component_scores:
                    component_scores[key].append(np.nan)
                continue
            
            # ──────────────────────────────────────────────────────────────────
            # Future window
            # ──────────────────────────────────────────────────────────────────
            future_end = min(i + lookforward, len(df))
            future_high = df['High'].iloc[i+1:future_end].values
            future_low = df['Low'].iloc[i+1:future_end].values
            future_atr = atr.iloc[i+1:future_end].values
            
            if len(future_high) < lookforward * 0.8:  # Need at least 80% of lookforward
                expansion_scores.append(np.nan)
                for key in component_scores:
                    component_scores[key].append(np.nan)
                continue
            
            # ══════════════════════════════════════════════════════════════════
            # COMPONENT 1: ATR Expansion (0-1 normalized)
            # ══════════════════════════════════════════════════════════════════
            max_future_atr = np.max(future_atr)
            atr_expansion_ratio = max_future_atr / current_atr
            
            # Normalize: 1.0x = 0.0, 1.5x = 1.0, >1.5x = 1.0 (capped)
            atr_expansion_score = min((atr_expansion_ratio - 1.0) / 0.5, 1.0)
            atr_expansion_score = max(atr_expansion_score, 0.0)
            
            component_scores['atr_expansion'].append(atr_expansion_score)
            
            # ══════════════════════════════════════════════════════════════════
            # COMPONENT 2: Volume Strength (0-1 normalized)
            # ══════════════════════════════════════════════════════════════════
            vol_ratio = volume_ratio.iloc[i]
            vol_z = volume_zscore.iloc[i]
            
            if pd.isna(vol_ratio) or pd.isna(vol_z):
                volume_strength_score = 0.0
            else:
                # Combine ratio and z-score
                # Ratio: 1.0x = 0.0, 1.5x = 1.0
                ratio_score = min((vol_ratio - 1.0) / 0.5, 1.0)
                ratio_score = max(ratio_score, 0.0)
                
                # Z-score: -1 = 0.0, +2 = 1.0
                z_score = min((vol_z + 1.0) / 3.0, 1.0)
                z_score = max(z_score, 0.0)
                
                volume_strength_score = 0.6 * ratio_score + 0.4 * z_score
            
            component_scores['volume_strength'].append(volume_strength_score)
            
            # ══════════════════════════════════════════════════════════════════
            # COMPONENT 3: Price Movement (0-1 normalized)
            # ══════════════════════════════════════════════════════════════════
            price_high_move = np.max(future_high) - current_close
            price_low_move = current_close - np.min(future_low)
            max_price_move = max(price_high_move, price_low_move)
            price_move_pct = max_price_move / current_close
            
            # Normalize: 0% = 0.0, 2% = 1.0, >2% = 1.0 (capped)
            price_movement_score = min(price_move_pct / 0.02, 1.0)
            
            component_scores['price_movement'].append(price_movement_score)
            
            # ══════════════════════════════════════════════════════════════════
            # COMPONENT 4: Sustainability (0-1 normalized)
            # ══════════════════════════════════════════════════════════════════
            # Measure how much of the move is sustained vs reversed
            
            if price_high_move > price_low_move:
                # Bullish move
                peak = np.max(future_high)
                peak_idx = np.argmax(future_high)
                
                # What's the lowest point after the peak?
                if peak_idx < len(future_low) - 1:
                    subsequent_low = np.min(future_low[peak_idx:])
                    retracement = (peak - subsequent_low) / max(peak - current_close, 0.0001)
                else:
                    retracement = 0.0
            else:
                # Bearish move
                trough = np.min(future_low)
                trough_idx = np.argmin(future_low)
                
                # What's the highest point after the trough?
                if trough_idx < len(future_high) - 1:
                    subsequent_high = np.max(future_high[trough_idx:])
                    retracement = (subsequent_high - trough) / max(current_close - trough, 0.0001)
                else:
                    retracement = 0.0
            
            # Sustainability: low retracement = high score
            # 0% retracement = 1.0, 50% retracement = 0.5, 100% = 0.0
            sustainability_score = max(1.0 - retracement, 0.0)
            
            component_scores['sustainability'].append(sustainability_score)
            
            # ══════════════════════════════════════════════════════════════════
            # COMBINED SCORE (weighted sum)
            # ══════════════════════════════════════════════════════════════════
            combined_score = (
                score_components['atr_expansion'] * atr_expansion_score +
                score_components['volume_strength'] * volume_strength_score +
                score_components['price_movement'] * price_movement_score +
                score_components['sustainability'] * sustainability_score
            )
            
            expansion_scores.append(combined_score)
            n_valid += 1
        
        # ══════════════════════════════════════════════════════════════════════
        # STEP 3: Pad with NaN
        # ══════════════════════════════════════════════════════════════════════
        
        pad_start = atr_period + volume_window
        pad_end = lookforward
        
        expansion_scores = (
            [np.nan] * pad_start + 
            expansion_scores + 
            [np.nan] * pad_end
        )
        
        for key in component_scores:
            component_scores[key] = (
                [np.nan] * pad_start +
                component_scores[key] +
                [np.nan] * pad_end
            )
        
        expansion_series = pd.Series(expansion_scores, index=df.index)
        
        # ══════════════════════════════════════════════════════════════════════
        # STEP 4: Convert to percentile-based labels
        # ══════════════════════════════════════════════════════════════════════
        
        valid_mask = ~expansion_series.isna()
        valid_scores = expansion_series[valid_mask]
        
        if len(valid_scores) == 0:
            self.logger.error("No valid scores computed!")
            return pd.Series(np.nan, index=df.index)
        
        # Compute percentile threshold
        threshold = valid_scores.quantile(1.0 - top_percentile)
        
        # Create binary labels
        labels = pd.Series(np.nan, index=df.index)
        labels[valid_mask] = (expansion_series[valid_mask] >= threshold).astype(int)
        
        # ══════════════════════════════════════════════════════════════════════
        # STEP 5: Statistics and diagnostics
        # ══════════════════════════════════════════════════════════════════════
        
        valid_labels = labels.dropna()
        total = len(valid_labels)
        
        if total > 0:
            n_favorable = (valid_labels == 1).sum()
            n_unfavorable = (valid_labels == 0).sum()
            pct_favorable = n_favorable / total * 100
            
            self.logger.info("\n" + "─" * 70)
            self.logger.info("LABEL STATISTICS")
            self.logger.info("─" * 70)
            self.logger.info(f"  Total valid:     {total:>8}")
            self.logger.info(f"  UNFAVORABLE (0): {n_unfavorable:>8} ({100-pct_favorable:>5.1f}%)")
            self.logger.info(f"  FAVORABLE (1):   {n_favorable:>8} ({pct_favorable:>5.1f}%)")
            self.logger.info(f"  NaN rows:        {labels.isna().sum():>8}")
            
            self.logger.info("\n" + "─" * 70)
            self.logger.info("SCORE DISTRIBUTION")
            self.logger.info("─" * 70)
            self.logger.info(f"  Threshold:       {threshold:.4f} ({(1-top_percentile)*100:.0f}th percentile)")
            self.logger.info(f"  Score range:     [{valid_scores.min():.4f}, {valid_scores.max():.4f}]")
            self.logger.info(f"  Mean score:      {valid_scores.mean():.4f}")
            self.logger.info(f"  Median score:    {valid_scores.median():.4f}")
            self.logger.info(f"  Std score:       {valid_scores.std():.4f}")
            
            # Component statistics
            self.logger.info("\n" + "─" * 70)
            self.logger.info("COMPONENT STATISTICS (for FAVORABLE samples)")
            self.logger.info("─" * 70)
            
            favorable_mask = (labels == 1) & valid_mask
            
            for component_name, scores in component_scores.items():
                scores_series = pd.Series(scores, index=df.index)
                fav_scores = scores_series[favorable_mask]
                
                if len(fav_scores) > 0:
                    self.logger.info(
                        f"  {component_name:20s}: "
                        f"mean={fav_scores.mean():.3f}, "
                        f"median={fav_scores.median():.3f}"
                    )
            
            # Quality check
            self.logger.info("\n" + "─" * 70)
            self.logger.info("QUALITY CHECK")
            self.logger.info("─" * 70)
            
            if pct_favorable < 5:
                self.logger.warning(
                    f"  ⚠ Very low favorable rate ({pct_favorable:.1f}%).\n"
                    f"     Consider increasing top_percentile (currently {top_percentile:.2f})"
                )
            elif pct_favorable > 30:
                self.logger.warning(
                    f"  ⚠ High favorable rate ({pct_favorable:.1f}%).\n"
                    f"     Consider decreasing top_percentile for higher quality"
                )
            else:
                self.logger.info(f"  ✓ Favorable rate acceptable ({pct_favorable:.1f}%)")
            
            # Check if threshold is meaningful
            if threshold < 0.1:
                self.logger.warning(
                    "  ⚠ Low score threshold — even weak expansions may be labeled favorable"
                )
            elif threshold > 0.7:
                self.logger.info(
                    "  ✓ High score threshold — only strong expansions labeled favorable"
                )
        else:
            self.logger.warning("⚠ No valid labels generated!")
        
        self.logger.info("=" * 70)
        
        return labels


   


    # ═════════════════════════════════════════════════════════════════════════
    # LABEL QUALITY DIAGNOSTICS (NEW)
    # ═════════════════════════════════════════════════════════════════════════

    def diagnose_label_quality(
        self,
        labels: pd.Series,
        plot: bool = True
    ) -> Dict:
        """
        Diagnose label temporal structure and clustering.
        
        Red flags:
            - Long favorable streaks (regime continuation, not drivers)
            - High autocorrelation (model learns "stay in regime")
            - Uneven distribution across time (cluster detection)
        
        Returns:
            Dict with diagnostic metrics
        """
        self.logger.info("=" * 70)
        self.logger.info("LABEL QUALITY DIAGNOSTICS")
        self.logger.info("=" * 70)
        
        valid_labels = labels.dropna()
        
        # ─── Streak Analysis ─────────────────────────────────────────────────
        # Group consecutive identical labels
        label_groups = (valid_labels != valid_labels.shift()).cumsum()
        streak_sizes = valid_labels.groupby(label_groups).agg(['size', 'first'])
        
        favorable_streaks = streak_sizes[streak_sizes['first'] == 1]['size']
        unfavorable_streaks = streak_sizes[streak_sizes['first'] == 0]['size']
        
        self.logger.info("\n📊 STREAK ANALYSIS:")
        self.logger.info(f"  Favorable streaks:")
        self.logger.info(f"    Mean:   {favorable_streaks.mean():.1f} bars")
        self.logger.info(f"    Median: {favorable_streaks.median():.1f} bars")
        self.logger.info(f"    Max:    {favorable_streaks.max():.0f} bars")
        self.logger.info(f"    Std:    {favorable_streaks.std():.1f} bars")
        
        self.logger.info(f"  Unfavorable streaks:")
        self.logger.info(f"    Mean:   {unfavorable_streaks.mean():.1f} bars")
        self.logger.info(f"    Median: {unfavorable_streaks.median():.1f} bars")
        
        # ─── Autocorrelation ─────────────────────────────────────────────────
        # High autocorrelation = model learns "regime persistence" not "drivers"
        autocorr_lag1 = valid_labels.autocorr(lag=1)
        autocorr_lag5 = valid_labels.autocorr(lag=5)
        autocorr_lag10 = valid_labels.autocorr(lag=10)
        
        self.logger.info("\n📈 TEMPORAL AUTOCORRELATION:")
        self.logger.info(f"  Lag 1:  {autocorr_lag1:.3f}")
        self.logger.info(f"  Lag 5:  {autocorr_lag5:.3f}")
        self.logger.info(f"  Lag 10: {autocorr_lag10:.3f}")
        
        # ─── Regime Clustering Risk ─────────────────────────────────────────
        if favorable_streaks.mean() > 10:
            self.logger.warning("  ⚠ LONG FAVORABLE STREAKS")
            self.logger.warning("    → Model may learn 'regime continuation' not 'causal drivers'")
            clustering_risk = "HIGH"
        elif favorable_streaks.mean() > 5:
            self.logger.warning("  ⚠ MODERATE FAVORABLE STREAKS")
            clustering_risk = "MODERATE"
        else:
            self.logger.info("  ✓ Favorable regimes are discrete events")
            clustering_risk = "LOW"
        
        if autocorr_lag1 > 0.7:
            self.logger.warning("  ⚠ HIGH LAG-1 AUTOCORRELATION")
            self.logger.warning("    → Labels are highly persistent (regime detection)")
        elif autocorr_lag1 > 0.5:
            self.logger.warning("  ⚠ MODERATE AUTOCORRELATION")
        else:
            self.logger.info("  ✓ Low autocorrelation (event-driven)")
        
        # ─── Distribution Over Time ─────────────────────────────────────────
        # Check if favorable regimes are evenly distributed
        favorable_pct = (valid_labels == 1).sum() / len(valid_labels)
        
        # Split time into 10 bins, check variance of favorable %
        n_bins = 10
        bin_size = len(valid_labels) // n_bins
        bin_favorable_pcts = []
        
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(valid_labels)
            bin_labels = valid_labels.iloc[start:end]
            bin_fav_pct = (bin_labels == 1).sum() / len(bin_labels) if len(bin_labels) > 0 else 0
            bin_favorable_pcts.append(bin_fav_pct)
        
        time_variance = np.std(bin_favorable_pcts)
        
        self.logger.info("\n⏱ TEMPORAL DISTRIBUTION:")
        self.logger.info(f"  Overall favorable %:  {favorable_pct:.1%}")
        self.logger.info(f"  Std across time bins: {time_variance:.3f}")
        
        if time_variance > 0.10:
            self.logger.warning("  ⚠ HIGH VARIANCE ACROSS TIME")
            self.logger.warning("    → Regimes cluster in certain periods")
        else:
            self.logger.info("  ✓ Stable distribution over time")
        
        # ─── Recommendation ──────────────────────────────────────────────────
        self.logger.info("\n💡 RECOMMENDATION:")
        
        if clustering_risk == "HIGH" and autocorr_lag1 > 0.7:
            self.logger.warning("  ⚠ CRITICAL: Labels show strong regime clustering")
            self.logger.warning("     Consider:")
            self.logger.warning("       1. Relax label conditions (increase favorable %)")
            self.logger.warning("       2. Add 'transition' labels (compression→expansion)")
            self.logger.warning("       3. Use regime-change prediction (not regime state)")
            recommendation = "RELAX_CONDITIONS"
        elif clustering_risk == "MODERATE":
            self.logger.info("  ⚠ Acceptable clustering, but monitor precision stability")
            recommendation = "MONITOR"
        else:
            self.logger.info("  ✓ Label quality is acceptable")
            recommendation = "OK"
        
        self.logger.info("=" * 70)
        
        # ─── Plotting ────────────────────────────────────────────────────────
        if plot:
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            
            # Plot 1: Streak distribution
            axes[0].hist(favorable_streaks, bins=30, alpha=0.7, color='green', 
                         label='Favorable', edgecolor='black')
            axes[0].hist(unfavorable_streaks, bins=30, alpha=0.7, color='red',
                         label='Unfavorable', edgecolor='black')
            axes[0].axvline(favorable_streaks.mean(), color='darkgreen', 
                           linestyle='--', label=f'Fav Mean: {favorable_streaks.mean():.1f}')
            axes[0].set_xlabel('Streak Length (bars)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Label Streak Distribution')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # Plot 2: Autocorrelation
            from pandas.plotting import autocorrelation_plot
            autocorrelation_plot(valid_labels, ax=axes[1])
            axes[1].set_title('Label Autocorrelation')
            axes[1].grid(alpha=0.3)
            
            # Plot 3: Temporal distribution
            axes[2].bar(range(n_bins), bin_favorable_pcts, alpha=0.7, color='blue')
            axes[2].axhline(favorable_pct, color='red', linestyle='--', 
                           label=f'Overall: {favorable_pct:.1%}')
            axes[2].set_xlabel('Time Bin')
            axes[2].set_ylabel('Favorable %')
            axes[2].set_title('Favorable Regime % Across Time')
            axes[2].legend()
            axes[2].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'label_quality_diagnostics.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"\n✓ Diagnostic plots saved to {self.plot_dir}")
        
        return {
            'favorable_streak_mean': favorable_streaks.mean(),
            'favorable_streak_median': favorable_streaks.median(),
            'favorable_streak_max': favorable_streaks.max(),
            'autocorr_lag1': autocorr_lag1,
            'autocorr_lag5': autocorr_lag5,
            'autocorr_lag10': autocorr_lag10,
            'temporal_variance': time_variance,
            'clustering_risk': clustering_risk,
            'recommendation': recommendation,
        }


    # ═════════════════════════════════════════════════════════════════════════
    # SAMPLE WEIGHTS
    # ═════════════════════════════════════════════════════════════════════════

    def compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Balanced sample weights for class imbalance.

        weight[i] = n_samples / (n_classes × count_of_class[y[i]])

        Args:
            y: Binary labels {0, 1}

        Returns:
            Float array of per-sample weights
        """
        y = y.astype(int)
        unique_classes = np.unique(y)
        n_samples  = len(y)
        n_classes  = 2
        class_counts = np.bincount(y, minlength=2)

        class_weights = np.zeros(2)
        for c in unique_classes:
            if class_counts[c] > 0:
                class_weights[c] = n_samples / (n_classes * class_counts[c])

        sample_weights = class_weights[y]

        self.logger.info("Sample weights (balanced):")
        for c in range(2):
            if class_counts[c] > 0:
                pct = class_counts[c] / n_samples * 100
                self.logger.info(
                    f"  Class {c} ({self.CLASS_LABELS[c]:>12}): "
                    f"n={class_counts[c]:>6} ({pct:>5.1f}%), "
                    f"weight={class_weights[c]:.4f}"
                )

        return sample_weights

    
    # ═════════════════════════════════════════════════════════════════════════
    # XGBOOST PARAMETER DICT (centralised)
    # ═════════════════════════════════════════════════════════════════════════

    def _build_xgb_params(self, verbosity: int = 0) -> Dict:
       
        return {
            'objective':         'binary:logistic',
            'eval_metric':       'logloss',
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
        Compute all evaluation metrics for binary classification.
        
        Focus metrics for regime detection:
            - Precision: avoid false favorable signals
            - Recall: catch most true favorable regimes
            - F1: balance
            - ROC-AUC: ranking quality
            - Average Precision: precision-recall curve area
        """
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Handle edge cases
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except:
            roc_auc = 0.5

        try:
            avg_precision = average_precision_score(y_true, y_pred_proba)
        except:
            avg_precision = 0.0

        return {
            f'{prefix}log_loss':        log_loss(y_true, y_pred_proba),
            f'{prefix}accuracy':        accuracy_score(y_true, y_pred),
            f'{prefix}balanced_acc':    balanced_accuracy_score(y_true, y_pred),
            f'{prefix}precision':       precision_score(y_true, y_pred, zero_division=0),
            f'{prefix}recall':          recall_score(y_true, y_pred, zero_division=0),
            f'{prefix}f1':              f1_score(y_true, y_pred, zero_division=0),
            f'{prefix}mcc':             matthews_corrcoef(y_true, y_pred),
            f'{prefix}roc_auc':         roc_auc,
            f'{prefix}avg_precision':   avg_precision,
        }


    # ═════════════════════════════════════════════════════════════════════════
    # TRAIN
    # ═════════════════════════════════════════════════════════════════════════

    def train(
        self,
        df: pd.DataFrame,
        validation_split:       float = 0.2,
        calibration_split:      float = 0.1,
        purge_gap:              Optional[int] = None,
        use_sample_weights:     bool  = True,
        early_stopping_rounds:  int   = 75,
        verbose_eval:           int   = 50,
        fit_calibrator:         bool  = True,
        num_boost_round:        Optional[int] = None,
        diagnose_labels:        bool  = True,   
        **label_kwargs
    ) -> Dict:
        """
        Train XGBoost regime detector with zero data leakage.

        Data split order (chronological):
            [─────── TRAIN ───────][── CALIB ──][── VAL ──]
               (1 − val − calib)      (calib)     (val)
            purge gaps are applied between each adjacent pair.

        Args:
            df:                   OHLCV DataFrame with DatetimeIndex
            validation_split:     Fraction for validation set
            calibration_split:    Fraction for calibration set
            purge_gap:            Bars between train/calib/val (auto = lookforward)
            use_sample_weights:   Apply balanced class weights during training
            early_stopping_rounds:Patience for early stopping
            verbose_eval:         Logging frequency (every N rounds)
            fit_calibrator:       Fit ProbabilityCalibrator on calibration set
            num_boost_round:      Override n_estimators if set
            drop_composite_scores: Drop handcrafted composite features
            diagnose_labels:      Run label quality diagnostics
            **label_kwargs:       Passed to create_expansion_regime_labels()

        Returns:
            Dict of training results and metrics
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        self.logger.info("=" * 70)
        self.logger.info(f"Training XGBoost Regime Detector V1.0 | rows={len(df)}")
        self.logger.info("=" * 70)

        # ─── Step 1: Feature Engineering ────────────────────────────────
        self.logger.info("Step 1/10 | Feature engineering...")
        df_feat = self.feature_engine.compute_features(df)
          
        # ─── Step 2: Labels ──────────────────────────────────────────────
        self.logger.info("Step 2/10 | Creating expansion regime labels...")
        labels = self.create_percentile_expansion_labels(df, **label_kwargs)

        # ─── Step 3: Label Diagnostics (NEW) ─────────────────────────────
        if diagnose_labels:
            self.logger.info("Step 3/10 | Diagnosing label quality...")
            label_diagnostics = self.diagnose_label_quality(labels, plot=True)
            
            if label_diagnostics['recommendation'] == 'RELAX_CONDITIONS':
                raise ValueError(
                    "Label quality unacceptable for production. "
                    "Adjust label thresholds and retry."
                )
        else:
            self.logger.info("Step 3/10 | Skipping label diagnostics")

        # ─── Step 4: Remove invalid rows ────────────────────────────────
        self.logger.info("Step 4/10 | Removing invalid rows...")
        valid_mask  = ~labels.isna()
        df_feat     = df_feat[valid_mask].copy()
        labels      = labels[valid_mask].copy()
        self.logger.info(f"  Valid rows: {len(df_feat)}")

        # ─── Step 5: Feature columns ─────────────────────────────────────
        self.logger.info("Step 5/10 | Selecting feature columns...")
        exclude_cols = {'Open', 'High', 'Low', 'Close', 'Volume',
                        'Time', 'Date', 'Datetime'}
        feature_cols = [c for c in df_feat.columns if c not in exclude_cols]
        if not feature_cols:
            raise ValueError("No feature columns found after exclusion")
        df_model = df_feat[feature_cols].copy()
        self.logger.info(f"  Features: {len(feature_cols)}")

        # ─── Step 6: Purged three-way split ──────────────────────────────
        self.logger.info("Step 6/10 | Purged three-way chronological split...")

        if purge_gap is None:
            purge_gap = label_kwargs.get('lookforward', 30)

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

        # ─── Step 7: Prepare arrays ──────────────────────────────────────
        self.logger.info("Step 7/10 | Preparing arrays...")
        self.feature_names = None
        self.n_features    = None

        X_train = self._prepare_X(train_data.copy(), fit_feature_names=True)
        X_calib = self._prepare_X(calib_data.copy())
        X_val   = self._prepare_X(val_data.copy())

        y_train = train_labels.reset_index(drop=True).values.astype(int)
        y_calib = calib_labels.reset_index(drop=True).values.astype(int)
        y_val   = val_labels.reset_index(drop=True).values.astype(int)

        # ─── Step 8: Sample weights ──────────────────────────────────────
        self.logger.info("Step 8/10 | Computing sample weights...")
        sample_weights = self.compute_sample_weights(y_train) if use_sample_weights else None

        # ─── Step 9: Build DMatrix & Train ───────────────────────────────
        self.logger.info("Step 9/10 | Building DMatrix and training...")
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

        # ─── Step 10: Calibration ────────────────────────────────────────
        if fit_calibrator:
            self.logger.info("Step 10/10 | Fitting probability calibrator...")
            calib_proba = self.model.predict(dcalib)
            self.calibrator.fit(calib_proba, y_calib)
            self.logger.info("  Calibrator fitted (isotonic regression)")
        else:
            self.logger.info("Step 10/10 | Skipping calibration (fit_calibrator=False)")

        # ─── Feature importance ──────────────────────────────────────────
        self.feature_importance = self.model.get_score(importance_type='gain')
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importance.items()
        ]).sort_values('importance', ascending=False)

        imp_path = os.path.join(self.log_dir, 'feature_importance.csv')
        importance_df.to_csv(imp_path, index=False)

        self.logger.info("\nTop 20 features (by gain):")
        for _, row in importance_df.head(20).iterrows():
            self.logger.info(f"  {row['feature']:40s}: {row['importance']:.2f}")

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

        # Store for OOT comparison
        self.training_history['val_precision'] = val_metrics['val_precision']
        self.training_history['val_recall'] = val_metrics['val_recall']
        self.training_history['val_f1'] = val_metrics['val_f1']

        # ─── Confusion matrix plot ────────────────────────────────────────
        target_names = [self.CLASS_LABELS[i] for i in range(2)]
        cm = confusion_matrix(y_val, (val_proba_cal >= 0.5).astype(int))

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix — Validation Set (Regime Detector V1.0)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'confusion_matrix.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # ─── Probability distribution plot ───────────────────────────────
        plt.figure(figsize=(10, 6))
        plt.hist(val_proba_cal[y_val == 0], bins=50, alpha=0.6, label='Unfavorable', color='red')
        plt.hist(val_proba_cal[y_val == 1], bins=50, alpha=0.6, label='Favorable', color='green')
        plt.axvline(0.5, color='black', linestyle='--', label='Decision Threshold')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Probability Distribution by True Class')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'probability_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # ─── Summary log ─────────────────────────────────────────────────
        self.logger.info("=" * 70)
        self.logger.info("TRAINING COMPLETE (BINARY REGIME DETECTOR)")
        self.logger.info("=" * 70)
        self.logger.info(f"  Best iteration: {self.best_iteration}")
        self.logger.info("")
        self.logger.info("  ✅ PRIMARY METRICS (focus here):")
        self.logger.info(f"     Precision  — train: {train_metrics['train_precision']:.4f}  "
                         f"val: {val_metrics['val_precision']:.4f}")
        self.logger.info(f"     Recall     — train: {train_metrics['train_recall']:.4f}  "
                         f"val: {val_metrics['val_recall']:.4f}")
        self.logger.info(f"     F1         — train: {train_metrics['train_f1']:.4f}  "
                         f"val: {val_metrics['val_f1']:.4f}")
        self.logger.info(f"     ROC-AUC    — train: {train_metrics['train_roc_auc']:.4f}  "
                         f"val: {val_metrics['val_roc_auc']:.4f}")
        self.logger.info(f"     Avg Prec   — train: {train_metrics['train_avg_precision']:.4f}  "
                         f"val: {val_metrics['val_avg_precision']:.4f}")
        self.logger.info("")
        self.logger.info("  Secondary:")
        self.logger.info(f"     MCC        — train: {train_metrics['train_mcc']:.4f}  "
                         f"val: {val_metrics['val_mcc']:.4f}")
        self.logger.info(f"     Bal. Acc   — train: {train_metrics['train_balanced_acc']:.4f}  "
                         f"val: {val_metrics['val_balanced_acc']:.4f}")
        self.logger.info(f"     Log loss   — train: {train_metrics['train_log_loss']:.4f}  "
                         f"val: {val_metrics['val_log_loss']:.4f}")

        precision_gap = train_metrics['train_precision'] - val_metrics['val_precision']
        status  = " Possible overfitting" if precision_gap > 0.15 else " Healthy"
        self.logger.info(f"\n  Overfit check (Precision gap = {precision_gap:.4f}): {status}")
        
        # Regime quality interpretation
        val_precision = val_metrics['val_precision']
        val_recall    = val_metrics['val_recall']
        
        self.logger.info("\n  📊 REGIME QUALITY INTERPRETATION:")
        if val_precision >= 0.7:
            self.logger.info("     ✓ HIGH precision: Most favorable signals are genuine")
        elif val_precision >= 0.5:
            self.logger.info("     ⚠ MODERATE precision: Some false favorable signals")
        else:
            self.logger.info("     ✗ LOW precision: Many false favorable signals — tighten thresholds")
            
        if val_recall >= 0.6:
            self.logger.info("     ✓ GOOD recall: Catching most expansion regimes")
        elif val_recall >= 0.4:
            self.logger.info("     ⚠ MODERATE recall: Missing some expansion regimes")
        else:
            self.logger.info("     ✗ LOW recall: Missing many regimes — relax thresholds")

        self.logger.info("=" * 70)

        # Classification report
        class_report = classification_report(
            y_val,
            (val_proba_cal >= 0.5).astype(int),
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
        
        if diagnose_labels:
            results['label_diagnostics'] = label_diagnostics
        
        return results


    # ═════════════════════════════════════════════════════════════════════════
    # CROSS-VALIDATE (UPDATED WITH PER-FOLD CALIBRATION)
    # ═════════════════════════════════════════════════════════════════════════

    def cross_validate(
        self,
        df: pd.DataFrame,
        n_splits:           int   = 5,
        purge_gap:          int   = 30,
        embargo:            int   = 0,
        use_sample_weights: bool  = True,
        early_stopping_rounds: int = 75,
        num_boost_round:    Optional[int] = None,
        calibrate_per_fold: bool  = True,  # NEW
        drop_composite_scores: bool = False,  # NEW
        **label_kwargs
    ) -> Dict:
        """
        Purged walk-forward cross-validation with per-fold calibration.

        Args:
            df:                   OHLCV DataFrame with DatetimeIndex
            n_splits:             Number of CV folds
            purge_gap:            Bars purged between train and val per fold
            embargo:              Additional bars excluded after val per fold
            use_sample_weights:   Balanced class weighting
            early_stopping_rounds:Early stopping patience
            num_boost_round:      Override n_estimators if set
            calibrate_per_fold:   Fit calibrator per fold (recommended)
            drop_composite_scores: Drop handcrafted features
            **label_kwargs:       Passed to label creation

        Returns:
            Summary dict with per-fold metrics and feature stability DataFrame
        """
        self.logger.info("=" * 70)
        self.logger.info(f"Purged {n_splits}-fold cross-validation (V1.0)")
        self.logger.info(f"  purge_gap={purge_gap}, embargo={embargo}")
        self.logger.info(f"  calibrate_per_fold={calibrate_per_fold}")
        self.logger.info("=" * 70)

        # ─── Feature engineering & labelling (done ONCE for all folds) ───
        self.logger.info("Engineering features (all folds)...")
        df_feat = self.feature_engine.compute_features(
            df,
            drop_composite_scores=drop_composite_scores
        )

        self.logger.info("Creating expansion regime labels (all folds)...")
        labels = self.create_expansion_regime_labels(df, **label_kwargs)

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

            # Labels
            y_tr  = labels.iloc[tr_idx].reset_index(drop=True).values.astype(int)
            y_val = labels.iloc[val_idx].reset_index(drop=True).values.astype(int)

            # ══════════════════════════════════════════════════════════════
            # PER-FOLD CALIBRATION (NEW)
            # ══════════════════════════════════════════════════════════════
            if calibrate_per_fold and len(tr_idx) > 200:
                # Hold out 20% of training set for calibration
                calib_size = int(len(X_tr_arr) * 0.2)
                
                X_train_sub = X_tr_arr[:-calib_size]
                X_calib_sub = X_tr_arr[-calib_size:]
                y_train_sub = y_tr[:-calib_size]
                y_calib_sub = y_tr[-calib_size:]
                
                # Recompute weights for sub-training set
                if use_sample_weights:
                    sw_sub = self.compute_sample_weights(y_train_sub)
                else:
                    sw_sub = None
                
                # DMatrix for sub-training
                dtrain_sub = xgb.DMatrix(
                    X_train_sub, label=y_train_sub,
                    weight=sw_sub,
                    feature_names=self.feature_names
                )
                dcalib_sub = xgb.DMatrix(
                    X_calib_sub, label=y_calib_sub,
                    feature_names=self.feature_names
                )
                
                dtrain_for_eval = dtrain_sub
            else:
                # Use full training set (no calibration)
                sw = self.compute_sample_weights(y_tr) if use_sample_weights else None
                dtrain_sub = xgb.DMatrix(
                    X_tr_arr, label=y_tr,
                    weight=sw,
                    feature_names=self.feature_names
                )
                dtrain_for_eval = dtrain_sub
                dcalib_sub = None

            # Validation DMatrix
            dval = xgb.DMatrix(
                X_val_arr, label=y_val,
                feature_names=self.feature_names
            )

            # Train
            evals_result = {}
            fold_model = xgb.train(
                params,
                dtrain_sub,
                num_boost_round       = num_boost_round or self.n_estimators,
                evals                 = [(dtrain_for_eval, 'train'), (dval, 'val')],
                early_stopping_rounds = early_stopping_rounds,
                evals_result          = evals_result,
                verbose_eval          = False
            )

            # ══════════════════════════════════════════════════════════════
            # Apply per-fold calibration
            # ══════════════════════════════════════════════════════════════
            if calibrate_per_fold and dcalib_sub is not None:
                calib_proba_raw = fold_model.predict(dcalib_sub)
                
                fold_calibrator = ProbabilityCalibrator(method='isotonic')
                fold_calibrator.fit(calib_proba_raw, y_calib_sub)
                
                # Calibrate validation probabilities
                val_proba_raw = fold_model.predict(dval)
                val_proba = fold_calibrator.transform(val_proba_raw)
                
                self.logger.debug(f"  Calibrator fitted on {len(y_calib_sub)} samples")
            else:
                val_proba = fold_model.predict(dval)

            # Feature importance
            fold_importances.append(fold_model.get_score(importance_type='gain'))

            # Metrics (now using calibrated probabilities)
            fold_metrics = self._evaluate(y_val, val_proba, prefix='val_')

            for k, v in fold_metrics.items():
                cv_results[k].append(v)
            cv_results['best_iterations'].append(fold_model.best_iteration)

            self.logger.info(
                f"  Precision={fold_metrics['val_precision']:.4f}  "
                f"Recall={fold_metrics['val_recall']:.4f}  "
                f"F1={fold_metrics['val_f1']:.4f}  "
                f"ROC-AUC={fold_metrics['val_roc_auc']:.4f}  "
                f"{'[CAL]' if calibrate_per_fold else '[RAW]'}"
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

        self.logger.info("Top 15 most stable features:")
        for _, row in stability_df.head(15).iterrows():
            self.logger.info(
                f"  {row['feature']:40s}: "
                f"stability={row['stability']:.3f}, "
                f"mean_rank={row['mean_rank']:.1f}, "
                f"std_rank={row['std_rank']:.1f}"
            )

        # ─── Summary statistics ───────────────────────────────────────────
        metric_keys = [
            'val_log_loss', 'val_accuracy', 'val_balanced_acc',
            'val_precision', 'val_recall', 'val_f1', 'val_mcc',
            'val_roc_auc', 'val_avg_precision'
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
        self.logger.info("CROSS-VALIDATION SUMMARY (BINARY REGIME DETECTOR)")
        self.logger.info(f"{'='*70}")
        self.logger.info("  ✅ PRIMARY METRICS:")
        self.logger.info(
            f"     Precision:    {summary['mean_val_precision']:.4f}"
            f" ± {summary['std_val_precision']:.4f}"
        )
        self.logger.info(
            f"     Recall:       {summary['mean_val_recall']:.4f}"
            f" ± {summary['std_val_recall']:.4f}"
        )
        self.logger.info(
            f"     F1:           {summary['mean_val_f1']:.4f}"
            f" ± {summary['std_val_f1']:.4f}"
        )
        self.logger.info(
            f"     ROC-AUC:      {summary['mean_val_roc_auc']:.4f}"
            f" ± {summary['std_val_roc_auc']:.4f}"
        )
        self.logger.info(
            f"     Avg Prec:     {summary['mean_val_avg_precision']:.4f}"
            f" ± {summary['std_val_avg_precision']:.4f}"
        )
        self.logger.info("  Secondary:")
        self.logger.info(
            f"     MCC:          {summary['mean_val_mcc']:.4f}"
            f" ± {summary['std_val_mcc']:.4f}"
        )
        self.logger.info(
            f"     Bal. Acc:     {summary['mean_val_balanced_acc']:.4f}"
            f" ± {summary['std_val_balanced_acc']:.4f}"
        )
        self.logger.info(
            f"     Log loss:     {summary['mean_val_log_loss']:.4f}"
            f" ± {summary['std_val_log_loss']:.4f}"
        )
        self.logger.info(
            f"     Best iter:    {summary['mean_best_iteration']:.1f}"
        )
        self.logger.info(f"{'='*70}")

        return summary




    # ═════════════════════════════════════════════════════════════════════════
    # FEATURE PRUNING BY STABILITY (NEW)
    # ═════════════════════════════════════════════════════════════════════════

    def prune_features_by_stability(
        self,
        stability_df: pd.DataFrame,
        top_n: int = 30,
        min_stability: float = 0.5
    ) -> List[str]:
        """
        Prune features based on cross-validation stability.
        
        Args:
            stability_df: Output from cross_validate()['feature_stability']
            top_n: Keep top N most stable features
            min_stability: Minimum stability threshold
            
        Returns:
            List of selected feature names
        """
        # Filter by minimum stability
        stable = stability_df[stability_df['stability'] >= min_stability].copy()
        
        # Sort by stability and take top N
        stable = stable.sort_values('stability', ascending=False).head(top_n)
        
        selected_features = stable['feature'].tolist()
        
        self.logger.info("=" * 70)
        self.logger.info("FEATURE PRUNING BY STABILITY")
        self.logger.info("=" * 70)
        self.logger.info(f"  Original features:  {len(stability_df)}")
        self.logger.info(f"  Selected features:  {len(selected_features)}")
        self.logger.info(f"  Min stability:      {min_stability:.3f}")
        self.logger.info(f"  Actual min:         {stable['stability'].min():.3f}")
        self.logger.info(f"  Actual max:         {stable['stability'].max():.3f}")
        
        # Save pruning metadata
        pruning_metadata = {
            'selected_features': selected_features,
            'n_original': len(stability_df),
            'n_selected': len(selected_features),
            'min_stability': min_stability,
            'top_n': top_n,
            'timestamp': datetime.now().isoformat(),
            'stability_stats': stable[['feature', 'stability', 'mean_rank']].to_dict('records')
        }
        
        pruning_path = os.path.join(self.log_dir, 'feature_pruning.json')
        with open(pruning_path, 'w') as f:
            json.dump(pruning_metadata, f, indent=2)
        
        self.logger.info(f"  Pruning metadata saved: {pruning_path}")
        self.logger.info("=" * 70)
        
        return selected_features


    def train_with_pruned_features(
        self,
        df: pd.DataFrame,
        selected_features: List[str],
        **kwargs
    ) -> Dict:
        """
        Retrain model using only selected features.
        
        This bypasses feature engineering and works directly with
        a pre-selected feature set.
        
        Usage:
            # First: run cross_validate to get stability
            cv_results = model.cross_validate(df, n_splits=5, **label_kwargs)
            
            # Second: prune features
            selected = model.prune_features_by_stability(
                cv_results['feature_stability'],
                top_n=30
            )
            
            # Third: retrain with pruned features
            results = model.train_with_pruned_features(df, selected, **label_kwargs)
        """
        self.logger.info("=" * 70)
        self.logger.info(f"RETRAINING WITH PRUNED FEATURES ({len(selected_features)} features)")
        self.logger.info("=" * 70)
        
        # Get drop_composite_scores from kwargs
        drop_composite = kwargs.pop('drop_composite_scores', False)
        
        # Compute full features
        df_feat = self.feature_engine.compute_features(
            df,
            drop_composite_scores=drop_composite
        )
        
        # Create labels
        labels = self.create_expansion_regime_labels(df, **kwargs)
        
        # Remove invalid rows
        valid_mask = ~labels.isna()
        df_feat = df_feat[valid_mask].copy()
        labels = labels[valid_mask].copy()
        
        # FILTER: Keep only selected features
        exclude_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Time', 'Date', 'Datetime'}
        available_features = [f for f in selected_features if f in df_feat.columns]
        
        if len(available_features) < len(selected_features):
            missing = set(selected_features) - set(available_features)
            self.logger.warning(f"  {len(missing)} selected features not found in data:")
            for feat in list(missing)[:5]:
                self.logger.warning(f"    - {feat}")
        
        df_model = df_feat[available_features].copy()
        
        self.logger.info(f"  Using {len(available_features)} features (pruned from full set)")
        
        # Now call regular train() but with pre-filtered features
        # We need to bypass feature engineering in train()
        
        # Temporarily store current feature engine
        original_engine = self.feature_engine
        
        # Create a dummy engine that just returns the input
        class DummyEngine:
            def compute_features(self, df, **kwargs):
                return df
        
        self.feature_engine = DummyEngine()
        
        # Prepare modified df with only selected features + OHLCV
        df_for_train = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        for feat in available_features:
            df_for_train[feat] = df_feat[feat]
        
        # Call train with dummy engine
        try:
            # Remove label creation from kwargs - we already have labels
            # So we pass a pre-made DataFrame that includes features
            results = self.train(df_for_train, **kwargs)
        finally:
            # Restore original engine
            self.feature_engine = original_engine
        
        return results


    # ═════════════════════════════════════════════════════════════════════════
    # SHAP ANALYSIS WITH LABEL-FEATURE COUPLING DETECTION (NEW)
    # ═════════════════════════════════════════════════════════════════════════

    def analyze_shap_interactions(
        self,
        df: pd.DataFrame,
        n_samples: int = 500,
        plot: bool = True,
        check_label_coupling: bool = True
    ) -> Dict:
        """
        SHAP analysis to verify model is learning true interactions.
        
        What to look for:
            ✓ GOOD: Top features are primitives (atr_14, volume_zscore, bb_width)
                    → Model learned complex interactions from simple building blocks
            
            ✗ BAD:  Top features are composites (expansion_quality_score, regime_quality)
                    → Model is just memorizing your handcrafted logic
        
        Args:
            df: OHLCV data
            n_samples: Number of samples for SHAP (balance speed vs accuracy)
            plot: Generate SHAP plots
            check_label_coupling: Check if top features are label formula proxies
            
        Returns:
            Dict with SHAP values and analysis
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        try:
            import shap
        except ImportError:
            raise ImportError("Install shap: pip install shap")
        
        self.logger.info("=" * 70)
        self.logger.info("SHAP INTERACTION ANALYSIS")
        self.logger.info("=" * 70)
        
        # Prepare data
        features = self.feature_engine.compute_features(df)
        for feat in self.feature_names:
            if feat not in features.columns:
                features[feat] = 0.0
        features = features[self.feature_names].ffill().bfill().fillna(0.0)
        
        # Sample for speed
        if len(features) > n_samples:
            sample_idx = np.random.choice(len(features), n_samples, replace=False)
            X_sample = features.iloc[sample_idx].values.astype(np.float32)
        else:
            X_sample = features.values.astype(np.float32)
        
        # Create explainer
        self.logger.info(f"Computing SHAP values for {len(X_sample)} samples...")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, shap_values is (n_samples, n_features)
        # Get mean absolute SHAP value per feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance DataFrame
        shap_importance = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Save
        shap_path = os.path.join(self.log_dir, 'shap_importance.csv')
        shap_importance.to_csv(shap_path, index=False)
        
        # ── Analysis ──────────────────────────────────────────────────────
        self.logger.info("\nTop 20 features by SHAP importance:")
        for i, row in shap_importance.head(20).iterrows():
            self.logger.info(f"  {i+1:>2}. {row['feature']:40s}: {row['mean_abs_shap']:.4f}")
        
        # Check for composite features in top 10
        composite_keywords = ['score', 'quality', 'strength', 'alignment', 'persistence']
        top_10 = shap_importance.head(10)['feature'].tolist()
        
        composite_in_top10 = [
            f for f in top_10 
            if any(keyword in f.lower() for keyword in composite_keywords)
        ]
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("SHAP INTERPRETATION:")
        
        if len(composite_in_top10) >= 5:
            self.logger.warning("⚠ WARNING: Model dominated by composite features")
            self.logger.warning(f"  Composites in top 10: {composite_in_top10}")
            self.logger.warning("  → Model is approximating YOUR handcrafted logic")
            self.logger.warning("  → Consider dropping composite features and retraining")
            interpretation = "APPROXIMATING_HANDCRAFT"
        else:
            self.logger.info("✓ GOOD: Model learning from primitive features")
            self.logger.info(f"  Composites in top 10: {len(composite_in_top10)}")
            self.logger.info("  → Model discovered real signal structure")
            interpretation = "LEARNED_INTERACTIONS"
        
        # ══════════════════════════════════════════════════════════════════
        # LABEL-FEATURE COUPLING CHECK (NEW)
        # ══════════════════════════════════════════════════════════════════
        coupling_features = []
        coupling_risk = None
        
        if check_label_coupling:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("LABEL-FEATURE COUPLING ANALYSIS")
            self.logger.info("=" * 70)
            
            # Features that directly appear in label formula
            label_formula_features = [
                'atr_expansion_ratio',
                'volume_ratio',
                'relative_volume',
                'bb_width',
                'volatility_breakout_strength',
            ]
            
            # Check if these dominate top 10 SHAP features
            coupling_features = [
                f for f in top_10
                if any(lf in f for lf in label_formula_features)
            ]
            
            coupling_ratio = len(coupling_features) / 10
            
            self.logger.info(f"  Label formula features in top 10: {len(coupling_features)}/10")
            if coupling_features:
                self.logger.info(f"  Coupling features: {coupling_features}")
            
            if coupling_ratio > 0.6:
                self.logger.warning("  ⚠ HIGH LABEL-FEATURE COUPLING")
                self.logger.warning("    → Model may be approximating label formula")
                self.logger.warning("    → Backtest performance may be inflated")
                self.logger.warning("  Recommendation:")
                self.logger.warning("    1. Check if top features are just label proxies")
                self.logger.warning("    2. Consider adding more derivative features")
                self.logger.warning("    3. Test on out-of-sample data")
                coupling_risk = "HIGH"
            elif coupling_ratio > 0.4:
                self.logger.warning("  ⚠ MODERATE LABEL-FEATURE COUPLING")
                coupling_risk = "MODERATE"
            else:
                self.logger.info("  ✓ Low coupling — model uses diverse features")
                coupling_risk = "LOW"
        
        self.logger.info("=" * 70)
        
        # ── Plotting ──────────────────────────────────────────────────────
        if plot:
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, 
                X_sample, 
                feature_names=self.feature_names,
                show=False,
                max_display=20
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.plot_dir, 'shap_summary.png'),
                dpi=150, bbox_inches='tight'
            )
            plt.close()
            
            # Bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_names,
                plot_type='bar',
                show=False,
                max_display=20
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.plot_dir, 'shap_bar.png'),
                dpi=150, bbox_inches='tight'
            )
            plt.close()
            
            self.logger.info(f"\n✓ SHAP plots saved to {self.plot_dir}")
        
        return {
            'shap_values': shap_values,
            'shap_importance': shap_importance,
            'interpretation': interpretation,
            'composite_in_top10': composite_in_top10,
            'coupling_risk': coupling_risk,
            'coupling_features': coupling_features,
            'explainer': explainer
        }


    # ═════════════════════════════════════════════════════════════════════════
    # OUT-OF-TIME VALIDATION (NEW)
    # ═════════════════════════════════════════════════════════════════════════

    def validate_out_of_time(
        self,
        df: pd.DataFrame,
        oot_split: float = 0.15,  # Last 15% reserved
        **label_kwargs
    ) -> Dict:
        """
        Final out-of-time validation on untouched holdout.
        
        This should be run ONCE before deployment.
        
        Data split:
            [────────── TRAIN + VAL + CALIB ──────────][──── OOT ────]
                   (Used during development)              (UNTOUCHED)
        
        Args:
            df: Full dataset
            oot_split: Fraction to reserve for OOT
            **label_kwargs: Passed to label creation
            
        Returns:
            OOT performance metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        self.logger.info("=" * 70)
        self.logger.info("OUT-OF-TIME VALIDATION (FINAL HOLDOUT)")
        self.logger.info("=" * 70)
        self.logger.warning("⚠ This should only be run ONCE before deployment")
        
        # Feature engineering
        df_feat = self.feature_engine.compute_features(df)
        labels = self.create_expansion_regime_labels(df, **label_kwargs)
        
        # Clean
        valid_mask = ~labels.isna()
        df_feat = df_feat[valid_mask].copy()
        labels = labels[valid_mask].copy()
        
        # Feature columns
        exclude_cols = {'Open', 'High', 'Low', 'Close', 'Volume',
                        'Time', 'Date', 'Datetime'}
        feature_cols = [c for c in df_feat.columns if c not in exclude_cols]
        df_model = df_feat[feature_cols].copy()
        
        # OOT split
        n = len(df_model)
        oot_size = int(n * oot_split)
        oot_start = n - oot_size
        
        oot_data = df_model.iloc[oot_start:]
        oot_labels = labels.iloc[oot_start:]
        
        self.logger.info(f"  OOT set: {len(oot_data)} rows ({oot_split:.1%} of data)")
        self.logger.info(f"  OOT period: {df.index[oot_start]} to {df.index[-1]}")
        
        # Prepare
        X_oot = self._prepare_X(oot_data.copy())
        y_oot = oot_labels.reset_index(drop=True).values.astype(int)
        
        # Predict
        doot = xgb.DMatrix(X_oot, label=y_oot, feature_names=self.feature_names)
        
        # Use best_iteration (CRITICAL FIX)
        if self.best_iteration is not None:
            oot_proba_raw = self.model.predict(
                doot,
                iteration_range=(0, self.best_iteration)
            )
        else:
            oot_proba_raw = self.model.predict(doot)
            self.logger.warning("  best_iteration not set — using all trees")
        
        # Calibrate
        if self.calibrator.is_fitted:
            oot_proba = self.calibrator.transform(oot_proba_raw)
        else:
            oot_proba = oot_proba_raw
            self.logger.warning("  No calibrator — using raw probabilities")
        
        # Evaluate
        oot_metrics = self._evaluate(y_oot, oot_proba, prefix='oot_')
        
        # Log
        self.logger.info("\n" + "=" * 70)
        self.logger.info("OOT PERFORMANCE (UNTOUCHED HOLDOUT)")
        self.logger.info("=" * 70)
        self.logger.info(f"  Precision:    {oot_metrics['oot_precision']:.4f}")
        self.logger.info(f"  Recall:       {oot_metrics['oot_recall']:.4f}")
        self.logger.info(f"  F1:           {oot_metrics['oot_f1']:.4f}")
        self.logger.info(f"  ROC-AUC:      {oot_metrics['oot_roc_auc']:.4f}")
        self.logger.info(f"  Avg Prec:     {oot_metrics['oot_avg_precision']:.4f}")
        self.logger.info(f"  MCC:          {oot_metrics['oot_mcc']:.4f}")
        
        # Compare to validation
        if 'val_precision' in self.training_history:
            val_precision = self.training_history.get('val_precision', 0)
            val_recall = self.training_history.get('val_recall', 0)
            val_f1 = self.training_history.get('val_f1', 0)
            
            precision_gap = val_precision - oot_metrics['oot_precision']
            recall_gap = val_recall - oot_metrics['oot_recall']
            f1_gap = val_f1 - oot_metrics['oot_f1']
            
            self.logger.info(f"\n  Validation precision:  {val_precision:.4f}")
            self.logger.info(f"  OOT precision:         {oot_metrics['oot_precision']:.4f}")
            self.logger.info(f"  Gap:                   {precision_gap:+.4f}")
            
            self.logger.info(f"\n  Validation F1:         {val_f1:.4f}")
            self.logger.info(f"  OOT F1:                {oot_metrics['oot_f1']:.4f}")
            self.logger.info(f"  Gap:                   {f1_gap:+.4f}")
            
            if abs(precision_gap) > 0.10 or abs(f1_gap) > 0.10:
                self.logger.warning("\n  ⚠ LARGE GAP — Model may not generalize")
                self.logger.warning("    → Review feature engineering")
                self.logger.warning("    → Check for label-feature coupling")
                self.logger.warning("    → Consider more regularization")
            else:
                self.logger.info("\n  ✓ Acceptable generalization")
        
        self.logger.info("=" * 70)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_oot, (oot_proba >= 0.5).astype(int))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['Unfavorable', 'Favorable'],
                    yticklabels=['Unfavorable', 'Favorable'])
        plt.title('OOT Confusion Matrix (Final Holdout)')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'oot_confusion_matrix.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            **oot_metrics,
            'oot_size': len(oot_data),
            'oot_start_date': str(df.index[oot_start]),
            'oot_end_date': str(df.index[-1]),
        }


    # ═════════════════════════════════════════════════════════════════════════
    # PREDICT (single-bar, for live use)
    # ═════════════════════════════════════════════════════════════════════════

    def predict(
        self,
        df: pd.DataFrame,
        return_details: bool = True
    ) -> Dict:
        """
        Raw prediction on latest bar (uses last row of df).

        Returns the expansion quality probability and binary prediction.
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
        
        # ══════════════════════════════════════════════════════════════════
        # CRITICAL FIX: Explicitly use best_iteration
        # ══════════════════════════════════════════════════════════════════
        if self.best_iteration is not None:
            raw_proba = self.model.predict(
                dmatrix,
                iteration_range=(0, self.best_iteration)
            )[0]
        else:
            raw_proba = self.model.predict(dmatrix)[0]
            self.logger.warning("best_iteration not set — using all trees")

        # Calibrate if available
        if self.calibrator.is_fitted:
            proba = self.calibrator.transform(np.array([raw_proba]))[0]
        else:
            proba = raw_proba

        regime = int(proba >= 0.5)
        regime_label = self.CLASS_LABELS[regime]

        result = {
            'expansion_quality_prob': float(proba),
            'regime': regime,
            'regime_label': regime_label,
        }

        if return_details:
            result['raw_probability'] = float(raw_proba)
            result['is_calibrated'] = self.calibrator.is_fitted

        # Log
        self.logger.info("=" * 70)
        self.logger.info("PREDICTION (BINARY REGIME)")
        self.logger.info(f"  Regime:       {regime} ({regime_label})")
        self.logger.info(f"  Probability:  {proba:.2%}")
        quality = "🟢 HIGH" if proba >= 0.7 else "🟡 MODERATE" if proba >= 0.5 else "🔴 LOW"
        self.logger.info(f"  Quality:      {quality}")
        self.logger.info("=" * 70)

        return result


    # ═════════════════════════════════════════════════════════════════════════
    # PREDICT FOR ENSEMBLE (primary output for production)
    # ═════════════════════════════════════════════════════════════════════════

    def predict_for_ensemble(
        self,
        df: pd.DataFrame,
        min_confidence: float = 0.55,
    ) -> Dict:
        """
        Ensemble-ready prediction with regime quality scoring.

        Designed to be consumed directly by the ensemble layer that
        combines trend, volatility, and momentum signals.

        Output contract:
            expansion_quality_prob  float [0,1]  Calibrated probability that
                                                  current regime is FAVORABLE.
                                                  Primary signal for ensemble.

            regime_strength         float [0,1]  Normalized regime quality score.
                                                  = expansion_quality_prob
                                                  (for symmetry with momentum model)

            is_favorable            bool         True if prob >= min_confidence

            is_abstaining           bool         True when model is uncertain
                                                  (prob close to 0.5)

            regime_context          dict         Key feature values for ensemble
                                                  meta-decisions.

        Usage in ensemble:
            - Use expansion_quality_prob as weight/filter
            - If is_favorable=False → reduce position size or skip trade
            - If is_abstaining=True → ensemble should be cautious
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
        
        # Use best_iteration
        if self.best_iteration is not None:
            raw_proba = self.model.predict(
                dmatrix,
                iteration_range=(0, self.best_iteration)
            )[0]
        else:
            raw_proba = self.model.predict(dmatrix)[0]

        # Calibrate
        if self.calibrator.is_fitted:
            proba = self.calibrator.transform(np.array([raw_proba]))[0]
        else:
            proba = raw_proba

        expansion_quality_prob = float(proba)
        regime_strength = expansion_quality_prob  # Normalized [0,1]

        is_favorable = expansion_quality_prob >= min_confidence
        
        # Abstain if probability is very close to decision boundary
        abstain_zone = 0.1  # ±10% around 0.5
        is_abstaining = abs(expansion_quality_prob - 0.5) < abstain_zone

        # Regime context: key features for ensemble meta-decisions
        last = features.iloc[-1]

        def _safe(col: str, default: float = 0.0) -> float:
            v = last.get(col, default)
            return float(v) if not pd.isna(v) else default

        regime_context = {
            # Volatility features
            'atr_expansion_ratio':      _safe('atr_expansion_ratio_14', 1.0),
            'atr_slope':                _safe('atr_slope_14'),
            'volatility_breakout_strength': _safe('volatility_breakout_strength'),
            'bb_width_pctrank':         _safe('bb_width_pctrank'),
            'realized_volatility_20':   _safe('realized_volatility_20'),
            
            # Volume features
            'volume_zscore':            _safe('volume_zscore'),
            'relative_volume':          _safe('relative_volume', 1.0),
            'volume_spike_ratio':       _safe('volume_spike_ratio', 1.0),
            'obv_slope':                _safe('obv_slope_10'),
            
            # Interaction features
            'volatility_volume_alignment': _safe('volatility_volume_alignment'),
            'volume_pressure':          _safe('volume_pressure'),
            'participation_strength':   _safe('participation_strength'),
            'compression_release':      _safe('compression_release'),
            
            # Transition features
            'bb_squeeze_duration':      _safe('bb_squeeze_duration'),
            'bars_since_expansion':     _safe('bars_since_expansion'),
        }

        timestamp = (
            df.index[-1].isoformat()
            if hasattr(df.index[-1], 'isoformat')
            else str(df.index[-1])
        )

        result = {
            # ── Core ensemble inputs ──────────────────────────────────────
            'expansion_quality_prob': expansion_quality_prob,  # PRIMARY
            'regime_strength':        regime_strength,
            'is_favorable':           is_favorable,
            'is_abstaining':          is_abstaining,

            # ── Context for ensemble meta-logic ───────────────────────────
            'regime_context': regime_context,

            # ── Metadata ──────────────────────────────────────────────────
            'regime_label':     self.CLASS_LABELS[int(is_favorable)],
            'raw_probability':  float(raw_proba),
            'is_calibrated':    self.calibrator.is_fitted,
            'timestamp':        timestamp,
            'min_confidence':   min_confidence,
        }

        self.logger.info("=" * 70)
        self.logger.info("ENSEMBLE PREDICTION (REGIME V1.0)")
        self.logger.info(f"  Expansion quality: {expansion_quality_prob:.2%}")
        self.logger.info(f"  Regime strength:   {regime_strength:.4f}")
        self.logger.info(f"  Is favorable:      {is_favorable}")
        self.logger.info(f"  Abstaining:        {is_abstaining}")
        
        quality_desc = (
            "🟢 EXCELLENT" if expansion_quality_prob >= 0.75 else
            "🟢 GOOD" if expansion_quality_prob >= 0.65 else
            "🟡 MODERATE" if expansion_quality_prob >= 0.55 else
            "🔴 POOR"
        )
        self.logger.info(f"  Quality:           {quality_desc}")
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
            trained/xgb_regime_v{version}.json     ← XGBoost native model
            trained/calibrator_v{version}.pkl      ← ProbabilityCalibrator
            trained/metadata_v{version}.pkl        ← Feature names, hyperparams, etc.
        """
        if self.model is None:
            raise ValueError("No trained model to save")

        version = version or datetime.now().strftime('%Y%m%d_%H%M%S')

        model_path    = model_path or os.path.join(
            self.trained_dir, f'xgb_regime_v{version}.json'
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
            'model_type':       'regime_detector',
            'num_classes':      2,
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
            }
        }

        joblib.dump(metadata, metadata_path)
        self.logger.info(f"✓ Metadata saved:      {metadata_path}")

        self.logger.info("=" * 70)
        self.logger.info("SAVE SUMMARY")
        self.logger.info(f"  Version:    {version}")
        self.logger.info(f"  Features:   {self.n_features}")
        self.logger.info(f"  Classes:    2 (UNFAVORABLE | FAVORABLE)")
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

        num_classes = metadata.get('num_classes', 2)
        if num_classes != 2:
            self.logger.warning(
                f"⚠ Loaded model has {num_classes} classes, expected 2"
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
    print("XGBoost Volatility-Volume Regime Detector V1.0 — Complete")
    print("=" * 70)
    print()
    print("New Features in This Version:")
    print("  ✓ Label quality diagnostics (streak analysis, autocorrelation)")
    print("  ✓ Per-fold calibration in cross-validation")
    print("  ✓ SHAP analysis with label-feature coupling detection")
    print("  ✓ Feature pruning by stability")
    print("  ✓ Out-of-time validation (final holdout)")
    print("  ✓ Explicit best_iteration enforcement in predict()")
    print("  ✓ Drop composite scores option")
    print("  ✓ Transition features (Block 13)")
    print()
    print("Complete Production Workflow:")
    print("  1. diagnose_label_quality()     → Check label clustering")
    print("  2. cross_validate()             → Assess stability (with calibration)")
    print("  3. prune_features_by_stability()→ Select top features")
    print("  4. train()                      → Final training (with diagnostics)")
    print("  5. analyze_shap_interactions()  → Verify learning (not memorizing)")
    print("  6. validate_out_of_time()       → Final check (run ONCE)")
    print("  7. save()                       → Deploy")
    print()
    print("=" * 70)