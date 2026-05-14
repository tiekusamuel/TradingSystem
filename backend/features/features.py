# features/technical_indicators.py - FINAL PRODUCTION VERSION

import pandas as pd
import pandas_ta as ta # type: ignore
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import warnings
    

class MomentumFeatureEngine:
    """
    Pure momentum feature engineering for OHLCV data.

    Design principles:
        - ZERO lookahead: every feature uses only data available at bar close
        - ATR-normalised where possible (removes volatility regime effect)
        - Covers 12 conceptual blocks from micro to macro momentum
        - Produces a flat feature DataFrame ready for tree models
    """

    def __init__(self, atr_period: int = 14):
        """
        Args:
            atr_period: Lookback for ATR baseline used in normalisation
        """
        self.atr_period = atr_period

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all momentum features.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
                Required columns: Open, High, Low, Close, Volume

        Returns:
            Feature DataFrame with same index as df.
            Does NOT include OHLCV columns.
        """
        self._validate_input(df)

        close  = df['Close']
        high   = df['High']
        low    = df['Low']
        open_  = df['Open']
        volume = df['Volume']

        feat = pd.DataFrame(index=df.index)

        # Compute ATR once — reused by multiple blocks
        atr = self._compute_atr(df, self.atr_period)

        # ── 12 Feature Blocks ─────────────────────────────────────────────
        feat = self._block_rate_of_change(feat, close)
        feat = self._block_momentum_acceleration(feat)
        feat = self._block_rsi_family(feat, close)
        feat = self._block_macd(feat, close)
        feat = self._block_atr_normalised(feat, close, atr)
        feat = self._block_volume_price(feat, df, close, volume)
        feat = self._block_streak(feat, close)
        feat = self._block_candle_body(feat, open_, high, low, close)
        feat = self._block_moving_average_structure(feat, close)
        feat = self._block_momentum_regime(feat, close)
        feat = self._block_session_time(feat, df)
        feat = self._block_cross_timeframe(feat, close)

        return feat

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if df.empty:
            raise ValueError("DataFrame is empty")

    # ──────────────────────────────────────────────────────────────────────────

    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        EWM ATR — shift(1) on previous close prevents any look-ahead.
        EWM is preferred over rolling mean: smoother, less lag.
        """
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low']  - prev_close).abs()
        ], axis=1).max(axis=1)

        return tr.ewm(span=period, min_periods=period).mean()

    def _compute_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """
        RSI with Wilder's EWM smoothing.
        Fills NaN with 50 (neutral) to avoid NaN propagation in features.
        """
        delta    = series.diff(1)
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        rsi      = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def _compute_streak(binary_series: pd.Series) -> pd.Series:
        """
        Vectorised signed streak.
        Positive = consecutive up bars, negative = consecutive down bars.
        Example: [1,1,1,0,0,1] → [1,2,3,−1,−2,1]
        """
        signed  = binary_series.replace(0, -1)
        changes = signed != signed.shift(1)
        groups  = changes.cumsum()
        counts  = binary_series.groupby(groups).cumcount() + 1
        return (counts * signed).fillna(0).astype(int)

    @staticmethod
    def _compute_rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Rolling VWAP (session-reset approximation using fixed window).
        Returns deviation of Close from VWAP as a fraction.
        """
        typical = (df['High'] + df['Low'] + df['Close']) / 3
        vol     = df['Volume']
        vwap    = (
            (typical * vol).rolling(window).sum() /
            vol.rolling(window).sum().replace(0, np.nan)
        )
        return (df['Close'] - vwap) / vwap.replace(0, np.nan)

    # ──────────────────────────────────────────────────────────────────────────
    # FEATURE BLOCKS
    # ──────────────────────────────────────────────────────────────────────────

    def _block_rate_of_change(
        self, feat: pd.DataFrame, close: pd.Series
    ) -> pd.DataFrame:
        """
        Block 1 — Rate of Change.
        Core momentum signal at multiple horizons.
        Log returns included: more stationary, better for trees.
        """
        for p in [1, 3, 5, 10, 20, 60]:
            prev = close.shift(p)
            feat[f'roc_{p}'] = (close - prev) / prev.replace(0, np.nan)

        for p in [1, 5, 20]:
            feat[f'log_ret_{p}'] = np.log(
                close / close.shift(p).replace(0, np.nan)
            )

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_momentum_acceleration(self, feat: pd.DataFrame) -> pd.DataFrame:
        """
        Block 2 — Momentum Acceleration & Jerk.
        Measures whether momentum is building, peaking, or exhausting.
            Acceleration = Δ(ROC)
            Jerk         = Δ(Acceleration)   ← catches exhaustion points
        """
        roc1 = feat['roc_1']
        roc5 = feat['roc_5']

        feat['momentum_accel_1'] = roc1 - roc1.shift(1)
        feat['momentum_accel_5'] = roc5 - roc5.shift(5)
        feat['momentum_jerk']    = (
            feat['momentum_accel_1'] - feat['momentum_accel_1'].shift(1)
        )

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_rsi_family(
        self, feat: pd.DataFrame, close: pd.Series
    ) -> pd.DataFrame:
        """
        Block 3 — RSI Family.
        Multi-period RSI + rate of change of RSI + divergence proxy.
        Divergence: price at new high but RSI lagging → potential reversal.
        """
        for p in [7, 14, 21]:
            feat[f'rsi_{p}'] = self._compute_rsi(close, p)

        # RSI momentum (how fast RSI itself is moving)
        feat['rsi_momentum'] = feat['rsi_14'] - feat['rsi_14'].shift(5)

        # Divergence proxy
        price_high_5 = close.rolling(5).max()
        price_low_5  = close.rolling(5).min()
        rsi_high_5   = feat['rsi_14'].rolling(5).max()
        rsi_low_5    = feat['rsi_14'].rolling(5).min()

        feat['rsi_divergence'] = np.where(
            (close >= price_high_5) & (feat['rsi_14'] < rsi_high_5 - 2),
            -1,   # bearish divergence
            np.where(
                (close <= price_low_5) & (feat['rsi_14'] > rsi_low_5 + 2),
                1,    # bullish divergence
                0
            )
        )

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_macd(
        self, feat: pd.DataFrame, close: pd.Series
    ) -> pd.DataFrame:
        """
        Block 4 — MACD Momentum.
        All values normalised by price level for cross-instrument use.
        Histogram slope and acceleration capture momentum phase transitions.
        """
        ema12  = close.ewm(span=12, min_periods=12).mean()
        ema26  = close.ewm(span=26, min_periods=26).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, min_periods=9).mean()
        hist   = macd - signal

        safe_close = close.replace(0, np.nan)

        feat['macd_normalised']        = macd   / safe_close
        feat['macd_signal_normalised'] = signal / safe_close
        feat['macd_histogram']         = hist   / safe_close
        feat['macd_hist_slope']        = hist - hist.shift(1)
        feat['macd_hist_accel']        = (
            feat['macd_hist_slope'] - feat['macd_hist_slope'].shift(1)
        )

        # Crossover event: +1 bullish cross, -1 bearish cross, 0 none
        feat['macd_crossover'] = np.where(
            (macd > signal) & (macd.shift(1) <= signal.shift(1)),  1,
            np.where(
                (macd < signal) & (macd.shift(1) >= signal.shift(1)), -1, 0
            )
        )

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_atr_normalised(
        self, feat: pd.DataFrame, close: pd.Series, atr: pd.Series
    ) -> pd.DataFrame:
        """
        Block 5 — ATR-Normalised Momentum.
        Divides raw price moves by ATR × sqrt(horizon) so the signal is
        comparable across different volatility regimes.
        """
        safe_close = close.replace(0, np.nan)
        feat['atr_normalised'] = atr / safe_close   # ATR as % of price

        safe_atr = atr.replace(0, np.nan)
        for p in [1, 5, 20]:
            raw_move = close - close.shift(p)
            feat[f'momentum_atr_ratio_{p}'] = raw_move / (safe_atr * np.sqrt(p))

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_volume_price(
        self,
        feat: pd.DataFrame,
        df: pd.DataFrame,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """
        Block 6 — Volume-Price Momentum.
        Volume confirms or contradicts price momentum.
        OBV momentum, force index, volume ratio, VWAP deviation.
        """
        # OBV
        direction = np.sign(close.diff(1))
        obv       = (direction * volume).cumsum()
        vol_mean  = volume.rolling(20).mean().replace(0, np.nan)

        feat['obv_momentum_5']  = (obv - obv.shift(5))  / vol_mean
        feat['obv_momentum_20'] = (obv - obv.shift(20)) / vol_mean

        # Volume ratio (current vs 20-bar average)
        feat['volume_ratio']   = volume / vol_mean
        feat['volume_ratio_5'] = volume.rolling(5).mean() / vol_mean

        # Force index (price change × volume), EWM smoothed
        force = close.diff(1) * volume
        feat['force_index_normalised'] = (
            force.ewm(span=13).mean() /
            (close * vol_mean).replace(0, np.nan)
        )

        # VWAP deviation
        feat['vwap_deviation'] = self._compute_rolling_vwap(df)

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_streak(
        self, feat: pd.DataFrame, close: pd.Series
    ) -> pd.DataFrame:
        """
        Block 7 — Consecutive Bar Streak.
        Captures persistence of directional pressure.
        Streak normalised by ATR ratio for magnitude context.
        """
        up_bars = (close > close.shift(1)).astype(int)
        feat['signed_streak'] = self._compute_streak(up_bars)

        # Magnitude of recent 1-bar moves × streak sign
        avg_move = close.diff(1).abs().rolling(5).mean()
        atr      = self._compute_atr(
            pd.DataFrame({'High': close, 'Low': close, 'Close': close}),
            self.atr_period
        )
        feat['streak_atr_ratio'] = (
            feat['signed_streak'] * avg_move
        ) / atr.replace(0, np.nan)

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_candle_body(
        self,
        feat: pd.DataFrame,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Block 8 — Candle Body Analysis.
        Body-to-range ratio encodes intrabar momentum commitment.
        Shadow imbalance reveals buying vs selling pressure.
        """
        body       = close - open_
        full_range = (high - low).replace(0, np.nan)

        feat['body_ratio']    = body / full_range           # −1 .. +1
        feat['body_ratio_ma5'] = feat['body_ratio'].rolling(5).mean()

        upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
        lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low

        feat['upper_shadow_ratio']  = upper_wick / full_range
        feat['lower_shadow_ratio']  = lower_wick / full_range

        # Positive = bullish pressure (large lower wick rejected selling)
        feat['shadow_momentum']     = (lower_wick - upper_wick) / full_range
        feat['shadow_momentum_ma5'] = feat['shadow_momentum'].rolling(5).mean()

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_moving_average_structure(
        self, feat: pd.DataFrame, close: pd.Series
    ) -> pd.DataFrame:
        """
        Block 9 — Moving Average Momentum Structure.
        Price position relative to MAs + MA slopes + alignment score.
        MA alignment: +1.0 fully bullish stack, −1.0 fully bearish stack.
        """
        mas = {}
        for p in [5, 10, 20, 50]:
            ma = close.rolling(p).mean()
            mas[p] = ma
            safe_ma = ma.replace(0, np.nan)
            feat[f'price_vs_ma_{p}'] = (close - ma) / safe_ma
            feat[f'ma_{p}_slope']    = (ma - ma.shift(5)) / safe_ma

        # Alignment score in [−1, +1]
        alignment = (
            (mas[5]  > mas[10]).astype(int) +
            (mas[10] > mas[20]).astype(int) +
            (mas[20] > mas[50]).astype(int)
        ) - (
            (mas[5]  < mas[10]).astype(int) +
            (mas[10] < mas[20]).astype(int) +
            (mas[20] < mas[50]).astype(int)
        )
        feat['ma_alignment'] = alignment / 3.0

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_momentum_regime(
        self, feat: pd.DataFrame, close: pd.Series
    ) -> pd.DataFrame:
        """
        Block 10 — Momentum Regime (Trending vs Mean-Reverting).
        Variance ratio ≈ Hurst exponent proxy:
            VR > 1 → trending (momentum regime)
            VR < 1 → mean-reverting
        Return autocorrelation: positive = momentum persistence.
        """
        ret1    = close.pct_change(1)
        var_1   = ret1.rolling(20).var().replace(0, np.nan)

        for h in [4, 8]:
            var_h = close.diff(h).rolling(20).var()
            feat[f'variance_ratio_{h}'] = (var_h / (h * var_1)).clip(0, 3)

        for lag in [1, 5]:
            feat[f'return_autocorr_{lag}'] = ret1.rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0.0,
                raw=False
            )

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_session_time(
        self, feat: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Block 11 — Circular Session & Time Features.
        Circular encoding prevents discontinuity at midnight / week boundary.
        Session indicators capture liquidity-driven momentum patterns.
        """
        minutes = df.index.hour * 60 + df.index.minute
        feat['time_sin'] = np.sin(2 * np.pi * minutes / 1440)
        feat['time_cos'] = np.cos(2 * np.pi * minutes / 1440)

        dow = df.index.dayofweek
        feat['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        feat['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        hour = df.index.hour
        feat['session_tokyo']            = ((hour >= 0)  & (hour < 9)).astype(np.float32)
        feat['session_london']           = ((hour >= 8)  & (hour < 16)).astype(np.float32)
        feat['session_ny']               = ((hour >= 13) & (hour < 21)).astype(np.float32)
        feat['session_overlap_london_ny']= ((hour >= 13) & (hour < 16)).astype(np.float32)

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_cross_timeframe(
        self,
        feat: pd.DataFrame,
        close: pd.Series
        ) -> pd.DataFrame:

        """
        Cross-timeframe momentum structure.
        """

        current_ret = close.pct_change(5)

        for tf_bars in [4, 24, 96]:

            prev = close.shift(tf_bars).replace(0, np.nan)

            # HTF return
            htf_ret = (close / prev) - 1

            feat[f'htf_momentum_{tf_bars}'] = htf_ret

            # Direction
            feat[f'htf_sign_{tf_bars}'] = np.sign(htf_ret)

            # Strength
            feat[f'htf_strength_{tf_bars}'] = (
                htf_ret.abs()
            )

            # Alignment with current momentum
            feat[f'alignment_{tf_bars}'] = (
                np.sign(current_ret)
                == np.sign(htf_ret)
            ).astype(int)

            # Distance from HTF mean
            htf_ma = close.rolling(tf_bars).mean()

            feat[f'htf_distance_ma_{tf_bars}'] = (
                (close - htf_ma)
                / htf_ma
            )

            # HTF persistence
            feat[f'htf_persistence_{tf_bars}'] = (
                (close > htf_ma)
                .rolling(5)
                .mean()
            )

        return feat
    
    
    
    
    


@dataclass
class TrendFeatureConfig:
    """All hyperparameters in one place — no magic numbers in the engine."""

    # SMA / EMA periods
    sma_periods: list = field(default_factory=lambda: [20, 50, 200])
    ema_periods: list = field(default_factory=lambda: [9, 21, 50, 200])

    # ADX / ATR
    adx_period: int = 14
    atr_period: int = 14

    # Swing-point detection
    swing_lookback: int = 10    # bars each side for local-extrema detection
    hh_hl_lookback: int = 50    # rolling window to score HH/HL density

    # Slope / consistency
    slope_window: int = 50      # linear-regression window

    # Normalisation (applied at end of pipeline)
    norm_window: int = 50       # rolling z-score / min-max window
    warmup_cutoff: int = 50     # drop first N rows after feature computation


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

class TrendFeatureEngine:
    """
    Pure trend-detection feature engineering for OHLCV data.

    Design principles:
        - ZERO lookahead: every feature uses only data available at bar close
        - ATR/price normalised where meaningful (removes regime dependency)
        - Covers 8 conceptual blocks: MAs, ADX, HH/HL structure, slope, volume,
          cross-timeframe, volatility context
        - Produces a flat feature DataFrame ready for LSTM models
        - Rolling normalisation applied to all features for stationarity
    """

    def __init__(self, config: Optional[TrendFeatureConfig] = None):
        self.cfg = config or TrendFeatureConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all trend features.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
                Required columns: Open, High, Low, Close
                Optional: Volume (if missing, volume features = 0)

        Returns:
            Feature DataFrame with same index as df.
            Does NOT include OHLCV columns.
            All features are normalised (z-score or min-max).
        """
        self._validate_input(df)

        close = df['Close']
        high = df['High']
        low = df['Low']
        open_ = df['Open']

        feat = pd.DataFrame(index=df.index)

        # Compute ATR once — reused by multiple blocks
        atr = self._compute_atr(df, self.cfg.atr_period)

        # ── 8 Feature Blocks ──────────────────────────────────────────────
        feat = self._block_sma(feat, close)
        feat = self._block_ema(feat, close)
        feat = self._block_adx(feat, df, atr)
        feat = self._block_hh_hl(feat, df)
        feat = self._block_slope(feat, close)
        feat = self._block_volume_trend(feat, df)
        feat = self._block_cross_timeframe(feat, close)
        feat = self._block_volatility_context(feat, close, atr)

        # ── Normalisation ─────────────────────────────────────────────────
        feat = self._normalise_features(feat)

        # ── Drop warmup period ────────────────────────────────────────────
        if self.cfg.warmup_cutoff > 0:
            feat = feat.iloc[self.cfg.warmup_cutoff:].reset_index(drop=True)

        return feat

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        required = ['Open', 'High', 'Low', 'Close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if df.empty:
            raise ValueError("DataFrame is empty")

    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        EWM ATR — shift(1) on previous close prevents any look-ahead.
        EWM is preferred over rolling mean: smoother, less lag.
        """
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs()
        ], axis=1).max(axis=1)

        return tr.ewm(span=period, min_periods=period).mean()

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        """Rolling z-score normalisation."""
        mu = series.rolling(window, min_periods=1).mean()
        sigma = series.rolling(window, min_periods=1).std()
        sigma = sigma.replace(0, np.nan).fillna(1e-10)
        return (series - mu) / sigma

    @staticmethod
    def _rolling_minmax(series: pd.Series, window: int) -> pd.Series:
        """Rescale to [-1, 1] using rolling min/max."""
        roll_min = series.rolling(window, min_periods=1).min()
        roll_max = series.rolling(window, min_periods=1).max()
        rng = (roll_max - roll_min).replace(0, np.nan).fillna(1e-10)
        scaled = (series - roll_min) / rng  # [0, 1]
        return scaled * 2 - 1  # [-1, 1]

    # ──────────────────────────────────────────────────────────────────────────
    # FEATURE BLOCKS
    # ──────────────────────────────────────────────────────────────────────────

    def _block_sma(self, feat: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """
        Block 1 — Simple Moving Average Trend Features.

        Features:
            sma_<p>          : (Close - SMA_p) / Close   — price deviation
            sma_cross_20_50  : SMA20 - SMA50             (raw, normalised later)
            sma_cross_50_200 : SMA50 - SMA200
            golden_cross     : +1 when SMA20 crosses above SMA50
            death_cross      : -1 when SMA20 crosses below SMA50
        """
        cfg = self.cfg
        raws = {}

        for p in cfg.sma_periods:
            raw = close.rolling(window=p, min_periods=1).mean()
            raws[p] = raw
            feat[f'sma_{p}'] = (close - raw) / close.replace(0, np.nan)

        if 20 in raws and 50 in raws:
            diff_20_50 = raws[20] - raws[50]
            feat['sma_cross_20_50'] = diff_20_50

            sign = np.sign(diff_20_50)
            prev_sign = sign.shift(1)
            feat['golden_cross'] = ((sign == 1) & (prev_sign == -1)).astype(float)
            feat['death_cross'] = ((sign == -1) & (prev_sign == 1)).astype(float) * -1

        if 50 in raws and 200 in raws:
            feat['sma_cross_50_200'] = raws[50] - raws[200]

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_ema(self, feat: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """
        Block 2 — Exponential Moving Average Trend Features.

        Features:
            ema_<p>           : (Close - EMA_p) / Close
            ema_cross_9_21    : EMA9 - EMA21   (momentum cross)
            ema_cross_50_200  : EMA50 - EMA200 (structural cross)
        """
        cfg = self.cfg
        raws = {}

        for p in cfg.ema_periods:
            raw = close.ewm(span=p, adjust=False, min_periods=1).mean()
            raws[p] = raw
            if p in (9, 21, 50):
                feat[f'ema_{p}'] = (close - raw) / close.replace(0, np.nan)

        if 9 in raws and 21 in raws:
            feat['ema_cross_9_21'] = raws[9] - raws[21]

        if 50 in raws and 200 in raws:
            feat['ema_cross_50_200'] = raws[50] - raws[200]

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_adx(
        self, feat: pd.DataFrame, df: pd.DataFrame, atr: pd.Series
    ) -> pd.DataFrame:
        """
        Block 3 — Wilder ADX & Directional Indicators.

        Features:
            di_plus   : +DI — bullish directional strength (0-100)
            di_minus  : -DI — bearish directional strength (0-100)
            di_diff   : DI+ - DI- (signed trend direction)
            adx       : Average Directional Index (trend strength 0-100)
            adx_14    : alias of adx
            atr_pct   : ATR as % of Close (volatility context)
        """
        p = self.cfg.adx_period

        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()

        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

        safe_atr = atr.replace(0, np.nan)

        di_plus = 100 * pos_dm.rolling(p, min_periods=1).mean() / safe_atr
        di_minus = 100 * neg_dm.rolling(p, min_periods=1).mean() / safe_atr

        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
        adx = dx.rolling(p, min_periods=1).mean()

        feat['di_plus'] = di_plus.fillna(25.0)
        feat['di_minus'] = di_minus.fillna(25.0)
        feat['di_diff'] = feat['di_plus'] - feat['di_minus']
        feat['adx'] = adx.fillna(25.0)
        feat['adx_14'] = feat['adx']
        feat['atr_pct'] = atr / df['Close'].replace(0, np.nan) * 100

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_hh_hl(self, feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Block 4 — Higher-High / Higher-Low Structure.

        Algorithm:
            1. Detect swing highs/lows as local extrema (± swing_lookback bars)
            2. Compare consecutive swings → HH/LH for highs, HL/LL for lows
            3. Roll over hh_hl_lookback bars to produce density scores
            4. Derive composite uptrend/downtrend flags
            5. Compute price position between last swing low and high

        Features:
            hh_signal       : +1 at Higher High bar
            hl_signal       : +1 at Higher Low bar
            lh_signal       : -1 at Lower High bar
            ll_signal       : -1 at Lower Low bar
            hh_hl_score     : rolling bullish-swing density (0-1)
            lh_ll_score     : rolling bearish-swing density (0-1)
            uptrend_flag    : +1 if bull score > 0.6
            downtrend_flag  : -1 if bear score > 0.6
            swing_position  : (Close - last swing low) / (last swing high - low)
        """
        lb = self.cfg.swing_lookback
        win = self.cfg.hh_hl_lookback
        n = len(df)

        highs = df['High'].values
        lows = df['Low'].values
        close = df['Close'].values

        # Swing detection
        sh_idx, sl_idx = [], []

        for i in range(lb, n - lb):
            window_h = highs[max(0, i - lb): i + lb + 1]
            window_l = lows[max(0, i - lb): i + lb + 1]
            if highs[i] == window_h.max():
                sh_idx.append(i)
            if lows[i] == window_l.min():
                sl_idx.append(i)

        # Per-bar signal arrays
        hh = np.zeros(n)
        hl = np.zeros(n)
        lh = np.zeros(n)
        ll = np.zeros(n)

        for k in range(1, len(sh_idx)):
            i, j = sh_idx[k], sh_idx[k - 1]
            if highs[i] > highs[j]:
                hh[i] = 1.0
            else:
                lh[i] = 1.0

        for k in range(1, len(sl_idx)):
            i, j = sl_idx[k], sl_idx[k - 1]
            if lows[i] > lows[j]:
                hl[i] = 1.0
            else:
                ll[i] = 1.0

        feat['hh_signal'] = hh
        feat['hl_signal'] = hl
        feat['lh_signal'] = lh * -1
        feat['ll_signal'] = ll * -1

        # Rolling density scores
        hh_s = pd.Series(hh, index=df.index)
        hl_s = pd.Series(hl, index=df.index)
        lh_s = pd.Series(lh, index=df.index)
        ll_s = pd.Series(ll, index=df.index)

        bull = hh_s + hl_s
        bear = lh_s + ll_s
        denom = (bull + bear).rolling(win, min_periods=1).sum() + 1e-10

        feat['hh_hl_score'] = bull.rolling(win, min_periods=1).sum() / denom
        feat['lh_ll_score'] = bear.rolling(win, min_periods=1).sum() / denom

        # Composite flags
        feat['uptrend_flag'] = (feat['hh_hl_score'] > 0.6).astype(float)
        feat['downtrend_flag'] = (feat['lh_ll_score'] > 0.6).astype(float) * -1

        # Swing-range position
        sh_ser = pd.Series(np.nan, index=df.index)
        sl_ser = pd.Series(np.nan, index=df.index)
        for i in sh_idx:
            sh_ser.iloc[i] = highs[i]
        for i in sl_idx:
            sl_ser.iloc[i] = lows[i]

        sh_ser = sh_ser.ffill()
        sl_ser = sl_ser.ffill()

        rng = (sh_ser - sl_ser).replace(0, np.nan)
        feat['swing_position'] = (
            (pd.Series(close, index=df.index) - sl_ser) / rng
        ).clip(0, 1)

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_slope(self, feat: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """
        Block 5 — Linear-Regression Slope & Trend Quality.

        Features:
            price_slope       : rolling OLS slope normalised by price level
            trend_strength    : R² of regression (0=random, 1=perfect trend)
            trend_consistency : rolling mean of log-return signs (-1…+1)
        """
        win = self.cfg.slope_window
        n = len(close)
        slopes = np.full(n, np.nan)
        r2 = np.full(n, np.nan)

        x = np.arange(win, dtype=float)
        x_mean = x.mean()
        x_dem = x - x_mean
        ss_x = (x_dem ** 2).sum()

        for i in range(win - 1, n):
            y = close.iloc[i - win + 1: i + 1].values.astype(float)
            y_mean = y.mean()
            y_dem = y - y_mean
            slope = (x_dem * y_dem).sum() / (ss_x + 1e-10)
            slopes[i] = slope / (y_mean + 1e-10)

            ss_y = (y_dem ** 2).sum()
            ss_res = ss_y - slope * (x_dem * y_dem).sum()
            r2[i] = 1.0 - max(ss_res, 0.0) / (ss_y + 1e-10)

        feat['price_slope'] = slopes
        feat['trend_strength'] = r2

        log_ret = np.log(close / close.shift(1).replace(0, np.nan))
        feat['trend_consistency'] = np.sign(log_ret).rolling(win, min_periods=1).mean()

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_volume_trend(
        self, feat: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Block 6 — Volume-Trend Confirmation.

        Feature:
            volume_trend : rolling OBV slope normalised by OBV magnitude
                          (positive = rising volume supporting trend)

        If Volume column is missing, feature is set to 0.
        """
        if 'Volume' not in df.columns:
            self.logger.warning("No 'Volume' column — volume_trend set to 0")
            feat['volume_trend'] = 0.0
            return feat

        win = self.cfg.slope_window
        direction = np.sign(df['Close'].diff().fillna(0))
        obv = (direction * df['Volume']).cumsum()
        obv_mag = obv.abs().rolling(win, min_periods=1).mean().replace(0, np.nan)

        feat['volume_trend'] = obv.diff(win) / obv_mag

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_cross_timeframe(
        self, feat: pd.DataFrame, close: pd.Series
    ) -> pd.DataFrame:
        """
        Block 7 — Cross-Timeframe Trend Structure.

        Evaluates higher-timeframe trend alignment.

        Features (per tf_bars in [4, 24, 96]):
            htf_momentum_<tf>    : return over tf bars
            htf_sign_<tf>        : direction of HTF momentum
            htf_alignment_<tf>   : +1 if current & HTF trend agree
            htf_distance_ma_<tf> : (Close - MA_tf) / MA_tf
            htf_persistence_<tf> : fraction of last 5 bars above HTF MA
        """
        current_ret = close.pct_change(5)

        for tf in [4, 24, 96]:
            prev = close.shift(tf).replace(0, np.nan)
            htf_ret = (close / prev) - 1

            feat[f'htf_momentum_{tf}'] = htf_ret
            feat[f'htf_sign_{tf}'] = np.sign(htf_ret)
            feat[f'htf_alignment_{tf}'] = (
                np.sign(current_ret) == np.sign(htf_ret)
            ).astype(float)

            htf_ma = close.rolling(tf, min_periods=1).mean()
            safe_ma = htf_ma.replace(0, np.nan)
            feat[f'htf_distance_ma_{tf}'] = (close - htf_ma) / safe_ma
            feat[f'htf_persistence_{tf}'] = (
                (close > htf_ma).rolling(5, min_periods=1).mean()
            )

        return feat

    # ──────────────────────────────────────────────────────────────────────────

    def _block_volatility_context(
        self, feat: pd.DataFrame, close: pd.Series, atr: pd.Series
    ) -> pd.DataFrame:
        """
        Block 8 — Volatility Context for Trend.

        Features:
            volatility_regime : rolling ATR percentile rank (0-1)
                               (1 = high vol, 0 = low vol)
            trend_quality     : ADX / volatility_regime
                               (strong trend with low noise)
        """
        win = self.cfg.norm_window

        # ATR percentile rank over rolling window
        feat['volatility_regime'] = atr.rolling(win, min_periods=1).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5,
            raw=False
        )

        # Trend quality = high ADX + low volatility
        if 'adx' in feat.columns:
            feat['trend_quality'] = feat['adx'] / (feat['volatility_regime'] + 0.1)
        else:
            feat['trend_quality'] = 0.0

        return feat

    # ──────────────────────────────────────────────────────────────────────────
    # NORMALISATION
    # ──────────────────────────────────────────────────────────────────────────

    def _normalise_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rolling normalisation to all features.

        Strategy:
            - Z-score: ratio/cross/slope features (unbounded)
            - Min-max: bounded oscillators (ADX 0-100, scores 0-1) → [-1, 1]
            - Clip: binary flags already in {-1, 0, +1}
        """
        win = self.cfg.norm_window

        # Z-score normalization for unbounded features
        zscore_cols = [
            'sma_20', 'sma_50', 'sma_200',
            'ema_9', 'ema_21', 'ema_50',
            'sma_cross_20_50', 'sma_cross_50_200',
            'ema_cross_9_21', 'ema_cross_50_200',
            'di_diff', 'price_slope', 'trend_consistency',
            'volume_trend', 'trend_quality'
        ]

        # Include cross-timeframe features
        for tf in [4, 24, 96]:
            zscore_cols.extend([
                f'htf_momentum_{tf}',
                f'htf_distance_ma_{tf}',
                f'htf_persistence_{tf}'
            ])

        for col in zscore_cols:
            if col in feat.columns:
                feat[col] = self._rolling_zscore(feat[col], win)

        # Min-max normalization for bounded features
        minmax_cols = [
            'adx', 'adx_14', 'di_plus', 'di_minus', 'atr_pct',
            'hh_hl_score', 'lh_ll_score', 'swing_position',
            'trend_strength', 'volatility_regime'
        ]

        for col in minmax_cols:
            if col in feat.columns:
                feat[col] = self._rolling_minmax(feat[col], win)

        # Clip binary flags
        binary_cols = [
            'golden_cross', 'death_cross',
            'hh_signal', 'hl_signal', 'lh_signal', 'll_signal',
            'uptrend_flag', 'downtrend_flag'
        ]

        # Include HTF alignment and sign
        for tf in [4, 24, 96]:
            binary_cols.extend([f'htf_sign_{tf}', f'htf_alignment_{tf}'])

        for col in binary_cols:
            if col in feat.columns:
                feat[col] = feat[col].clip(-1, 1)

        return feat.ffill().bfill().fillna(0.0)
    
    
    
    





# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CLASSES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CompositeWeights:
    """Weights for composite features (if enabled)."""
    vv_alignment: float = 0.4
    participation: float = 0.3
    body_volume: float = 0.3
    
    def __post_init__(self):
        total = self.vv_alignment + self.participation + self.body_volume
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    atr_periods: List[int] = None
    vol_windows: List[int] = None
    bb_period: int = 20
    bb_std: float = 2.0
    volume_ma_periods: List[int] = None
    volume_zscore_window: int = 20
    obv_periods: List[int] = None
    regime_window: int = 100
    enable_mtf: bool = False
    enable_composites: bool = False  # NEW: Disabled by default
    composite_weights: Optional[CompositeWeights] = None
    
    def __post_init__(self):
        # Set defaults
        if self.atr_periods is None:
            self.atr_periods = [7, 14, 21]
        if self.vol_windows is None:
            self.vol_windows = [10, 20, 50]
        if self.volume_ma_periods is None:
            self.volume_ma_periods = [10, 20, 50]
        if self.obv_periods is None:
            self.obv_periods = [5, 10, 20]
        if self.composite_weights is None:
            self.composite_weights = CompositeWeights()


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY CLASSES
# ═════════════════════════════════════════════════════════════════════════════

class FeatureNormalizer:
    """Centralized normalization methods (all vectorized)."""
    
    @staticmethod
    def safe_divide(numerator: pd.Series, denominator: pd.Series, 
                    fill_value: float = np.nan) -> pd.Series:
        """Division with automatic zero handling."""
        denom_safe = denominator.replace(0, np.nan)
        return numerator / denom_safe
    
    @staticmethod
    def z_score(series: pd.Series, window: int = 20, 
                min_periods: Optional[int] = None) -> pd.Series:
        """Rolling z-score normalization."""
        if min_periods is None:
            min_periods = max(2, window // 2)
        
        mean = series.rolling(window=window, min_periods=min_periods).mean()
        std = series.rolling(window=window, min_periods=min_periods).std()
        return FeatureNormalizer.safe_divide(series - mean, std)
    
    @staticmethod
    def percentile_rank_vectorized(series: pd.Series, window: int = 100,
                                   min_periods: int = 20) -> pd.Series:
        """
        Vectorized rolling percentile rank.
        Much faster than .apply(lambda x: rank()).
        """
        def rank_pct(arr):
            """Compute percentile rank of last element."""
            if len(arr) < 2:
                return np.nan
            return (arr[-1] > arr[:-1]).sum() / (len(arr) - 1)
        
        return series.rolling(window=window, min_periods=min_periods).apply(
            rank_pct, raw=True, engine='numba'  # Use numba for speed
        )
    
    @staticmethod
    def min_max_scale(series: pd.Series, window: int = 20,
                     min_periods: Optional[int] = None) -> pd.Series:
        """Rolling min-max normalization to [0, 1]."""
        if min_periods is None:
            min_periods = max(2, window // 2)
        
        min_val = series.rolling(window=window, min_periods=min_periods).min()
        max_val = series.rolling(window=window, min_periods=min_periods).max()
        return FeatureNormalizer.safe_divide(series - min_val, max_val - min_val)
    
    @staticmethod
    def ratio_to_ma(series: pd.Series, window: int = 20,
                   min_periods: Optional[int] = None) -> pd.Series:
        """Ratio to moving average."""
        if min_periods is None:
            min_periods = max(1, window // 2)
        
        ma = series.rolling(window=window, min_periods=min_periods).mean()
        return FeatureNormalizer.safe_divide(series, ma)


class VectorizedHelpers:
    """Vectorized helper functions for complex operations."""
    
    @staticmethod
    def consecutive_count(condition: pd.Series) -> pd.Series:
        """
        Count consecutive True values.
        
        Example:
            [T, T, F, T, T, T] -> [1, 2, 0, 1, 2, 3]
        """
        # Create groups when condition changes
        groups = (condition != condition.shift()).cumsum()
        # Count within each group, but only for True values
        return condition.groupby(groups).cumsum()
    
    @staticmethod
    def bars_since_event(event: pd.Series) -> pd.Series:
        """
        Count bars since last True event.
        
        Example:
            [F, T, F, F, T, F] -> [nan, 0, 1, 2, 0, 1]
        """
        # Create event groups
        event_groups = event.cumsum()
        # Count within each group, but invert
        return (~event).groupby(event_groups).cumsum()
    
    @staticmethod
    def rolling_rank_last(series: pd.Series, window: int, 
                         min_periods: int = 20) -> pd.Series:
        """
        Optimized rolling rank of last value (percentile).
        Uses pure NumPy for speed.
        """
        result = np.full(len(series), np.nan)
        arr = series.values
        
        for i in range(min_periods - 1, len(arr)):
            start_idx = max(0, i - window + 1)
            window_data = arr[start_idx:i+1]
            
            if len(window_data) >= min_periods:
                # Percentile rank of last element
                result[i] = (window_data[-1] > window_data[:-1]).sum() / (len(window_data) - 1)
        
        return pd.Series(result, index=series.index)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class VolatilityVolumeFeatureEngine:
   

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Args:
            config: FeatureConfig object with all parameters
        """
        self.config = config or FeatureConfig()
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Suppress numba warnings
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    
    # ═════════════════════════════════════════════════════════════════════════
    # MAIN COMPUTE METHOD
    # ═════════════════════════════════════════════════════════════════════════

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
       
       
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(c in df.columns for c in required):
            raise ValueError(f"DataFrame must contain {required}")
        
        self.logger.info("Computing optimized volatility-volume features...")
        self.logger.info(f"  Input rows: {len(df)}")
        self.logger.info(f"  Composite features: {'ENABLED' if self.config.enable_composites else 'DISABLED'}")
        
        df = df.copy()
        
        df = self._block_session_time(df)
        # Block 1: ATR Features
        df = self._compute_atr_features(df)
        
        # Block 2: Realized Volatility
        df = self._compute_realized_volatility(df)
        
        # Block 3: Bollinger Bands
        df = self._compute_bollinger_bands(df)
        
        # Block 4: Volatility Regime
        df = self._compute_volatility_regime(df)
        
        # Block 5: Volume Normalization
        df = self._compute_volume_normalization(df)
        
        # Block 6: Volume Momentum
        df = self._compute_volume_momentum(df)
        
        # Block 7: OBV Features
        df = self._compute_obv_features(df)
        
        # Block 8: Volume-Price Interaction
        df = self._compute_volume_price_interaction(df)
        
        # Block 9: Advanced Volatility-Volume Interactions
        df = self._compute_advanced_interactions(df)
        
        # Block 10: Range & Body Analysis
        df = self._compute_range_body_features(df)
        
        # Block 11: Multi-Timeframe Context (only if data provided)
        if self.config.enable_mtf:
            self.logger.warning("MTF enabled but not implemented - skipping")
        
        # Block 12: Rolling Regime Statistics
        df = self._compute_regime_statistics(df)
        
        # Block 13: Transition Features
        df = self._compute_transition_features(df)
        
        # Count features
        n_features = len([c for c in df.columns if c not in required])
        self.logger.info(f"  Features computed: {n_features}")
        self.logger.info(f"  Output rows: {len(df)}")
        
        return df

    
    def _block_session_time(
        self,df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Block 11 — Circular Session & Time Features.
        Circular encoding prevents discontinuity at midnight / week boundary.
        Session indicators capture liquidity-driven momentum patterns.
        """
        minutes = df.index.hour * 60 + df.index.minute
        df['time_sin'] = np.sin(2 * np.pi * minutes / 1440)
        df['time_cos'] = np.cos(2 * np.pi * minutes / 1440)

        dow = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        hour = df.index.hour
        df['session_tokyo']            = ((hour >= 0)  & (hour < 9)).astype(np.float32)
        df['session_london']           = ((hour >= 8)  & (hour < 16)).astype(np.float32)
        df['session_ny']               = ((hour >= 13) & (hour < 21)).astype(np.float32)
        df['session_overlap_london_ny']= ((hour >= 13) & (hour < 16)).astype(np.float32)

        return df
    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 1: ATR FEATURES
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_atr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR-based volatility expansion features (optimized)."""
        self.logger.debug("  Block 1: ATR features...")
        
        # True Range (vectorized)
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        
        for period in self.config.atr_periods:
            # ATR
            atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
            df[f'atr_{period}'] = atr
            
            # ATR slope (normalized)
            df[f'atr_slope_{period}'] = FeatureNormalizer.safe_divide(
                atr.diff(5), atr.shift(5)
            )
            
            # ATR expansion ratio
            atr_ma = atr.rolling(window=period*2, min_periods=period).mean()
            df[f'atr_expansion_ratio_{period}'] = FeatureNormalizer.safe_divide(
                atr, atr_ma
            )
            
            # ATR percentile rank (optimized)
            df[f'atr_pctrank_{period}'] = VectorizedHelpers.rolling_rank_last(
                atr, window=self.config.regime_window, min_periods=20
            )
        
        # Normalized ATR (z-score instead of price ratio)
        atr_14 = df[f'atr_{self.config.atr_periods[1]}']
        df['atr_zscore'] = FeatureNormalizer.z_score(atr_14, window=100)
        
        # Volatility breakout strength
        atr_50 = atr_14.rolling(window=50, min_periods=20).mean()
        df['volatility_breakout_strength'] = FeatureNormalizer.safe_divide(
            atr_14 - atr_50, atr_50
        )
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 2: REALIZED VOLATILITY
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Realized volatility measures (optimized)."""
        self.logger.debug("  Block 2: Realized volatility...")
        
        # Log returns
        log_returns = np.log(FeatureNormalizer.safe_divide(
            df['Close'], df['Close'].shift(1)
        ))
        
        for window in self.config.vol_windows:
            min_periods = max(2, window // 2)
            
            # Standard deviation of returns
            df[f'realized_volatility_{window}'] = (
                log_returns.rolling(window=window, min_periods=min_periods).std() 
                * np.sqrt(252)
            )
            
            # Parkinson volatility (vectorized)
            hl_ratio = np.log(FeatureNormalizer.safe_divide(df['High'], df['Low']))
            
            # Optimized Parkinson calculation
            hl_sq_sum = (hl_ratio ** 2).rolling(window=window, min_periods=min_periods).sum()
            df[f'parkinson_volatility_{window}'] = (
                np.sqrt(hl_sq_sum / (4 * window * np.log(2))) * np.sqrt(252)
            )
        
        # Garman-Klass volatility (most efficient OHLC estimator)
        hl = np.log(FeatureNormalizer.safe_divide(df['High'], df['Low'])) ** 2
        co = np.log(FeatureNormalizer.safe_divide(df['Close'], df['Open'])) ** 2
        gk = 0.5 * hl - (2 * np.log(2) - 1) * co
        
        df['garman_klass_volatility'] = (
            np.sqrt(gk.rolling(window=20, min_periods=10).mean()) * np.sqrt(252)
        )
        
        # Volatility of volatility
        rv_20 = df['realized_volatility_20']
        rv_mean = rv_20.rolling(window=20, min_periods=10).mean()
        rv_std = rv_20.rolling(window=20, min_periods=10).std()
        df['volatility_of_volatility'] = FeatureNormalizer.safe_divide(rv_std, rv_mean)
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 3: BOLLINGER BANDS
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands analysis (optimized)."""
        self.logger.debug("  Block 3: Bollinger Bands...")
        
        close = df['Close']
        period = self.config.bb_period
        min_periods = max(2, period // 2)
        
        # Bollinger Bands
        bb_ma = close.rolling(window=period, min_periods=min_periods).mean()
        bb_std = close.rolling(window=period, min_periods=min_periods).std()
        
        bb_upper = bb_ma + self.config.bb_std * bb_std
        bb_lower = bb_ma - self.config.bb_std * bb_std
        
        # BB width (normalized)
        bb_width = FeatureNormalizer.safe_divide(bb_upper - bb_lower, bb_ma)
        df['bb_width'] = bb_width
        
        # BB width percentile rank (optimized)
        df['bb_width_pctrank'] = VectorizedHelpers.rolling_rank_last(
            bb_width, window=self.config.regime_window, min_periods=20
        )
        
        # BB squeeze (width below 20th percentile)
        df['bb_squeeze'] = (df['bb_width_pctrank'] < 0.2).astype(int)
        
        # Price position in bands
        df['bb_position'] = FeatureNormalizer.safe_divide(
            close - bb_lower, bb_upper - bb_lower
        ).clip(0, 1)
        
        # BB width change
        df['bb_width_change'] = FeatureNormalizer.safe_divide(
            bb_width.diff(5), bb_width.shift(5)
        )
        
        # BB expansion strength
        bb_width_min = bb_width.rolling(window=20, min_periods=10).min()
        df['bb_expansion_strength'] = FeatureNormalizer.safe_divide(
            bb_width - bb_width_min, bb_width_min
        )
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 4: VOLATILITY REGIME
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility regime features (optimized)."""
        self.logger.debug("  Block 4: Volatility regime...")
        
        # Use ATR as volatility proxy
        atr_short = df[f'atr_{self.config.atr_periods[0]}']
        atr_long = df[f'atr_{self.config.atr_periods[-1]}']
        
        # Compression ratio
        df['volatility_compression_ratio'] = FeatureNormalizer.safe_divide(
            atr_short, atr_long
        )
        
        # Expansion velocity
        df['volatility_expansion_velocity'] = FeatureNormalizer.safe_divide(
            atr_short.diff(3), atr_short.shift(3)
        )
        
        # Volatility regime state (vectorized)
        comp_ratio = df['volatility_compression_ratio']
        df['volatility_regime_state'] = (
            (comp_ratio >= 0.8).astype(int) + 
            (comp_ratio > 1.2).astype(int)
        )
        
        # Range expansion ratio
        candle_range = df['High'] - df['Low']
        avg_range = candle_range.rolling(window=20, min_periods=10).mean()
        df['range_expansion_ratio'] = FeatureNormalizer.safe_divide(
            candle_range, avg_range
        )
        
        # True range expansion ratio
        atr_14 = df[f'atr_{self.config.atr_periods[1]}']
        tr_approx = atr_14 * self.config.atr_periods[1]
        tr_ma = tr_approx.rolling(window=20, min_periods=10).mean()
        df['tr_expansion_ratio'] = FeatureNormalizer.safe_divide(tr_approx, tr_ma)
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 5: VOLUME NORMALIZATION
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_volume_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalized volume features (optimized, no redundancy)."""
        self.logger.debug("  Block 5: Volume normalization...")
        
        volume = df['Volume']
        
        # Relative volume vs multiple periods
        for period in self.config.volume_ma_periods:
            min_periods = max(1, period // 2)
            vol_ma = volume.rolling(window=period, min_periods=min_periods).mean()
            df[f'relative_volume_{period}'] = FeatureNormalizer.safe_divide(
                volume, vol_ma
            )
        
        # Volume z-score
        df['volume_zscore'] = FeatureNormalizer.z_score(
            volume, window=self.config.volume_zscore_window
        )
        
        # Volume percentile rank (optimized)
        df['volume_pctrank'] = VectorizedHelpers.rolling_rank_last(
            volume, window=self.config.regime_window, min_periods=20
        )
        
        # Volume above average flag
        df['volume_above_avg'] = (df['relative_volume_20'] > 1.0).astype(int)
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 6: VOLUME MOMENTUM
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_volume_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume momentum and acceleration (optimized)."""
        self.logger.debug("  Block 6: Volume momentum...")
        
        volume = df['Volume']
        
        # Volume momentum
        df['volume_momentum_5'] = FeatureNormalizer.safe_divide(
            volume.diff(5), volume.shift(5)
        )
        
        # Volume acceleration
        df['volume_acceleration'] = df['volume_momentum_5'].diff(3)
        
        # Volume spike detection
        vol_ma_20 = volume.rolling(window=20, min_periods=10).mean()
        df['volume_spike_ratio'] = FeatureNormalizer.safe_divide(volume, vol_ma_20)
        df['volume_spike'] = (df['volume_spike_ratio'] > 2.0).astype(int)
        
        # Volume surge count
        vol_above_avg = (df['relative_volume_20'] > 1.0).astype(int)
        df['volume_surge_count'] = vol_above_avg.rolling(window=5, min_periods=3).sum()
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 7: OBV FEATURES (ENHANCED)
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_obv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """On-Balance Volume features with enhanced divergence detection."""
        self.logger.debug("  Block 7: OBV features (enhanced)...")
        
        # On-Balance Volume
        price_change = df['Close'].diff()
        volume_signed = df['Volume'] * np.sign(price_change)
        obv = volume_signed.cumsum()
        
        # Normalize OBV (z-score for stationarity)
        df['obv_zscore'] = FeatureNormalizer.z_score(obv, window=50)
        
        # OBV slope (multiple periods)
        for period in self.config.obv_periods:
            df[f'obv_slope_{period}'] = obv.diff(period) / period
        
        # OBV momentum (normalized)
        obv_ma = obv.rolling(window=20, min_periods=10).mean()
        df['obv_momentum'] = FeatureNormalizer.safe_divide(obv - obv_ma, obv_ma.abs())
        
        # Enhanced OBV-Price Divergence (multi-window)
        df['obv_divergence_strength'] = self._compute_divergence_strength(
            price=df['Close'],
            indicator=obv,
            windows=[5, 10, 20]
        )
        
        return df
    
    def _compute_divergence_strength(
        self, 
        price: pd.Series, 
        indicator: pd.Series,
        windows: List[int] = [5, 10, 20]
    ) -> pd.Series:
        """
        Multi-window divergence strength calculation.
        
        Returns:
            Positive values = bullish divergence (price down, indicator up)
            Negative values = bearish divergence (price up, indicator down)
            Zero = no divergence
        """
        divergence_scores = []
        
        for window in windows:
            # Rolling highs and lows
            price_high = price.rolling(window).max()
            price_low = price.rolling(window).min()
            ind_high = indicator.rolling(window).max()
            ind_low = indicator.rolling(window).min()
            
            # Bearish divergence: price makes new high, indicator doesn't
            price_new_high = (price >= price_high.shift(window))
            ind_fails_high = (indicator < ind_high.shift(window))
            bearish_div = price_new_high & ind_fails_high
            
            # Magnitude: how much did indicator fail by?
            ind_lag_high = FeatureNormalizer.safe_divide(
                ind_high.shift(window) - indicator,
                ind_high.shift(window).abs()
            )
            bearish_strength = bearish_div.astype(float) * ind_lag_high * -1
            
            # Bullish divergence: price makes new low, indicator doesn't
            price_new_low = (price <= price_low.shift(window))
            ind_fails_low = (indicator > ind_low.shift(window))
            bullish_div = price_new_low & ind_fails_low
            
            ind_lag_low = FeatureNormalizer.safe_divide(
                indicator - ind_low.shift(window),
                ind_low.shift(window).abs()
            )
            bullish_strength = bullish_div.astype(float) * ind_lag_low
            
            # Combine
            divergence_scores.append(bearish_strength + bullish_strength)
        
        # Average across windows
        return pd.concat(divergence_scores, axis=1).mean(axis=1)

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 8: VOLUME-PRICE INTERACTION
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_volume_price_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-price interaction features (optimized)."""
        self.logger.debug("  Block 8: Volume-price interaction...")
        
        # Volume pressure
        price_change_pct = df['Close'].pct_change()
        df['volume_pressure'] = df['relative_volume_20'] * price_change_pct
        
        # Money Flow Index
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        # Positive and negative money flow
        mf_pos = money_flow.where(typical_price > typical_price.shift(1), 0)
        mf_neg = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        mf_pos_sum = mf_pos.rolling(window=14, min_periods=7).sum()
        mf_neg_sum = mf_neg.rolling(window=14, min_periods=7).sum()
        
        mf_ratio = FeatureNormalizer.safe_divide(mf_pos_sum, mf_neg_sum)
        df['money_flow_index'] = 100 - (100 / (1 + mf_ratio))
        
        # VWAP deviation
        vwap_num = (typical_price * df['Volume']).rolling(window=20, min_periods=10).sum()
        vwap_den = df['Volume'].rolling(window=20, min_periods=10).sum()
        vwap = FeatureNormalizer.safe_divide(vwap_num, vwap_den)
        df['vwap_deviation'] = FeatureNormalizer.safe_divide(df['Close'] - vwap, vwap)
        
        # Volume-weighted momentum
        vol_weighted_change = price_change_pct * df['relative_volume_20']
        df['volume_weighted_momentum'] = (
            vol_weighted_change.rolling(window=10, min_periods=5).mean()
        )
        
        # Directional volume ratio
        up_volume = df['Volume'].where(df['Close'] > df['Close'].shift(1), 0)
        down_volume = df['Volume'].where(df['Close'] < df['Close'].shift(1), 0)
        
        up_vol_sum = up_volume.rolling(window=10, min_periods=5).sum()
        down_vol_sum = down_volume.rolling(window=10, min_periods=5).sum()
        
        df['directional_volume_ratio'] = FeatureNormalizer.safe_divide(
            up_vol_sum, down_vol_sum
        )
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 9: ADVANCED INTERACTIONS
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_advanced_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced volatility-volume interactions.
        Composite features only created if explicitly enabled.
        """
        self.logger.debug("  Block 9: Advanced interactions...")
        
        # Get base components
        atr_exp = df['atr_expansion_ratio_14']
        rel_vol = df['relative_volume_20']
        vol_zscore = df['volume_zscore']
        
        # ── Primitive Features (always computed) ────────────────────────────
        
        # Volatility-Volume Alignment
        df['volatility_volume_alignment'] = (atr_exp - 1.0) * (rel_vol - 1.0)
        
        # Participation Strength
        df['participation_strength'] = vol_zscore * df['volatility_breakout_strength']
        
        # Compression Release
        df['compression_release'] = df['bb_width_change'] * rel_vol
        
        # Volume Efficiency
        price_move = FeatureNormalizer.safe_divide(
            df['High'] - df['Low'], df['Close']
        )
        volume_norm = FeatureNormalizer.safe_divide(
            df['Volume'],
            df['Volume'].rolling(window=20, min_periods=10).mean()
        )
        df['volume_efficiency'] = FeatureNormalizer.safe_divide(
            price_move, volume_norm
        )
        
        # ── Composite Features (optional) ────────────────────────────────────
        
        if self.config.enable_composites:
            self.logger.debug("    Computing composite features...")
            
            # Expansion Quality Score
            candle_body = FeatureNormalizer.safe_divide(
                (df['Close'] - df['Open']).abs(), df['Open']
            )
            
            weights = self.config.composite_weights
            df['expansion_quality_score'] = (
                weights.vv_alignment * df['volatility_volume_alignment'] +
                weights.participation * df['participation_strength'] +
                weights.body_volume * (candle_body * rel_vol)
            )
            
            # Regime Quality (with retracement control)
            close_max_5 = df['Close'].rolling(window=5).max()
            close_min_5 = df['Close'].rolling(window=5).min()
            range_5 = close_max_5 - close_min_5
            
            retracement = FeatureNormalizer.safe_divide(
                close_max_5 - df['Close'], range_5
            ).clip(0, 1)
            
            df['regime_quality'] = (
                df['expansion_quality_score'] * (1 - retracement)
            )
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 10: RANGE & BODY ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_range_body_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candle range and body structure (optimized)."""
        self.logger.debug("  Block 10: Range & body analysis...")
        
        # Candle components
        candle_range = df['High'] - df['Low']
        candle_body = (df['Close'] - df['Open']).abs()
        upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
        lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
        
        # Body size (normalized)
        df['candle_body_pct'] = FeatureNormalizer.safe_divide(
            candle_body, df['Close']
        )
        
        # Body dominance
        df['body_dominance'] = FeatureNormalizer.safe_divide(
            candle_body, candle_range
        )
        
        # Shadow ratio
        df['shadow_ratio'] = FeatureNormalizer.safe_divide(
            upper_shadow, lower_shadow
        )
        
        # Candle strength
        atr_14 = df['atr_14']
        df['candle_strength'] = FeatureNormalizer.safe_divide(candle_body, atr_14)
        
        # Range expansion
        avg_range = candle_range.rolling(window=20, min_periods=10).mean()
        df['candle_range_ratio'] = FeatureNormalizer.safe_divide(
            candle_range, avg_range
        )
        
        # Directional candle strength
        directional_body = FeatureNormalizer.safe_divide(
            df['Close'] - df['Open'], df['Open']
        )
        df['directional_candle_strength'] = directional_body * df['relative_volume_20']
        
        # Wick percentage
        total_wicks = upper_shadow + lower_shadow
        df['wick_percentage'] = FeatureNormalizer.safe_divide(
            total_wicks, candle_range
        )
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 12: ROLLING REGIME STATISTICS
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_regime_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling regime statistics (optimized)."""
        self.logger.debug("  Block 12: Rolling regime statistics...")
        
        # Recent expansion frequency
        is_expanding = (df['atr_expansion_ratio_14'] > 1.2).astype(int)
        df['expansion_frequency_20'] = (
            is_expanding.rolling(window=20, min_periods=10).mean()
        )
        
        # Consecutive expansion bars (vectorized)
        df['expansion_streak'] = VectorizedHelpers.consecutive_count(is_expanding)
        
        # Mean expansion quality (only if composites enabled)
        if self.config.enable_composites and 'expansion_quality_score' in df.columns:
            df['mean_expansion_quality_20'] = (
                df['expansion_quality_score'].rolling(window=20, min_periods=10).mean()
            )
        
        # Volatility stability
        if 'volatility_of_volatility' in df.columns:
            df['volatility_stability'] = 1.0 / (1.0 + df['volatility_of_volatility'])
        
        # Volume consistency
        vol_mean = df['Volume'].rolling(window=20, min_periods=10).mean()
        vol_std = df['Volume'].rolling(window=20, min_periods=10).std()
        vol_cv = FeatureNormalizer.safe_divide(vol_std, vol_mean)
        df['volume_consistency'] = 1.0 / (1.0 + vol_cv)
        
        # Regime persistence (only if composites enabled)
        if self.config.enable_composites:
            df['regime_persistence'] = (
                df['expansion_frequency_20'] * 
                df.get('volatility_stability', 1.0) *
                df['volume_consistency']
            )
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 13: TRANSITION FEATURES
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_transition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transition and duration features (optimized)."""
        self.logger.debug("  Block 13: Transition features...")
        
        # ── BB Squeeze Duration (vectorized) ─────────────────────────────────
        is_squeezed = (df['bb_width_pctrank'] < 0.2).astype(int)
        df['bb_squeeze_duration'] = VectorizedHelpers.consecutive_count(is_squeezed)
        
        # Bars since squeeze ended
        squeeze_end = (is_squeezed.shift(1) == 1) & (is_squeezed == 0)
        df['bars_since_squeeze_end'] = VectorizedHelpers.bars_since_event(squeeze_end)
        
        # ── Volatility Compression Duration ──────────────────────────────────
        is_compressed = (df['atr_expansion_ratio_14'] < 0.9).astype(int)
        df['volatility_compression_duration'] = VectorizedHelpers.consecutive_count(
            is_compressed
        )
        
        # ── Time Since Last Expansion ────────────────────────────────────────
        is_expanded = (df['atr_expansion_ratio_14'] > 1.2).astype(int)
        expansion_event = (is_expanded.shift(1) == 0) & (is_expanded == 1)
        df['bars_since_expansion'] = VectorizedHelpers.bars_since_event(expansion_event)
        
        # ── Range Compression Duration ───────────────────────────────────────
        range_pctrank = VectorizedHelpers.rolling_rank_last(
            df['range_expansion_ratio'], window=100, min_periods=20
        )
        is_range_compressed = (range_pctrank < 0.2).astype(int)
        df['range_compression_duration'] = VectorizedHelpers.consecutive_count(
            is_range_compressed
        )
        
        # ── Volume Drought Duration ──────────────────────────────────────────
        is_low_volume = (df['relative_volume_20'] < 1.0).astype(int)
        df['volume_drought_duration'] = VectorizedHelpers.consecutive_count(
            is_low_volume
        )
        
        # ── Compression Intensity ────────────────────────────────────────────
        atr_14 = df['atr_14']
        atr_min_20 = atr_14.rolling(window=20, min_periods=10).min()
        df['compression_intensity'] = FeatureNormalizer.safe_divide(
            atr_14 - atr_min_20, atr_min_20
        )
        
        # ── Coiled Spring Score (only if composites enabled) ─────────────────
        if self.config.enable_composites:
            df['coiled_spring_score'] = (
                (df['bb_squeeze_duration'] / 20) *
                (1.0 - df['compression_intensity'].fillna(0)) *
                (df['volatility_compression_duration'] / 30)
            ).clip(0, 5)
        
        return df

