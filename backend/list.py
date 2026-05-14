class VolatilityVolumeFeatureEngine:
    """
    Volatility-Volume Feature Engineering for Regime Detection.
    
    Purpose:
        Generate features that detect tradable expansion regimes characterized by:
            ✓ Healthy volatility expansion
            ✓ Strong volume participation
            ✓ Sustainable momentum conditions
            ✓ Low noise and false breakout probability
    
    Feature blocks:
        Block  1: ATR Features (expansion, slope, breakout strength)
        Block  2: Realized Volatility (rolling std, Parkinson, Garman-Klass)
        Block  3: Bollinger Bands (width, percentile rank, squeeze detection)
        Block  4: Volatility Regime (compression ratio, expansion velocity)
        Block  5: Volume Normalization (relative, z-score, percentile)
        Block  6: Volume Momentum (acceleration, spike detection)
        Block  7: On-Balance Volume (OBV) Features
        Block  8: Volume-Price Interaction (pressure, divergence)
        Block  9: Advanced Volatility-Volume Interactions
        Block 10: Range & Body Analysis (candle structure)
        Block 11: Multi-Timeframe Context (optional, if data available)
        Block 12: Rolling Regime Statistics
        Block 13: Transition Features (NEW - predict regime shifts)
    
    All features are strictly causal (zero lookahead bias).
    """

    def __init__(
        self,
        atr_periods:        List[int] = [7, 14, 21],
        vol_windows:        List[int] = [10, 20, 50],
        bb_period:          int = 20,
        bb_std:             float = 2.0,
        volume_ma_periods:  List[int] = [10, 20, 50],
        volume_zscore_window: int = 20,
        obv_periods:        List[int] = [5, 10, 20],
        regime_window:      int = 100,
        enable_mtf:         bool = False,
    ):
        """
        Args:
            atr_periods:         Periods for ATR calculation
            vol_windows:         Windows for realized volatility
            bb_period:           Bollinger Bands period
            bb_std:              Bollinger Bands standard deviation multiplier
            volume_ma_periods:   Periods for volume moving averages
            volume_zscore_window: Window for volume z-score
            obv_periods:         Periods for OBV momentum
            regime_window:       Window for regime percentile calculations
            enable_mtf:          Enable multi-timeframe features (requires resampled data)
        """
        self.atr_periods = atr_periods
        self.vol_windows = vol_windows
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_ma_periods = volume_ma_periods
        self.volume_zscore_window = volume_zscore_window
        self.obv_periods = obv_periods
        self.regime_window = regime_window
        self.enable_mtf = enable_mtf
        
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            ))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)

    # ═════════════════════════════════════════════════════════════════════════
    # MAIN COMPUTE METHOD
    # ═════════════════════════════════════════════════════════════════════════

    def compute_features(
        self, 
        df: pd.DataFrame,
        drop_composite_scores: bool = False  # NEW: Force model to learn interactions
    ) -> pd.DataFrame:
        """
        Compute all volatility-volume features.
        
        Args:
            df: OHLCV DataFrame with DatetimeIndex
            drop_composite_scores: If True, exclude handcrafted composite features
                                  to force model to learn interactions from primitives
            
        Returns:
            DataFrame with all original columns + computed features
        """
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(c in df.columns for c in required):
            raise ValueError(f"DataFrame must contain {required}")
        
        self.logger.info("Computing volatility-volume features...")
        self.logger.info(f"  Input rows: {len(df)}")
        
        df = df.copy()
        
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
        
        # Block 11: Multi-Timeframe Context (optional)
        if self.enable_mtf:
            df = self._compute_mtf_context(df)
        
        # Block 12: Rolling Regime Statistics
        df = self._compute_regime_statistics(df)
        
        # Block 13: Transition Features (NEW)
        df = self._compute_transition_features(df)
        
        # ═════════════════════════════════════════════════════════════════
        # DROP COMPOSITE SCORES (if requested)
        # ═════════════════════════════════════════════════════════════════
        if drop_composite_scores:
            composite_features = [
                'expansion_quality_score',
                'regime_quality',
                'regime_persistence',
                'coiled_spring_score',
            ]
            
            dropped_count = 0
            for feat in composite_features:
                if feat in df.columns:
                    df.drop(columns=[feat], inplace=True)
                    dropped_count += 1
            
            if dropped_count > 0:
                self.logger.info(f"  ✓ Dropped {dropped_count} composite features")
                self.logger.info("    Model will learn interactions from primitives")
        
        n_features = len([c for c in df.columns if c not in required])
        self.logger.info(f"  Features computed: {n_features}")
        self.logger.info(f"  Output rows: {len(df)}")
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 1: ATR FEATURES
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_atr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ATR-based volatility expansion features.
        
        Features:
            - ATR (multiple periods)
            - ATR slope
            - ATR expansion ratio (current vs rolling mean)
            - ATR percentile rank
            - Volatility breakout strength
        """
        self.logger.debug("  Block 1: ATR features...")
        
        # True Range (zero lookahead)
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        
        for period in self.atr_periods:
            # ATR
            atr = tr.ewm(span=period, min_periods=period).mean()
            df[f'atr_{period}'] = atr
            
            # ATR slope
            df[f'atr_slope_{period}'] = atr.diff(5) / atr.shift(5).replace(0, np.nan)
            
            # ATR expansion ratio (current vs SMA)
            atr_ma = atr.rolling(window=period*2, min_periods=period).mean()
            df[f'atr_expansion_ratio_{period}'] = atr / atr_ma.replace(0, np.nan)
            
            # ATR percentile rank
            df[f'atr_pctrank_{period}'] = (
                atr.rolling(window=self.regime_window, min_periods=20)
                .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            )
        
        # Normalized ATR (ATR / close price)
        df['atr_normalised'] = df[f'atr_{self.atr_periods[1]}'] / df['Close']
        
        # Volatility breakout strength
        # = how much current volatility exceeds recent average
        atr_14 = df[f'atr_{self.atr_periods[1]}']
        atr_50 = atr_14.rolling(window=50, min_periods=20).mean()
        df['volatility_breakout_strength'] = (
            (atr_14 - atr_50) / atr_50.replace(0, np.nan)
        )
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 2: REALIZED VOLATILITY
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realized volatility measures.
        
        Features:
            - Rolling standard deviation of returns
            - Parkinson volatility (high-low range based)
            - Garman-Klass volatility (OHLC based)
            - Volatility of volatility
        """
        self.logger.debug("  Block 2: Realized volatility...")
        
        # Log returns
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        
        for window in self.vol_windows:
            # Standard deviation of returns
            df[f'realized_volatility_{window}'] = (
                log_returns.rolling(window=window, min_periods=int(window*0.5))
                .std() * np.sqrt(252)  # Annualized
            )
            
            # Parkinson volatility (uses high-low range)
            hl_ratio = np.log(df['High'] / df['Low'])
            df[f'parkinson_volatility_{window}'] = (
                hl_ratio.rolling(window=window, min_periods=int(window*0.5))
                .apply(lambda x: np.sqrt(np.sum(x**2) / (4 * len(x) * np.log(2))), raw=True)
                * np.sqrt(252)
            )
        
        # Garman-Klass volatility (most efficient OHLC estimator)
        hl = np.log(df['High'] / df['Low']) ** 2
        co = np.log(df['Close'] / df['Open']) ** 2
        gk = 0.5 * hl - (2 * np.log(2) - 1) * co
        df['garman_klass_volatility_20'] = (
            gk.rolling(window=20, min_periods=10).mean().apply(np.sqrt) * np.sqrt(252)
        )
        
        # Volatility of volatility (regime instability measure)
        rv_20 = df['realized_volatility_20']
        df['volatility_of_volatility'] = (
            rv_20.rolling(window=20, min_periods=10).std() / 
            rv_20.rolling(window=20, min_periods=10).mean().replace(0, np.nan)
        )
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 3: BOLLINGER BANDS
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bollinger Bands analysis.
        
        Features:
            - BB width (volatility measure)
            - BB width percentile rank
            - BB squeeze detection (width compression)
            - Price position in bands
            - BB expansion/contraction rate
        """
        self.logger.debug("  Block 3: Bollinger Bands...")
        
        close = df['Close']
        
        # Bollinger Bands
        bb_ma = close.rolling(window=self.bb_period, min_periods=int(self.bb_period*0.5)).mean()
        bb_std = close.rolling(window=self.bb_period, min_periods=int(self.bb_period*0.5)).std()
        
        bb_upper = bb_ma + self.bb_std * bb_std
        bb_lower = bb_ma - self.bb_std * bb_std
        
        # BB width (normalized by price)
        bb_width = (bb_upper - bb_lower) / bb_ma.replace(0, np.nan)
        df['bb_width'] = bb_width
        
        # BB width percentile rank (detects expansion vs compression)
        df['bb_width_pctrank'] = (
            bb_width.rolling(window=self.regime_window, min_periods=20)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        )
        
        # BB squeeze (width below 20th percentile = compression)
        df['bb_squeeze'] = (df['bb_width_pctrank'] < 0.2).astype(int)
        
        # Price position in bands (0 = lower, 0.5 = middle, 1 = upper)
        df['bb_position'] = (
            (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        ).clip(0, 1)
        
        # BB width change (expansion/contraction velocity)
        df['bb_width_change'] = bb_width.diff(5) / bb_width.shift(5).replace(0, np.nan)
        
        # BB expansion strength (width increase from recent low)
        bb_width_min = bb_width.rolling(window=20, min_periods=10).min()
        df['bb_expansion_strength'] = (
            (bb_width - bb_width_min) / bb_width_min.replace(0, np.nan)
        )
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 4: VOLATILITY REGIME
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility regime features.
        
        Features:
            - Compression ratio (recent volatility vs long-term)
            - Expansion velocity (rate of volatility increase)
            - Volatility regime state (compressed/normal/expanded)
            - Range expansion ratio
        """
        self.logger.debug("  Block 4: Volatility regime...")
        
        # Use ATR as volatility proxy
        atr_short = df[f'atr_{self.atr_periods[0]}']
        atr_long = df[f'atr_{self.atr_periods[-1]}']
        
        # Compression ratio (short-term vs long-term)
        df['volatility_compression_ratio'] = (
            atr_short / atr_long.replace(0, np.nan)
        )
        
        # Expansion velocity (rate of ATR increase)
        df['volatility_expansion_velocity'] = (
            atr_short.diff(3) / atr_short.shift(3).replace(0, np.nan)
        )
        
        # Volatility regime state
        # 0 = compressed (<0.8), 1 = normal (0.8-1.2), 2 = expanded (>1.2)
        comp_ratio = df['volatility_compression_ratio']
        df['volatility_regime_state'] = (
            (comp_ratio >= 0.8).astype(int) + 
            (comp_ratio > 1.2).astype(int)
        )
        
        # Range expansion ratio (today's range vs average)
        candle_range = df['High'] - df['Low']
        avg_range = candle_range.rolling(window=20, min_periods=10).mean()
        df['range_expansion_ratio'] = candle_range / avg_range.replace(0, np.nan)
        
        # True range expansion ratio
        tr = df[f'atr_{self.atr_periods[1]}'] * self.atr_periods[1]  # Approx TR sum
        tr_ma = tr.rolling(window=20, min_periods=10).mean()
        df['tr_expansion_ratio'] = tr / tr_ma.replace(0, np.nan)
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 5: VOLUME NORMALIZATION
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_volume_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalized volume features.
        
        CRITICAL: Never use raw volume — always normalize.
        
        Features:
            - Relative volume (vs multiple MAs)
            - Volume z-score
            - Volume percentile rank
            - Volume ratio (current / recent average)
        """
        self.logger.debug("  Block 5: Volume normalization...")
        
        volume = df['Volume']
        
        # Relative volume vs multiple periods
        for period in self.volume_ma_periods:
            vol_ma = volume.rolling(window=period, min_periods=int(period*0.5)).mean()
            df[f'relative_volume_{period}'] = volume / vol_ma.replace(0, np.nan)
        
        # Primary relative volume (20-period)
        df['relative_volume'] = df['relative_volume_20']
        
        # Volume z-score
        vol_mean = volume.rolling(window=self.volume_zscore_window, min_periods=10).mean()
        vol_std = volume.rolling(window=self.volume_zscore_window, min_periods=10).std()
        df['volume_zscore'] = (volume - vol_mean) / vol_std.replace(0, np.nan)
        
        # Volume percentile rank
        df['volume_pctrank'] = (
            volume.rolling(window=self.regime_window, min_periods=20)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        )
        
        # Volume ratio (simple current / 20-period MA)
        df['volume_ratio'] = df['relative_volume_20']
        
        # Volume above average flag
        df['volume_above_avg'] = (df['relative_volume'] > 1.0).astype(int)
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 6: VOLUME MOMENTUM
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_volume_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume momentum and acceleration.
        
        Features:
            - Volume momentum (rate of change)
            - Volume acceleration
            - Volume spike detection
            - Volume surge ratio
        """
        self.logger.debug("  Block 6: Volume momentum...")
        
        volume = df['Volume']
        
        # Volume momentum (5-bar ROC)
        df['volume_momentum_5'] = (
            volume.diff(5) / volume.shift(5).replace(0, np.nan)
        )
        
        # Volume acceleration (change in momentum)
        df['volume_acceleration'] = df['volume_momentum_5'].diff(3)
        
        # Volume spike detection (volume > 2x recent average)
        vol_ma_20 = volume.rolling(window=20, min_periods=10).mean()
        df['volume_spike_ratio'] = volume / vol_ma_20.replace(0, np.nan)
        df['volume_spike'] = (df['volume_spike_ratio'] > 2.0).astype(int)
        
        # Volume surge (sustained high volume)
        # = number of above-average volume bars in last 5 bars
        vol_above_avg = (df['relative_volume_20'] > 1.0).astype(int)
        df['volume_surge_count'] = vol_above_avg.rolling(window=5, min_periods=3).sum()
        
        # Tick volume acceleration (if available, else use Volume)
        df['tick_volume_acceleration'] = (
            volume.diff(3) - volume.diff(3).shift(3)
        ) / volume.shift(3).replace(0, np.nan)
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 7: OBV FEATURES
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_obv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        On-Balance Volume features.
        
        Features:
            - OBV
            - OBV slope (momentum)
            - OBV divergence vs price
            - OBV z-score
        """
        self.logger.debug("  Block 7: OBV features...")
        
        # On-Balance Volume
        price_change = df['Close'].diff()
        volume_signed = df['Volume'] * np.sign(price_change)
        obv = volume_signed.cumsum()
        df['obv'] = obv
        
        # OBV slope (multiple periods)
        for period in self.obv_periods:
            df[f'obv_slope_{period}'] = (
                obv.diff(period) / period
            )
        
        # OBV momentum (normalized)
        obv_ma = obv.rolling(window=20, min_periods=10).mean()
        df['obv_momentum_5'] = (
            (obv - obv_ma) / obv_ma.abs().replace(0, np.nan)
        )
        
        # OBV z-score
        obv_std = obv.rolling(window=20, min_periods=10).std()
        df['obv_zscore'] = (obv - obv_ma) / obv_std.replace(0, np.nan)
        
        # OBV-Price divergence
        # Price makes higher high but OBV doesn't (bearish divergence)
        # Price makes lower low but OBV doesn't (bullish divergence)
        close_5high = df['Close'].rolling(window=5).max()
        obv_5high = obv.rolling(window=5).max()
        
        price_higher_high = df['Close'] > close_5high.shift(5)
        obv_not_higher_high = obv <= obv_5high.shift(5)
        
        df['obv_bearish_divergence'] = (
            price_higher_high & obv_not_higher_high
        ).astype(int)
        
        close_5low = df['Close'].rolling(window=5).min()
        obv_5low = obv.rolling(window=5).min()
        
        price_lower_low = df['Close'] < close_5low.shift(5)
        obv_not_lower_low = obv >= obv_5low.shift(5)
        
        df['obv_bullish_divergence'] = (
            price_lower_low & obv_not_lower_low
        ).astype(int)
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 8: VOLUME-PRICE INTERACTION
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_volume_price_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-price interaction features.
        
        Features:
            - Volume pressure (volume * price change)
            - Money flow (volume * typical price)
            - Volume-weighted price momentum
            - VWAP deviation
        """
        self.logger.debug("  Block 8: Volume-price interaction...")
        
        # Volume pressure (volume-weighted price change)
        price_change_pct = df['Close'].pct_change()
        df['volume_pressure'] = (
            df['relative_volume_20'] * price_change_pct
        )
        
        # Money Flow Index components
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        # Positive and negative money flow
        mf_pos = money_flow.where(typical_price > typical_price.shift(1), 0)
        mf_neg = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        mf_pos_sum = mf_pos.rolling(window=14, min_periods=7).sum()
        mf_neg_sum = mf_neg.rolling(window=14, min_periods=7).sum()
        
        mf_ratio = mf_pos_sum / mf_neg_sum.replace(0, np.nan)
        df['money_flow_index'] = 100 - (100 / (1 + mf_ratio))
        
        # VWAP (Volume-Weighted Average Price)
        vwap = (typical_price * df['Volume']).rolling(window=20, min_periods=10).sum() / \
               df['Volume'].rolling(window=20, min_periods=10).sum()
        df['vwap_deviation'] = (df['Close'] - vwap) / vwap.replace(0, np.nan)
        
        # Volume-weighted momentum
        # = average of (price change * relative volume) over N bars
        vol_weighted_change = price_change_pct * df['relative_volume_20']
        df['volume_weighted_momentum'] = (
            vol_weighted_change.rolling(window=10, min_periods=5).mean()
        )
        
        # Directional volume (up volume vs down volume)
        up_volume = df['Volume'].where(df['Close'] > df['Close'].shift(1), 0)
        down_volume = df['Volume'].where(df['Close'] < df['Close'].shift(1), 0)
        
        up_vol_sum = up_volume.rolling(window=10, min_periods=5).sum()
        down_vol_sum = down_volume.rolling(window=10, min_periods=5).sum()
        
        df['directional_volume_ratio'] = (
            up_vol_sum / down_vol_sum.replace(0, np.nan)
        )
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 9: ADVANCED VOLATILITY-VOLUME INTERACTIONS
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_advanced_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced volatility-volume interaction features.
        
        CRITICAL: These are the most important features for regime detection.
        
        Features:
            - Volatility-volume alignment (both high/low together)
            - Participation strength (volume * ATR expansion)
            - Compression release (volume surge after volatility compression)
            - Quality expansion score (volatility + volume + continuation)
        """
        self.logger.debug("  Block 9: Advanced interactions...")
        
        # Get base components
        atr_exp = df['atr_expansion_ratio_14']
        rel_vol = df['relative_volume_20']
        vol_zscore = df['volume_zscore']
        
        # Volatility-Volume Alignment
        # Both volatility and volume are elevated → quality expansion
        # Formula: (ATR expansion - 1) * (relative volume - 1)
        # Positive when both > 1, negative when both < 1
        df['volatility_volume_alignment'] = (
            (atr_exp - 1.0) * (rel_vol - 1.0)
        )
        
        # Participation Strength
        # = volume z-score * volatility breakout strength
        # High when volume is elevated during volatility breakout
        df['participation_strength'] = (
            vol_zscore * df['volatility_breakout_strength']
        )
        
        # Compression Release
        # = BB width change * relative volume
        # Detects volume surge during volatility expansion from compression
        df['compression_release'] = (
            df['bb_width_change'] * rel_vol
        )
        
        # Expansion Quality Score
        # Combines:
        #   - Volatility expansion
        #   - Volume participation
        #   - Price momentum
        candle_body = (df['Close'] - df['Open']).abs() / df['Open']
        
        df['expansion_quality_score'] = (
            0.4 * df['volatility_volume_alignment'] +
            0.3 * df['participation_strength'] +
            0.3 * (candle_body * rel_vol)
        )
        
        # Volume Efficiency
        # = price movement per unit of volume
        # High efficiency = good directional conviction
        price_move = (df['High'] - df['Low']) / df['Close']
        volume_norm = df['Volume'] / df['Volume'].rolling(window=20, min_periods=10).mean()
        df['volume_efficiency'] = price_move / volume_norm.replace(0, np.nan)
        
        # Regime Quality (expansion with controlled retracement)
        # = expansion strength * (1 - retracement ratio)
        atr_14 = df['atr_14']
        close_max_5 = df['Close'].rolling(window=5).max()
        close_min_5 = df['Close'].rolling(window=5).min()
        range_5 = close_max_5 - close_min_5
        
        # Retracement: how much of the 5-bar range was given back
        retracement = (close_max_5 - df['Close']) / range_5.replace(0, np.nan)
        retracement = retracement.clip(0, 1)
        
        df['regime_quality'] = (
            df['expansion_quality_score'] * (1 - retracement)
        )
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 10: RANGE & BODY ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_range_body_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Candle range and body structure.
        
        Features:
            - Body size ratio
            - Upper/lower shadow ratio
            - Body dominance (body vs full range)
            - Candle strength (body size relative to ATR)
        """
        self.logger.debug("  Block 10: Range & body analysis...")
        
        # Candle components
        candle_range = df['High'] - df['Low']
        candle_body = (df['Close'] - df['Open']).abs()
        upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
        lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
        
        # Body size (normalized by price)
        df['candle_body_pct'] = candle_body / df['Close']
        
        # Body dominance (body / full range)
        df['body_dominance'] = candle_body / candle_range.replace(0, np.nan)
        
        # Shadow ratio (upper / lower)
        df['shadow_ratio'] = upper_shadow / lower_shadow.replace(0, np.nan)
        
        # Candle strength (body size / ATR)
        atr_14 = df['atr_14']
        df['candle_strength'] = candle_body / atr_14.replace(0, np.nan)
        
        # Range expansion (candle range / average range)
        avg_range = candle_range.rolling(window=20, min_periods=10).mean()
        df['candle_range_ratio'] = candle_range / avg_range.replace(0, np.nan)
        
        # Directional strength (signed body size)
        directional_body = (df['Close'] - df['Open']) / df['Open']
        df['directional_candle_strength'] = directional_body * df['relative_volume_20']
        
        # Wick percentage (total wicks / full range)
        total_wicks = upper_shadow + lower_shadow
        df['wick_percentage'] = total_wicks / candle_range.replace(0, np.nan)
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 11: MULTI-TIMEFRAME CONTEXT (Optional)
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_mtf_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-timeframe regime context.
        
        NOTE: This requires pre-resampled higher timeframe data.
        If not available, this block is skipped.
        
        Features:
            - Higher timeframe ATR expansion
            - Higher timeframe volume
            - Cross-timeframe regime alignment
        """
        self.logger.debug("  Block 11: Multi-timeframe context (placeholder)...")
        
        # Placeholder: would require resampled data
        # For now, create dummy features
        df['htf_atr_expansion'] = 1.0
        df['htf_volume_ratio'] = 1.0
        df['mtf_regime_alignment'] = 0.5
        
        self.logger.warning("    MTF features not implemented — requires resampled data")
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 12: ROLLING REGIME STATISTICS
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_regime_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling regime statistics.
        
        Features:
            - Favorable regime frequency (recent history)
            - Expansion episode duration
            - Mean expansion quality (rolling)
            - Volatility stability
        """
        self.logger.debug("  Block 12: Rolling regime statistics...")
        
        # Recent expansion frequency
        # = % of bars with ATR expansion > 1.2 in last 20 bars
        is_expanding = (df['atr_expansion_ratio_14'] > 1.2).astype(int)
        df['expansion_frequency_20'] = (
            is_expanding.rolling(window=20, min_periods=10).mean()
        )
        
        # Consecutive expansion bars
        df['expansion_streak'] = (
            is_expanding.groupby((is_expanding != is_expanding.shift()).cumsum()).cumsum()
        )
        
        # Mean expansion quality (rolling)
        if 'expansion_quality_score' in df.columns:
            df['mean_expansion_quality_20'] = (
                df['expansion_quality_score'].rolling(window=20, min_periods=10).mean()
            )
        
        # Volatility stability (inverse of volatility-of-volatility)
        if 'volatility_of_volatility' in df.columns:
            df['volatility_stability'] = (
                1.0 / (1.0 + df['volatility_of_volatility'])
            )
        
        # Volume consistency (inverse of volume coefficient of variation)
        vol_mean = df['Volume'].rolling(window=20, min_periods=10).mean()
        vol_std = df['Volume'].rolling(window=20, min_periods=10).std()
        vol_cv = vol_std / vol_mean.replace(0, np.nan)
        df['volume_consistency'] = 1.0 / (1.0 + vol_cv)
        
        # Regime persistence score
        # = expansion frequency * volatility stability * volume consistency
        df['regime_persistence'] = (
            df['expansion_frequency_20'] * 
            df.get('volatility_stability', 1.0) *
            df['volume_consistency']
        )
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK 13: TRANSITION FEATURES (NEW)
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_transition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        BLOCK 13: Transition and Duration Features
        
        Predicts regime SHIFTS (compression → expansion).
        
        Features:
            - BB squeeze duration (bars in compression)
            - Volatility compression duration
            - Time since last expansion
            - Range compression percentile duration
            - Volume drought duration
        """
        self.logger.debug("  Block 13: Transition features...")
        
        # ── BB Squeeze Duration ──────────────────────────────────────────────
        # How long has BB been compressed?
        is_squeezed = (df['bb_width_pctrank'] < 0.2).astype(int)
        
        # Count consecutive squeeze bars
        df['bb_squeeze_duration'] = (
            is_squeezed.groupby((is_squeezed != is_squeezed.shift()).cumsum()).cumsum()
        )
        
        # Time since last squeeze ended
        squeeze_end = (is_squeezed.shift(1) == 1) & (is_squeezed == 0)
        df['bars_since_squeeze_end'] = (
            (~squeeze_end).groupby(squeeze_end.cumsum()).cumsum()
        )
        
        # ── Volatility Compression Duration ──────────────────────────────────
        # How long has ATR been below average?
        is_compressed = (df['atr_expansion_ratio_14'] < 0.9).astype(int)
        
        df['volatility_compression_duration'] = (
            is_compressed.groupby((is_compressed != is_compressed.shift()).cumsum()).cumsum()
        )
        
        # ── Time Since Last Expansion ────────────────────────────────────────
        # Bars since ATR expanded above 1.2x
        is_expanded = (df['atr_expansion_ratio_14'] > 1.2).astype(int)
        expansion_event = (is_expanded.shift(1) == 0) & (is_expanded == 1)
        
        df['bars_since_expansion'] = (
            (~expansion_event).groupby(expansion_event.cumsum()).cumsum()
        )
        
        # ── Range Compression Percentile Duration ────────────────────────────
        # How long has range been in bottom 20th percentile?
        range_pctrank = (
            df['range_expansion_ratio']
            .rolling(window=100, min_periods=20)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        )
        
        is_range_compressed = (range_pctrank < 0.2).astype(int)
        
        df['range_compression_duration'] = (
            is_range_compressed.groupby(
                (is_range_compressed != is_range_compressed.shift()).cumsum()
            ).cumsum()
        )
        
        # ── Volume Drought Duration ──────────────────────────────────────────
        # Consecutive bars with below-average volume
        is_low_volume = (df['relative_volume_20'] < 1.0).astype(int)
        
        df['volume_drought_duration'] = (
            is_low_volume.groupby((is_low_volume != is_low_volume.shift()).cumsum()).cumsum()
        )
        
        # ── Compression Intensity ────────────────────────────────────────────
        # How compressed is volatility right now? (deeper = bigger release)
        atr_min_20 = df['atr_14'].rolling(window=20, min_periods=10).min()
        df['compression_intensity'] = (
            (df['atr_14'] - atr_min_20) / atr_min_20.replace(0, np.nan)
        )
        
        # ── Transition Signal ────────────────────────────────────────────────
        # Combined: long compression + low intensity = coiled spring
        df['coiled_spring_score'] = (
            (df['bb_squeeze_duration'] / 20) *           # Normalize duration
            (1.0 - df['compression_intensity']) *        # Deeper compression
            (df['volatility_compression_duration'] / 30) # Sustained compression
        ).clip(0, 5)
        
        return df

    # ═════════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═════════════════════════════════════════════════════════════════════════

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of computed feature names (excluding OHLCV).
        """
        exclude = {'Open', 'High', 'Low', 'Close', 'Volume', 'Time', 'Date', 'Datetime'}
        return [c for c in df.columns if c not in exclude]

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Return feature groups for importance analysis.
        """
        return {
            'atr': [f'atr_{p}' for p in self.atr_periods] + 
                   [f'atr_slope_{p}' for p in self.atr_periods] +
                   [f'atr_expansion_ratio_{p}' for p in self.atr_periods],
            
            'volatility': [
                'volatility_breakout_strength',
                'volatility_compression_ratio',
                'volatility_expansion_velocity',
                'volatility_regime_state',
            ],
            
            'volume': [
                'relative_volume', 'volume_zscore', 'volume_pctrank',
                'volume_momentum_5', 'volume_spike_ratio',
            ],
            
            'bollinger': [
                'bb_width', 'bb_width_pctrank', 'bb_squeeze',
                'bb_expansion_strength', 'bb_width_change',
            ],
            
            'obv': [
                'obv', 'obv_momentum_5', 'obv_zscore',
                'obv_bearish_divergence', 'obv_bullish_divergence',
            ],
            
            'interactions': [
                'volatility_volume_alignment',
                'participation_strength',
                'compression_release',
                'expansion_quality_score',
                'regime_quality',
            ],
            
            'regime_stats': [
                'expansion_frequency_20',
                'regime_persistence',
                'volatility_stability',
                'volume_consistency',
            ],
            
            'transitions': [  # NEW
                'bb_squeeze_duration',
                'volatility_compression_duration',
                'bars_since_expansion',
                'range_compression_duration',
                'volume_drought_duration',
                'compression_intensity',
                'coiled_spring_score',
            ],
        }





"""
Optimized Volatility-Volume Feature Engineering for Regime Detection.

Version: 2.0
Optimizations:
    ✓ Vectorized operations (10-100x faster)
    ✓ Removed hardcoded composite weights
    ✓ Eliminated redundant features
    ✓ Standardized normalization
    ✓ Memory-efficient computation
    ✓ Enhanced divergence detection
    ✓ Configurable composite features
    ✓ Adaptive feature windows
"""

