# features/technical_indicators.py - FINAL PRODUCTION VERSION

import pandas as pd
import pandas_ta as ta # type: ignore
import numpy as np
import logging


class TechnicalFeatures:
    """
    Generate technical indicators - Pure pandas, no warnings
    """
    
    _warned_features = set()
    
    @staticmethod
    def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        df = df.copy()
        
        logger = logging.getLogger(__name__)
        
        # Ensure correct column names
        if 'close' in df.columns:
            df.rename(columns={
                'open': 'Open', 'high': 'High', 
                'low': 'Low', 'close': 'Close', 
                'volume': 'Volume'
            }, inplace=True)
        
        try:
            # ==========================================
            # TREND INDICATORS
            # ==========================================
            
            df['SMA_20'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / df['Close']
            df['SMA_50'] = (df['Close'] - df['Close'].rolling(window=50).mean()) / df['Close']
            df['SMA_200'] = (df['Close'] - df['Close'].rolling(window=200).mean()) / df['Close']
            
            df['EMA_9'] = (df['Close'].ewm(span=9, adjust=False, min_periods=1).mean()) / df['Close']
            df['EMA_21'] = (df['Close'].ewm(span=21, adjust=False, min_periods=1).mean()) / df['Close']
            df['EMA_50'] = (df['Close'].ewm(span=50, adjust=False, min_periods=1).mean()) / df['Close']
            
            # ==========================================
            # MACD
            # ==========================================
            
            ema_12 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
            macd_raw = ema_12 - ema_26
            df['MACD'] = (macd_raw / df['Close'])*100
            macd_signal_raw = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            df['MACD_signal'] = (macd_signal_raw / df['Close'])*100
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # ==========================================
            # RSI
            # ==========================================
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 0.0001)
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'] = df['RSI'].fillna(50)  # FIXED: No inplace
            
            # RSI 30
            gain_30 = (delta.where(delta > 0, 0)).rolling(window=30, min_periods=1).mean()
            loss_30 = (-delta.where(delta < 0, 0)).rolling(window=30, min_periods=1).mean()
            rs_30 = gain_30 / (loss_30 + 0.0001)
            df['RSI_30'] = 100 - (100 / (1 + rs_30))
            df['RSI_30'] = df['RSI_30'].fillna(50)  # FIXED: No inplace
            
            # ==========================================
            # STOCHASTIC
            # ==========================================
            
            low_14 = df['Low'].rolling(window=14, min_periods=1).min()
            high_14 = df['High'].rolling(window=14, min_periods=1).max()
            k_raw = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 0.0001)
            df['STOCH_K'] = ((k_raw/50.0)- 1.0).fillna(0)  # FIXED
            d_raw = df['STOCH_K'].rolling(window=3, min_periods=1).mean()
            df['STOCH_D'] = ((d_raw/50.0)- 1.0).fillna(0)  # FIXED
            df['Stoch_diff'] = df['STOCH_K'] - df['STOCH_D']
            
            # ==========================================
            # ADX
            # ==========================================
            
            high_diff = df['High'].diff()
            low_diff = -df['Low'].diff()
            
            pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift())
            tr3 = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(window=14, min_periods=1).mean()
            
            df['DI_plus'] = 100 * (pos_dm.rolling(window=14, min_periods=1).mean() / (atr + 0.0001))
            df['DI_minus'] = 100 * (neg_dm.rolling(window=14, min_periods=1).mean() / (atr + 0.0001))
            
            dx = 100 * abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus'] + 0.0001)
            df['ADX'] = dx.rolling(window=14, min_periods=1).mean()
            df['ADX'] = df['ADX'].fillna(25)  # FIXED
            df['DI_plus'] = df['DI_plus'].fillna(25)  # FIXED
            df['DI_minus'] = df['DI_minus'].fillna(25)  # FIXED
            
            # ==========================================
            # CCI
            # ==========================================
            
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(window=14, min_periods=1).mean()
            mad = (tp - sma_tp).abs().rolling(window=14, min_periods=1).mean()
            df['CCI'] = (tp - sma_tp) / (0.015 * mad + 0.0001)
            df['CCI'] = df['CCI'].fillna(0)  # FIXED
            
            # ==========================================
            # MOMENTUM
            # ==========================================
            
            df['MOM'] = df['Close'].diff(periods=10)
            df['MOM'] = df['MOM'].fillna(0)  # FIXED
            
            # ==========================================
            # WILLIAMS %R
            # ==========================================
            
            high_14_willr = df['High'].rolling(window=14, min_periods=1).max()
            low_14_willr = df['Low'].rolling(window=14, min_periods=1).min()
            df['WILLR'] = -100 * (high_14_willr - df['Close']) / (high_14_willr - low_14_willr + 0.0001)
            df['WILLR'] = df['WILLR'].fillna(-50)  # FIXED
            
            # ==========================================
            # ATR
            # ==========================================
            
            df['ATR'] = atr
            df['ATR'] = df['ATR'].fillna(df['Close'] * 0.001)  # FIXED
            
            df['NATR'] = 100 * df['ATR'] / df['Close']
            df['NATR'] = df['NATR'].fillna(1)  # FIXED
            
            # ==========================================
            # BOLLINGER BANDS
            # ==========================================
            
            df['BBANDS_middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
            std = df['Close'].rolling(window=20, min_periods=1).std()
            df['BBANDS_upper'] = df['BBANDS_middle'] + (2 * std)
            df['BBANDS_lower'] = df['BBANDS_middle'] - (2 * std)
            df['BBANDS_width'] = df['BBANDS_upper'] - df['BBANDS_lower']
            df['BBANDS_pct'] = (df['Close'] - df['BBANDS_lower']) / (df['BBANDS_upper'] - df['BBANDS_lower'] + 0.0001)
            
            # ==========================================
            # VOLUME INDICATORS
            # ==========================================
            
            # OBV
            obv_raw = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            df['OBV'] = (obv_raw - obv_raw.rolling(20).mean()) / (obv_raw.rolling(20).std() + 0.0001)
            
            # AD
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 0.0001)
            ad_raw= (clv * df['Volume']).cumsum()
            
            z_period = 20
            ad_mean = ad_raw.rolling(window=z_period, min_periods=1).mean()
            ad_std = ad_raw.rolling(window=z_period, min_periods=1).std()
            df['AD'] = (ad_raw - ad_mean) / (ad_std + 0.0001)
            
            # ADOSC
            ad_ema_3 = df['AD'].ewm(span=3, adjust=False, min_periods=1).mean()
            ad_ema_10 = df['AD'].ewm(span=10, adjust=False, min_periods=1).mean()
            df['ADOSC'] = ad_ema_3 - ad_ema_10
            
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            
            # MFI
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14, min_periods=1).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14, min_periods=1).sum()
            
            mfi_ratio = positive_flow / (negative_flow + 0.0001)
            df['MFI'] = 100 - (100 / (1 + mfi_ratio))
            df['MFI'] = df['MFI'].fillna(50)  # FIXED
            
            # ==========================================
            # CUSTOM FEATURES
            # ==========================================
            
            df['Price_ROC'] = df['Close'].pct_change(periods=10) * 100
            df['Price_ROC_5'] = df['Close'].pct_change(periods=5) * 100
            df['Volume_ROC'] = df['Volume'].pct_change(periods=10) * 100
            
            df['BB_position'] = (df['Close'] - df['BBANDS_lower']) / (df['BBANDS_upper'] - df['BBANDS_lower'] + 0.0001)
            df['Trend_Strength'] = (df['Close'] - df['SMA_50']) / (df['ATR'] + 0.0001)
            
            # Support/Resistance
            df['Swing_High'] = df['High'].rolling(window=20, min_periods=1).max()
            df['Swing_Low'] = df['Low'].rolling(window=20, min_periods=1).min()
            df['Distance_to_High'] = (df['Swing_High'] - df['Close']) / (df['Close'] + 0.0001) * 100
            df['Distance_to_Low'] = (df['Close'] - df['Swing_Low']) / (df['Close'] + 0.0001) * 100
            
            df['Volatility_Ratio'] = df['ATR'] / (df['Close'] + 0.0001) * 100
            
            # ==========================================
            # LAG FEATURES
            # ==========================================
            
            lag_periods = [1, 2, 3, 5, 10]
            
            for lag in lag_periods:
                df[f'Return_lag_{lag}'] = df['Close'].pct_change(periods=lag)
                df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
                df[f'RSI_lag_{lag}'] = df['RSI'].shift(lag)
                df[f'MACD_lag_{lag}'] = df['MACD'].shift(lag)
            
            # ==========================================
            # FINAL CLEANUP
            # ==========================================
            
            df = df.ffill().bfill().fillna(0)
            
        except Exception as e:
            logger.error(f"Error adding features: {e}", exc_info=True)
            # Ensure minimum features
            if 'RSI' not in df.columns:
                df['RSI'] = 50
            if 'ATR' not in df.columns:
                df['ATR'] = df['Close'] * 0.001
        
        return df
    
    @staticmethod
    def create_ml_features(df: pd.DataFrame) -> pd.DataFrame:
        """Select features for ML models"""
        
        # COMPLETE feature list (FIXED: Added missing features)
        feature_columns = [
            # Momentum
            'RSI', 'RSI_30', 'STOCH_K', 'STOCH_D', 'Stoch_diff', 'CCI', 'MOM', 'WILLR', 'MFI',
            
            # Trend
            'MACD', 'MACD_signal', 'MACD_hist', 'ADX', 'DI_plus', 'DI_minus',
            
            # Volatility
            'ATR', 'NATR', 'BBANDS_width', 'BBANDS_pct', 'Volatility_Ratio',
            
            # Bollinger Bands
            'BBANDS_upper', 'BBANDS_middle', 'BBANDS_lower',
            
            # Volume
            'OBV', 'AD', 'ADOSC', 'Volume_ROC',
            
            # Custom
            'BB_position', 'Price_ROC', 'Price_ROC_5', 'Trend_Strength',
            'Distance_to_High', 'Distance_to_Low',
            
            # Moving Averages (FIXED: Added all)
            'SMA_20', 'SMA_50', 'SMA_200', 'EMA_9', 'EMA_21', 'EMA_50',
            
            # Support/Resistance
            'Swing_High', 'Swing_Low',
            
            # Volume Indicators (FIXED: Added Volume_SMA_20)
            'Volume_SMA_20',
            
            # Lag features
            'Return_lag_1', 'Return_lag_2', 'Return_lag_3', 'Return_lag_5', 'Return_lag_10',
            'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_5', 'Volume_lag_10',
            'RSI_lag_1', 'RSI_lag_2', 'RSI_lag_3', 'RSI_lag_5', 'RSI_lag_10',
            'MACD_lag_1', 'MACD_lag_2', 'MACD_lag_3', 'MACD_lag_5', 'MACD_lag_10'
        ]
        
        # Create DataFrame with ALL features
        df_ml = pd.DataFrame(index=df.index)
        
        missing_features = []
        
        for col in feature_columns:
            if col in df.columns:
                df_ml[col] = df[col]
            else:
                df_ml[col] = 0.0
                missing_features.append(col)
        
        # Log missing features once
        if missing_features:
            if not all(f in TechnicalFeatures._warned_features for f in missing_features):
                logger = logging.getLogger(__name__)
                logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")
                TechnicalFeatures._warned_features.update(missing_features)
        
        # Fill NaN
        df_ml = df_ml.ffill().bfill().fillna(0)
        
        return df_ml
    
    
    @staticmethod
    def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        
        logger = logging.getLogger(__name__)
        
        
        try:
            df['SMA_20'] = (df['Close'] - df['Close'].rolling(window=20,min_periods=1).mean()) / df['Close']
            df['SMA_50'] = (df['Close'] - df['Close'].rolling(window=50,min_periods=1).mean()) / df['Close']
            df['SMA_200'] = (df['Close'] - df['Close'].rolling(window=200,min_periods=1).mean()) / df['Close']
            
           
            
            df['EMA_9'] = (df['Close'].ewm(span=9, adjust=False, min_periods=1).mean()) / df['Close']
            df['EMA_21'] = (df['Close'].ewm(span=21, adjust=False, min_periods=1).mean()) / df['Close']
            df['EMA_50'] = (df['Close'].ewm(span=50, adjust=False, min_periods=1).mean()) / df['Close']
            
            ema_9_raw = df['Close'].ewm(span=9, adjust=False, min_periods=1).mean()
            ema_21_raw = df['Close'].ewm(span=21, adjust=False, min_periods=1).mean()
            ema_50_raw = df['Close'].ewm(span=50, adjust=False, min_periods=1).mean()
            ema_200_raw = df['Close'].ewm(span=200, adjust=False, min_periods=1).mean()
             # EMA Crossover signals
             
            diff = ema_9_raw - ema_21_raw
            df['EMA_Cross_9_21'] = (diff - diff.rolling(window=50).mean()) / (diff.rolling(window=50).std() + 1e-10)
            diff_cross = ema_50_raw - ema_200_raw
            df['EMA_Cross_50_200'] = (diff_cross - diff_cross.rolling(window=50).mean()) / (diff_cross.rolling(window=50).std() + 1e-10)
            
          
            # ADX (Average Directional Index)

            
            # True Range components
            high_diff = df['High'].diff()
            low_diff = -df['Low'].diff()
            
            # Directional Movement
            pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            # True Range (TR)
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift())
            tr3 = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Average True Range (ATR)
            atr_14 = tr.rolling(window=14, min_periods=1).mean()
            
            # Directional Indicators
            df['DI_plus'] = 100 * (pos_dm.rolling(window=14, min_periods=1).mean() / (atr_14 + 1e-10))
            df['DI_minus'] = 100 * (neg_dm.rolling(window=14, min_periods=1).mean() / (atr_14 + 1e-10))
            
            # ADX calculation
            dx = 100 * abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus'] + 1e-10)
            df['ADX'] = dx.rolling(window=14, min_periods=1).mean()
            df['ADX_14'] = df['ADX']  # Alias for compatibility
            
            # Fill NaN with neutral values
            df['ADX'] = df['ADX'].fillna(25)
            df['ADX_14'] = df['ADX_14'].fillna(25)
            df['DI_plus'] = df['DI_plus'].fillna(25)
            df['DI_minus'] = df['DI_minus'].fillna(25)
            
           
            
        except Exception as e:
            logger.error(f"Error adding features: {e}", exc_info=True)
            # Ensure minimum features
            for col in ['SMA_20', 'SMA_50', 'EMA_50', 'MACD', 'ADX']:
                if col not in df.columns:
                    df[col] = 0.0
        
        df = df.iloc[50:].reset_index(drop=True)          
                    
        return df
    
    
    @staticmethod
    def create_trend_ml_features(df: pd.DataFrame) -> pd.DataFrame:
        """Select features for ML models"""
        
        # COMPLETE feature list (FIXED: Added missing features)
        feature_columns = [
            
            # Moving Averages (FIXED: Added all)
            'SMA_20', 'SMA_50', 'SMA_200', 'SMA_20_raw',
            
            'EMA_9', 'EMA_21', 'EMA_50',
            
            'EMA_Cross_9_21', 'EMA_Cross_50_200',
            
            'ADX', 'ADX_14', 'DI_plus', 'DI_minus'
            
           
            
            
              
           
        ]
        
        # Create DataFrame with ALL features
        df_ml = pd.DataFrame(index=df.index)
        
        missing_features = []
        
        for col in feature_columns:
            if col in df.columns:
                df_ml[col] = df[col]
            else:
                df_ml[col] = 0.0
                missing_features.append(col)
        
        # Log missing features once
        if missing_features:
            if not all(f in TechnicalFeatures._warned_features for f in missing_features):
                logger = logging.getLogger(__name__)
                logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")
                TechnicalFeatures._warned_features.update(missing_features)
        
        # Fill NaN
        df_ml = df_ml.ffill().bfill().fillna(0)
        
        return df_ml
    
    
    @staticmethod
    def create_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        
        logger = logging.getLogger(__name__)
        
        
        try:
            
            df['Momentum_ROC_5'] = df['Close'].pct_change(periods=5)
            df['Momentum_ROC_10'] = df['Close'].pct_change(periods=10)
            df['Momentum_ROC_20'] = df['Close'].pct_change(periods=20)
            
            # Raw Momentum (normalized by price)
            df['Momentum_Raw_5'] = df['Close'].diff(periods=5) / (df['Close'] + 1e-10)
            df['Momentum_Raw_10'] = df['Close'].diff(periods=10) / (df['Close'] + 1e-10)
            
            # Momentum Acceleration (2nd derivative)
            df['Momentum_Acceleration'] = df['Momentum_ROC_10'].diff(periods=5)
            
            
            
            # 2. RSI FEATURES (6 features)
            
            delta = df['Close'].diff()
            
            # RSI 14-period (standard)
            gain_14 = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss_14 = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs_14 = gain_14 / (loss_14 + 1e-10)
            df['RSI_14'] = (100 - (100 / (1 + rs_14))).fillna(50)
            
            # RSI 7-period (fast)
            gain_7 = delta.where(delta > 0, 0).rolling(window=7, min_periods=1).mean()
            loss_7 = (-delta.where(delta < 0, 0)).rolling(window=7, min_periods=1).mean()
            rs_7 = gain_7 / (loss_7 + 1e-10)
            df['RSI_7'] = (100 - (100 / (1 + rs_7))).fillna(50)
            
            # RSI 21-period (slow)
            gain_21 = delta.where(delta > 0, 0).rolling(window=21, min_periods=1).mean()
            loss_21 = (-delta.where(delta < 0, 0)).rolling(window=21, min_periods=1).mean()
            rs_21 = gain_21 / (loss_21 + 1e-10)
            df['RSI_21'] = (100 - (100 / (1 + rs_21))).fillna(50)
            
            # RSI Divergence (fast vs slow)
            df['RSI_Divergence'] = df['RSI_7'] - df['RSI_21']
            
            # RSI Trend (direction)
            df['RSI_Trend'] = df['RSI_14'].diff(periods=3)
            
            # RSI Momentum (rate of change)
            df['RSI_Momentum'] = df['RSI_14'].pct_change(periods=5)
            
            
            
            # 3. STOCHASTIC FEATURES (5 features)
           
            
            # Stochastic 14-period
            low_14 = df['Low'].rolling(window=14, min_periods=1).min()
            high_14 = df['High'].rolling(window=14, min_periods=1).max()
            df['STOCH_K_14'] = (100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-10)).fillna(50)
            
            # Stochastic %D (3-period SMA of %K)
            df['STOCH_D_14'] = df['STOCH_K_14'].rolling(window=3, min_periods=1).mean().fillna(50)
            
            # Fast Stochastic (5-period)
            low_5 = df['Low'].rolling(window=5, min_periods=1).min()
            high_5 = df['High'].rolling(window=5, min_periods=1).max()
            df['STOCH_K_5'] = (100 * (df['Close'] - low_5) / (high_5 - low_5 + 1e-10)).fillna(50)
            
            # Stochastic Divergence (%K - %D)
            df['STOCH_Divergence'] = df['STOCH_K_14'] - df['STOCH_D_14']
            
            # Stochastic Trend
            df['STOCH_Trend'] = df['STOCH_K_14'].diff(periods=3)
            
            
            # 4. CCI FEATURES (3 features)
        
            
            # Typical Price
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            
            # CCI 14-period
            sma_tp_14 = tp.rolling(window=14, min_periods=1).mean()
            mad_14 = (tp - sma_tp_14).abs().rolling(window=14, min_periods=1).mean()
            df['CCI_14'] = ((tp - sma_tp_14) / (0.015 * mad_14 + 1e-10)).fillna(0)
            
            # CCI 20-period
            sma_tp_20 = tp.rolling(window=20, min_periods=1).mean()
            mad_20 = (tp - sma_tp_20).abs().rolling(window=20, min_periods=1).mean()
            df['CCI_20'] = ((tp - sma_tp_20) / (0.015 * mad_20 + 1e-10)).fillna(0)
            
            # CCI Trend
            df['CCI_Trend'] = df['CCI_14'].diff(periods=3)
            
            
            
            # 5. WILLIAMS %R FEATURES (2 features)
            
            high_14_wr = df['High'].rolling(window=14, min_periods=1).max()
            low_14_wr = df['Low'].rolling(window=14, min_periods=1).min()
            df['WILLR_14'] = (-100 * (high_14_wr - df['Close']) / (high_14_wr - low_14_wr + 1e-10)).fillna(-50)
            
            # Williams %R 7-period (fast)
            high_7 = df['High'].rolling(window=7, min_periods=1).max()
            low_7 = df['Low'].rolling(window=7, min_periods=1).min()
            df['WILLR_7'] = (-100 * (high_7 - df['Close']) / (high_7 - low_7 + 1e-10)).fillna(-50)
            
            
        
            # 6. MOMENTUM STRENGTH (3 features)
            
            
            # Composite Momentum Strength
            rsi_strength = abs(df['RSI_14'] - 50) / 50
            stoch_strength = abs(df['STOCH_K_14'] - 50) / 50
            cci_strength = (abs(df['CCI_14']) / 200).clip(0, 1)
            
            df['Momentum_Strength_Composite'] = (rsi_strength + stoch_strength + cci_strength) / 3
            
            # Momentum Consistency (multi-timeframe agreement)
            roc_5_dir = np.sign(df['Momentum_ROC_5'])
            roc_10_dir = np.sign(df['Momentum_ROC_10'])
            roc_20_dir = np.sign(df['Momentum_ROC_20'])
            
            agreement = (
                (roc_5_dir == roc_10_dir).astype(int) +
                (roc_10_dir == roc_20_dir).astype(int) +
                (roc_5_dir == roc_20_dir).astype(int)
            )
            df['Momentum_Consistency'] = agreement / 3
            
            # Momentum Velocity (speed of change)
            df['Momentum_Velocity'] = df['Momentum_Strength_Composite'].diff(periods=3)
            
            
            
            # 7. DIVERGENCE FEATURES (2 features)
            
            
            # Price direction
            price_change = df['Close'].diff(periods=5)
            price_direction = np.sign(price_change)
            
            # RSI-Price Divergence
            rsi_change = df['RSI_14'].diff(periods=5)
            rsi_direction = np.sign(rsi_change)
            df['RSI_Price_Divergence'] = (rsi_direction - price_direction) / 2
            
            # Stochastic-Price Divergence
            stoch_change = df['STOCH_K_14'].diff(periods=5)
            stoch_direction = np.sign(stoch_change)
            df['STOCH_Price_Divergence'] = (stoch_direction - price_direction) / 2
            
            
            
            # 8. MOMENTUM ZONES (3 features)
            
            
            # RSI Zone (-1: oversold, 0: neutral, +1: overbought)
            df['RSI_Zone'] = np.where(
                df['RSI_14'] < 30, -1,
                np.where(df['RSI_14'] > 70, 1, 0)
            )
            
            # Stochastic Zone
            df['STOCH_Zone'] = np.where(
                df['STOCH_K_14'] < 20, -1,
                np.where(df['STOCH_K_14'] > 80, 1, 0)
            )
            
            # Composite Momentum Zone
            df['Momentum_Zone'] = (df['RSI_Zone'] + df['STOCH_Zone']) / 2
            
            
            
            # NORMALIZATION (always applied)
            
            
            # RSI: 0-100 → 0-1
            df['RSI_14'] = df['RSI_14'] / 100
            df['RSI_7'] = df['RSI_7'] / 100
            df['RSI_21'] = df['RSI_21'] / 100
            df['RSI_Divergence'] = df['RSI_Divergence'] / 100
            df['RSI_Trend'] = df['RSI_Trend'].clip(-50, 50) / 50
            df['RSI_Momentum'] = df['RSI_Momentum'].clip(-1, 1)
            
            # Stochastic: 0-100 → 0-1
            df['STOCH_K_14'] = df['STOCH_K_14'] / 100
            df['STOCH_D_14'] = df['STOCH_D_14'] / 100
            df['STOCH_K_5'] = df['STOCH_K_5'] / 100
            df['STOCH_Divergence'] = df['STOCH_Divergence'] / 100
            df['STOCH_Trend'] = df['STOCH_Trend'].clip(-50, 50) / 50
            
            # Williams %R: -100-0 → 0-1
            df['WILLR_14'] = (df['WILLR_14'] + 100) / 100
            df['WILLR_7'] = (df['WILLR_7'] + 100) / 100
            
            # CCI: unbounded → -1 to 1
            df['CCI_14'] = np.tanh(df['CCI_14'] / 150) 
            df['CCI_20'] = np.tanh(df['CCI_20'] / 150)
            df['CCI_Trend'] = np.tanh(df['CCI_Trend'] / 50)
            
            # ROC: % change → -1 to 1
            df['Momentum_ROC_5'] = df['Momentum_ROC_5'].clip(-0.1, 0.1) / 0.1
            df['Momentum_ROC_10'] = df['Momentum_ROC_10'].clip(-0.1, 0.1) / 0.1
            df['Momentum_ROC_20'] = df['Momentum_ROC_20'].clip(-0.1, 0.1) / 0.1
            
            # Raw Momentum
            df['Momentum_Raw_5'] = df['Momentum_Raw_5'].clip(-0.1, 0.1) / 0.1
            df['Momentum_Raw_10'] = df['Momentum_Raw_10'].clip(-0.1, 0.1) / 0.1
            
            # Acceleration
            df['Momentum_Acceleration'] = df['Momentum_Acceleration'].clip(-0.05, 0.05) / 0.05
            
            # Velocity
            df['Momentum_Velocity'] = df['Momentum_Velocity'].clip(-0.5, 0.5) / 0.5
            
    
            
        except Exception as e:
            logger.error(f"Error adding features: {e}", exc_info=True)
            # Ensure minimum features
            for col in ['Momentum_ROC_5', 'RSI_14', 'STOCH_K_14']:
                if col not in df.columns:
                    df[col] = 0.0
                    
                    
        return df
    
    
    @staticmethod
    def create_momentum_ml_features(df: pd.DataFrame) -> pd.DataFrame:
        """Select features for ML models"""
        
        # COMPLETE feature list (FIXED: Added missing features)
        feature_columns = [
            # Momentum
            'RSI_14', 'RSI_7', 'RSI_21', 'RSI_Divergence', 'RSI_Trend', 'RSI_Momentum',
            
            'STOCH_K_14', 'STOCH_D_14', 'STOCH_K_5', 'STOCH_Divergence', 'STOCH_Trend',
            
            'CCI_14', 'CCI_20', 'CCI_Trend',
            
            'WILLR_14', 'WILLR_7',
            
            'Momentum_ROC_5', 'Momentum_ROC_10', 'Momentum_ROC_20',
            
            'Momentum_Raw_5', 'Momentum_Raw_10',
            
            'Momentum_Acceleration', 'Momentum_Velocity',
            
            'Momentum_Strength_Composite', 'Momentum_Consistency', 'Momentum_Zone',
            
            'RSI_Price_Divergence', 'STOCH_Price_Divergence',
            
            'RSI_Zone', 'STOCH_Zone','Momentum_Zone'
        ]
        
        # Create DataFrame with ALL features
        df_ml = pd.DataFrame(index=df.index)
        
        missing_features = []
        
        for col in feature_columns:
            if col in df.columns:
                df_ml[col] = df[col]
            else:
                df_ml[col] = 0.0
                missing_features.append(col)
        
        # Log missing features once
        if missing_features:
            if not all(f in TechnicalFeatures._warned_features for f in missing_features):
                logger = logging.getLogger(__name__)
                logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")
                TechnicalFeatures._warned_features.update(missing_features)
        
        # Fill NaN
        df_ml = df_ml.ffill().bfill().fillna(0)
        
        return df_ml
    
    
    
    
    @staticmethod
    def create_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        
        logger = logging.getLogger(__name__)
        
        
        try:
            
            df['garch_log_return'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
            df['garch_squared_return'] = df['garch_log_return'] ** 2
            df['garch_abs_return'] = abs(df['garch_log_return'])
            df['garch_squared_return_lag1'] = df['garch_squared_return'].shift(1).fillna(0)
            df['garch_squared_return_lag5'] = df['garch_squared_return'].shift(5).fillna(0)
            
            
            # 2. ROLLING VARIANCE ESTIMATORS (6 features)
            
            
            df['garch_realized_var_5'] = df['garch_log_return'].rolling(window=5, min_periods=1).std() ** 2
            df['garch_realized_var_10'] = df['garch_log_return'].rolling(window=10, min_periods=1).std() ** 2
            df['garch_realized_var_20'] = df['garch_log_return'].rolling(window=20, min_periods=1).std() ** 2
            df['garch_realized_var_5'] = df['garch_realized_var_5'].fillna(1e-6)
            df['garch_realized_var_10'] = df['garch_realized_var_10'].fillna(1e-6)
            df['garch_realized_var_20'] = df['garch_realized_var_20'].fillna(1e-6)
            
            hl_ratio = np.log(df['High'] / (df['Low'] + 1e-10))
            df['garch_parkinson'] = (1 / (4 * np.log(2))) * (hl_ratio ** 2)
            df['garch_parkinson'] = df['garch_parkinson'].fillna(1e-6)
            
            hl_sq = 0.5 * (np.log(df['High'] / (df['Low'] + 1e-10)) ** 2)
            co_sq = (2 * np.log(2) - 1) * (np.log(df['Close'] / (df['Open'] + 1e-10)) ** 2)
            df['garch_garman_klass'] = hl_sq - co_sq
            df['garch_garman_klass'] = df['garch_garman_klass'].fillna(1e-6).clip(lower=0)
            
            rs1 = np.log(df['High'] / (df['Close'] + 1e-10)) * np.log(df['High'] / (df['Open'] + 1e-10))
            rs2 = np.log(df['Low'] / (df['Close'] + 1e-10)) * np.log(df['Low'] / (df['Open'] + 1e-10))
            df['garch_rogers_satchell'] = np.sqrt(rs1 + rs2)
            df['garch_rogers_satchell'] = df['garch_rogers_satchell'].fillna(1e-6).clip(lower=0)
            
            
           
            # 3. VOLATILITY CLUSTERING (4 features)
    
            
            df['garch_volatility_autocorr'] = df['garch_squared_return'].rolling(window=20, min_periods=1).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
            ).fillna(0)
            
            df['garch_volatility_persistence'] = df['garch_realized_var_5'] / (df['garch_realized_var_20'] + 1e-10)
            
            df['garch_conditional_variance'] = df['garch_squared_return'].ewm(
                alpha=0.06, adjust=False, min_periods=1
            ).mean()
            
            var_80th = df['garch_realized_var_20'].rolling(window=100, min_periods=1).quantile(0.8)
            df['garch_high_vol_regime'] = (df['garch_realized_var_20'] > var_80th).astype(float)
            
        
            # 4. ARCH EFFECTS (3 features)
            
            
            df['garch_arch1'] = df['garch_squared_return_lag1']
            df['garch_arch5'] = df['garch_squared_return'].rolling(window=5, min_periods=1).mean()
            df['garch_leverage_effect'] = (df['garch_log_return'] < 0).astype(float) * df['garch_abs_return']
            
            
            
            # 5. LONG MEMORY INDICATORS (4 features)
            
            df['garch_realized_range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
            
            abs_return_lag = df['garch_abs_return'].shift(1).fillna(0)
            df['garch_bipower_variation'] = (np.pi / 2) * df['garch_abs_return'] * abs_return_lag
            df['garch_bipower_variation'] = df['garch_bipower_variation'].rolling(window=10, min_periods=1).mean()
            
            df['garch_vol_of_vol'] = df['garch_realized_var_20'].rolling(window=20, min_periods=1).std().fillna(0)
            df['garch_volatility_trend'] = df['garch_realized_var_20'].diff(periods=5)
            
            
        
            # 6. DISTRIBUTION FEATURES (3 features)
            
            
            df['garch_skewness'] = df['garch_log_return'].rolling(window=20, min_periods=3).skew().fillna(0)
            df['garch_kurtosis'] = df['garch_log_return'].rolling(window=20, min_periods=4).kurt().fillna(0)
            
            def rolling_tail_risk(returns, window=50, threshold=2):
                def tail_pct(x):
                    if len(x) < 2:
                        return 0
                    std = x.std()
                    if std == 0:
                        return 0
                    extreme = abs(x) > (threshold * std)
                    return extreme.sum() / len(x)
                return returns.rolling(window=window, min_periods=2).apply(tail_pct, raw=False).fillna(0)
            
            df['garch_tail_risk'] = rolling_tail_risk(df['garch_log_return'], window=50, threshold=2)
            
            
            
            # NORMALIZATION
    
            
            df['garch_log_return'] = df['garch_log_return'].clip(-0.05, 0.05) / 0.05
            df['garch_squared_return'] = np.log1p(df['garch_squared_return'] * 10000) / 10
            df['garch_abs_return'] = df['garch_abs_return'].clip(0, 0.05) / 0.05
            df['garch_squared_return_lag1'] = np.log1p(df['garch_squared_return_lag1'] * 10000) / 10
            df['garch_squared_return_lag5'] = np.log1p(df['garch_squared_return_lag5'] * 10000) / 10
            df['garch_realized_var_5'] = np.log1p(df['garch_realized_var_5'] * 10000) / 10
            df['garch_realized_var_10'] = np.log1p(df['garch_realized_var_10'] * 10000) / 10
            df['garch_realized_var_20'] = np.log1p(df['garch_realized_var_20'] * 10000) / 10
            df['garch_parkinson'] = np.log1p(df['garch_parkinson'] * 10000) / 10
            df['garch_garman_klass'] = np.log1p(df['garch_garman_klass'] * 10000) / 10
            df['garch_rogers_satchell'] = df['garch_rogers_satchell'].clip(0, 0.1) / 0.1
            df['garch_volatility_autocorr'] = df['garch_volatility_autocorr'].clip(-1, 1)
            df['garch_volatility_persistence'] = df['garch_volatility_persistence'].clip(0, 5) / 5
            df['garch_conditional_variance'] = np.log1p(df['garch_conditional_variance'] * 10000) / 10
            df['garch_arch1'] = df['garch_arch1'].clip(0, 1)
            df['garch_arch5'] = np.log1p(df['garch_arch5'] * 10000) / 10
            df['garch_leverage_effect'] = df['garch_leverage_effect'].clip(0, 0.05) / 0.05
            df['garch_realized_range'] = df['garch_realized_range'].clip(0, 0.1) / 0.1
            df['garch_bipower_variation'] = np.log1p(df['garch_bipower_variation'] * 10000) / 10
            df['garch_vol_of_vol'] = np.log1p(df['garch_vol_of_vol'] * 100000) / 10
            
            trend_std = df['garch_volatility_trend'].rolling(window=50, min_periods=1).std()
            df['garch_volatility_trend'] = (df['garch_volatility_trend'] / (trend_std + 1e-10)).clip(-3, 3) / 3
            
            df['garch_skewness'] = df['garch_skewness'].clip(-3, 3) / 3
            df['garch_kurtosis'] = df['garch_kurtosis'].clip(0, 10) / 10
            
            
            
        
            
            
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill().fillna(0)
            
    
            
        except Exception as e:
            logger.error(f"Error adding features: {e}", exc_info=True)
            # Ensure minimum features
            for col in ['garch_log_return', 'garch_squared_return', 'garch_realized_var_20']:
                if col not in df.columns:
                    df[col] = 0.0
                    
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)           
        return df
    
    
    
    @staticmethod
    def create_volatility_ml_features(df: pd.DataFrame) -> pd.DataFrame:
        """Select features for ML models"""
        
        # COMPLETE feature list (FIXED: Added missing features)
        feature_columns = [
            # Volatility
            'garch_log_return', 'garch_squared_return', 'garch_abs_return',
            
            'garch_squared_return_lag1', 'garch_squared_return_lag5',  
             
            'garch_realized_var_5', 'garch_realized_var_10', 'garch_realized_var_20',
            
            'garch_parkinson', 'garch_garman_klass', 'garch_rogers_satchell',
            
            'garch_volatility_autocorr', 'garch_volatility_persistence', 'garch_conditional_variance', 'garch_high_vol_regime',
            
            'garch_arch1', 'garch_arch5', 'garch_leverage_effect',
            
            'garch_realized_range', 'garch_bipower_variation', 'garch_vol_of_vol', 'garch_volatility_trend',
            
            'garch_skewness', 'garch_kurtosis', 'garch_tail_risk'
            
        ]
        
        # Create DataFrame with ALL features
        df_ml = pd.DataFrame(index=df.index)
        
        missing_features = []
        
        for col in feature_columns:
            if col in df.columns:
                df_ml[col] = df[col]
            else:
                df_ml[col] = 0.0
                missing_features.append(col)
        
        # Log missing features once
        if missing_features:
            if not all(f in TechnicalFeatures._warned_features for f in missing_features):
                logger = logging.getLogger(__name__)
                logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")
                TechnicalFeatures._warned_features.update(missing_features)
        
        # Fill NaN
        df_ml = df_ml.ffill().bfill().fillna(0)
        
        return df_ml
    
    
    @staticmethod
    def create_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        
        logger = logging.getLogger(__name__) 
        
        try:
            df['vol_sma_10'] = df['Volume'].rolling(window=10, min_periods=1).mean()
            df['vol_sma_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            df['vol_sma_50'] = df['Volume'].rolling(window=50, min_periods=1).mean()
            
            df['vol_ratio_20'] = df['Volume'] / (df['vol_sma_20'] + 1e-10)
            
            df['vol_trend_direction'] = df['vol_sma_10'] / (df['vol_sma_50'] + 1e-10)
            
            df['vol_momentum'] = df['Volume'].pct_change(periods=5).fillna(0)
            
            
           
            # 2. ON-BALANCE VOLUME (OBV) (5 features)
           
            
            obv_raw = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            
            obv_mean = obv_raw.rolling(window=20, min_periods=1).mean()
            obv_std = obv_raw.rolling(window=20, min_periods=1).std()
            df['vol_obv'] = (obv_raw - obv_mean) / (obv_std + 1e-10)
            df['vol_obv'] = df['vol_obv'].fillna(0)
            
            df['vol_obv_ema_10'] = df['vol_obv'].ewm(span=10, adjust=False, min_periods=1).mean()
            df['vol_obv_ema_20'] = df['vol_obv'].ewm(span=20, adjust=False, min_periods=1).mean()
            
            df['vol_obv_divergence'] = df['vol_obv_ema_10'] - df['vol_obv_ema_20']
            
            df['vol_obv_trend'] = df['vol_obv'].diff(periods=5)
            
            
           
            # 3. ACCUMULATION/DISTRIBUTION (4 features)
           
            
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
            ad_raw = (clv * df['Volume']).cumsum()
            
            ad_mean = ad_raw.rolling(window=20, min_periods=1).mean()
            ad_std = ad_raw.rolling(window=20, min_periods=1).std()
            df['vol_ad'] = (ad_raw - ad_mean) / (ad_std + 1e-10)
            df['vol_ad'] = df['vol_ad'].fillna(0)
            
            ad_ema_3 = df['vol_ad'].ewm(span=3, adjust=False, min_periods=1).mean()
            ad_ema_10 = df['vol_ad'].ewm(span=10, adjust=False, min_periods=1).mean()
            df['vol_adosc'] = ad_ema_3 - ad_ema_10
            
            df['vol_ad_trend'] = df['vol_ad'].diff(periods=5)
            
            price_change = df['Close'].diff(periods=5)
            price_direction = np.sign(price_change)
            ad_change = df['vol_ad'].diff(periods=5)
            ad_direction = np.sign(ad_change)
            df['vol_ad_divergence'] = (ad_direction - price_direction) / 2
            
            
        
            # 4. MONEY FLOW INDEX (MFI) (4 features)
            
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            positive_mf_sum = positive_flow.rolling(window=14, min_periods=1).sum()
            negative_mf_sum = negative_flow.rolling(window=14, min_periods=1).sum()
            
            mfi_ratio = positive_mf_sum / (negative_mf_sum + 1e-10)
            df['vol_mfi'] = (100 - (100 / (1 + mfi_ratio))).fillna(50)
            
            df['vol_mfi_trend'] = df['vol_mfi'].diff(periods=3)
            
            mfi_change = df['vol_mfi'].diff(periods=5)
            mfi_direction = np.sign(mfi_change)
            df['vol_mfi_divergence'] = (mfi_direction - price_direction) / 2
            
            df['vol_mfi_zone'] = np.where(
                df['vol_mfi'] < 20, -1,
                np.where(df['vol_mfi'] > 80, 1, 0)
            )
            
            
           
            # 5. VOLUME PRICE INDICATORS (5 features)
           
            
            vwap_numerator = (typical_price * df['Volume']).rolling(window=20, min_periods=1).sum()
            vwap_denominator = df['Volume'].rolling(window=20, min_periods=1).sum()
            df['vol_vwap'] = vwap_numerator / (vwap_denominator + 1e-10)
            df['vol_vwap'] = df['vol_vwap'].fillna(df['Close'])
            df['vol_vwap_distance'] = (df['Close'] - df['vol_vwap']) / (df['Close'] + 1e-10)
            
            def rolling_correlation(x, y, window=20):
                return x.rolling(window=window, min_periods=1).corr(y).fillna(0)
            
            df['vol_price_corr'] = rolling_correlation(df['Volume'], df['Close'], window=20)
            
            price_returns = df['Close'].pct_change().fillna(0)
            price_volatility = price_returns.rolling(window=20, min_periods=1).std()
            volume_normalized = df['Volume'] / (df['vol_sma_20'] + 1e-10)
            df['vol_volatility_ratio'] = volume_normalized / (price_volatility + 1e-10)
            
            price_up = df['Close'] > df['Close'].shift(1)
            price_down = df['Close'] < df['Close'].shift(1)
            
            pvi = df['Volume'].copy()
            pvi[~price_up] = 0
            pvi_cumsum = pvi.cumsum()
            pvi_mean = pvi_cumsum.rolling(window=20, min_periods=1).mean()
            pvi_std = pvi_cumsum.rolling(window=20, min_periods=1).std()
            df['vol_pvi'] = (pvi_cumsum - pvi_mean) / (pvi_std + 1e-10)
            df['vol_pvi'] = df['vol_pvi'].fillna(0)
            
            nvi = df['Volume'].copy()
            nvi[~price_down] = 0
            nvi_cumsum = nvi.cumsum()
            nvi_mean = nvi_cumsum.rolling(window=20, min_periods=1).mean()
            nvi_std = nvi_cumsum.rolling(window=20, min_periods=1).std()
            df['vol_nvi'] = (nvi_cumsum - nvi_mean) / (nvi_std + 1e-10)
            df['vol_nvi'] = df['vol_nvi'].fillna(0)
            
            volume_pct = df['Volume'] / (df['Volume'].rolling(window=20, min_periods=1).sum() + 1e-10)
            df['vol_concentration'] = volume_pct.rolling(window=5, min_periods=1).sum()
            
            
            
            # 6. VOLUME PATTERNS (4 features)
            
            
            vol_std = df['Volume'].rolling(window=20, min_periods=1).std()
            df['vol_spike'] = (df['Volume'] > df['vol_sma_20'] + 2 * vol_std).astype(float)
            
            df['vol_dryup'] = (df['Volume'] < df['vol_sma_20'] * 0.5).astype(float)
            
            vol_5_dir = np.sign(df['Volume'].diff(periods=1))
            vol_10_dir = np.sign(df['vol_sma_10'].diff(periods=1))
            vol_20_dir = np.sign(df['vol_sma_20'].diff(periods=1))
            
            agreement = (
                (vol_5_dir == vol_10_dir).astype(int) +
                (vol_10_dir == vol_20_dir).astype(int) +
                (vol_5_dir == vol_20_dir).astype(int)
            )
            df['vol_consistency'] = agreement / 3
            
            df['vol_relative_strength'] = df['Volume'].rolling(window=20, min_periods=1).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) if len(x) > 1 else 0.5,
                raw=False
            ).fillna(0.5)
            
            
            
            # NORMALIZATION
            
            
            df['vol_sma_10'] = df['vol_sma_10'] / (df['vol_sma_50'] + 1e-10)
            df['vol_sma_10'] = df['vol_sma_10'].clip(0, 5) / 5
            
            df['vol_sma_20'] = df['vol_sma_20'] / (df['vol_sma_50'] + 1e-10)
            df['vol_sma_20'] = df['vol_sma_20'].clip(0, 5) / 5
            
            df['vol_sma_50'] = df['vol_sma_50'] / (df['vol_sma_50'].rolling(window=100, min_periods=1).mean() + 1e-10)
            df['vol_sma_50'] = df['vol_sma_50'].clip(0, 3) / 3
            
            df['vol_ratio_20'] = df['vol_ratio_20'].clip(0, 5) / 5
            
            df['vol_trend_direction'] = df['vol_trend_direction'].clip(0, 3) / 3
            
            df['vol_momentum'] = df['vol_momentum'].clip(-2, 2) / 2
            
            df['vol_obv'] = df['vol_obv'].clip(-3, 3) / 3
            df['vol_obv_ema_10'] = df['vol_obv_ema_10'].clip(-3, 3) / 3
            df['vol_obv_ema_20'] = df['vol_obv_ema_20'].clip(-3, 3) / 3
            df['vol_obv_divergence'] = df['vol_obv_divergence'].clip(-2, 2) / 2
            df['vol_obv_trend'] = df['vol_obv_trend'].clip(-2, 2) / 2
            
            df['vol_ad'] = df['vol_ad'].clip(-3, 3) / 3
            df['vol_adosc'] = df['vol_adosc'].clip(-2, 2) / 2
            df['vol_ad_trend'] = df['vol_ad_trend'].clip(-2, 2) / 2
            df['vol_ad_divergence'] = df['vol_ad_divergence'].clip(-1, 1)
            
            df['vol_mfi'] = df['vol_mfi'] / 100
            df['vol_mfi_trend'] = df['vol_mfi_trend'].clip(-50, 50) / 50
            df['vol_mfi_divergence'] = df['vol_mfi_divergence'].clip(-1, 1)
            
            df['vol_vwap_distance'] = df['vol_vwap_distance'].clip(-0.1, 0.1) / 0.1
            
            df['vol_price_corr'] = df['vol_price_corr'].clip(-1, 1)
            
            df['vol_volatility_ratio'] = df['vol_volatility_ratio'].clip(0, 10) / 10
            
            df['vol_pvi'] = df['vol_pvi'].clip(-3, 3) / 3
            df['vol_nvi'] = df['vol_nvi'].clip(-3, 3) / 3
            
            df['vol_concentration'] = df['vol_concentration'].clip(0, 1)
            
        except Exception as e:
            logger.error(f"Error adding features: {e}", exc_info=True)
            # Ensure minimum features
            for col in ['vol_sma_20', 'vol_obv', 'vol_ad']:
                if col not in df.columns:
                    df[col] = 0.0
                    
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)
                   
        return df
    
    
    @staticmethod
    def create_volume_ml_features(df: pd.DataFrame) -> pd.DataFrame:    
        """Select features for ML models"""
        
        # COMPLETE feature list (FIXED: Added missing features)
        feature_columns = [
            # Volume
            'vol_sma_10', 'vol_sma_20', 'vol_sma_50', 'vol_ratio_20', 'vol_trend_direction', 'vol_momentum',
            
            'vol_obv', 'vol_obv_ema_10', 'vol_obv_ema_20', 'vol_obv_divergence', 'vol_obv_trend',
            
            'vol_ad', 'vol_adosc', 'vol_ad_trend', 'vol_ad_divergence',
            
            'vol_mfi', 'vol_mfi_trend', 'vol_mfi_divergence', 'vol_mfi_zone',
            
            'vol_vwap_distance', 'vol_price_corr', 'vol_volatility_ratio', 'vol_pvi', 'vol_nvi', 'vol_concentration',
            
            'vol_spike', 'vol_dryup', 'vol_consistency', 'vol_relative_strength'
        ]
        
        # Create DataFrame with ALL features
        df_ml = pd.DataFrame(index=df.index)
        
        missing_features = []
        
        for col in feature_columns:
            if col in df.columns:
                df_ml[col] = df[col]
            else:
                df_ml[col] = 0.0
                missing_features.append(col)
        
        # Log missing features once
        if missing_features:
            if not all(f in TechnicalFeatures._warned_features for f in missing_features):
                logger = logging.getLogger(__name__)
                logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")
                TechnicalFeatures._warned_features.update(missing_features)
        
        # Fill NaN
        df_ml = df_ml.ffill().bfill().fillna(0)
        
        return df_ml