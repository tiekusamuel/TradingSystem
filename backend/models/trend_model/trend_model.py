
# models/ml_models/lstm_predictor.py - PRODUCTION VERSION V2.0

import numpy as np
import pandas as pd
import logging
import os
from typing import Tuple, Optional, Dict, List, Union
from datetime import datetime

from tensorflow import keras

from keras import layers, models, optimizers, callbacks, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import TimeSeriesSplit
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LSTMPredictor:
    """
    LSTM model for price direction prediction with comprehensive validation
    
    Features:
    - Data leakage prevention
    - Feature alignment validation
    - Class imbalance handling
    - Model versioning
    - Cross-validation support
    - Configurable architecture
    """
    
    def __init__(
        self,
        lookback: int = 220,
        features: int = 20,
        lstm_units: List[int] = [64, 32],
        dropout: float = 0.4,
        dense_units: int = 16,
        learning_rate: float = 0.0003,
        weight_decay: float = 0.0001,
    ):
        """
        Initialize LSTM Predictor
        
        Args:
            lookback: Number of timesteps to look back
            features: Number of features (will be auto-detected during training)
            lstm_units: List of LSTM layer units
            dropout: Dropout rate
            dense_units: Dense layer units before output
            learning_rate: Adam optimizer learning rate
            weight_decay: L2 regularization weight decay
            model_dir: Base directory for model artifacts
        """
        # Validate inputs
        if lookback <= 0:
            raise ValueError(f"lookback must be > 0, got {lookback}")
        if features <= 0:
            raise ValueError(f"features must be > 0, got {features}")
        if not lstm_units or any(u <= 0 for u in lstm_units):
            raise ValueError(f"lstm_units must be non-empty with positive values")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        # Model architecture
        self.lookback = lookback
        self.features = features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Model state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.n_features = None
        
        
        
        self.checkpoint_dir = "models/trend_model"
        self.log_dir = "models/trend_model"
        self.trained_dir = "models/trend_model"
        
        # Create directories
        for directory in [self.checkpoint_dir, self.log_dir, self.trained_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        #tf_version = tf.__version__
        #self.logger.info(f"LSTM Predictor initialized (TensorFlow {tf_version})")
        self.logger.info(f"Architecture: {lstm_units} LSTM units, {dropout} dropout")
        
    
    def build_model(self) -> models.Sequential:
        """
        Build LSTM architecture with configurable parameters
        
        Returns:
            Compiled Keras model
        """
        if self.n_features is None or self.n_features <= 0:
            raise ValueError(
                f"Invalid feature count: {self.n_features}. "
                "Call train() first or set n_features manually."
            )
        
        self.logger.info(f"Building model: lookback={self.lookback}, features={self.n_features}")
        
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.lookback, self.n_features)))
        
        # LSTM layers (configurable)
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                kernel_regularizer=regularizers.l2(self.weight_decay)
            ))
            model.add(layers.Dropout(self.dropout))
            
            # LayerNorm for all but last LSTM layer
            if return_sequences:
                model.add(layers.LayerNormalization())
        
        # Dense layers
        model.add(layers.Dense(
            self.dense_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.weight_decay)
        ))
        model.add(layers.Dropout(self.dropout * 0.67))  # Reduced dropout for dense
        
        # Output layer (3 classes: SELL, HOLD, BUY)
        model.add(layers.Dense(3, activation='softmax', dtype='float32'))
        
        # Compile
        optimizer = optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        
        # Log architecture
        param_count = model.count_params()
        self.logger.info(f"Model built: {param_count:,} parameters")
        
        # Save architecture summary
        summary_path = os.path.join(self.log_dir, 'model_architecture.txt')
        with open(summary_path, 'w',encoding='utf-8') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        self.logger.info(f"Architecture saved to {summary_path}")
        
        return model
    
    
    def create_labels(
        self,
        df: pd.DataFrame,
        threshold: float = 0.0005,
        horizon: int = 10
    ) -> pd.Series:
        """
        Create classification labels WITHOUT forward-looking bias
        
        Args:
            df: DataFrame with 'Close' column
            threshold: Price change threshold for BUY/SELL (0.0005 = 0.05%)
            horizon: Number of periods to look ahead
            
        Returns:
            Series with labels (0=SELL, 1=HOLD, 2=BUY, NaN=unknown)
        """
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
        
        # Calculate future returns
        future_return = df['Close'].shift(-horizon) / df['Close'] - 1
        
        # Handle inf values
        future_return = future_return.replace([np.inf, -np.inf], np.nan)
        
        # Create labels
        labels = np.where(
            future_return > threshold, 2,      # BUY
            np.where(future_return < -threshold, 0, 1)  # SELL or HOLD
        )
        
        labels = pd.Series(labels, index=df.index, dtype=float)
        
        # ✅ FIX: Mark last `horizon` rows as unknown (no forward-looking bias)
        labels.iloc[-horizon:] = np.nan
        
        # Log class distribution (excluding NaN)
        valid_labels = labels.dropna()
        if len(valid_labels) == 0:
            raise ValueError("No valid labels created")
        
        class_counts = valid_labels.value_counts().sort_index()
        total = len(valid_labels)
        
        self.logger.info(f"Labels created: threshold={threshold:.4f}, horizon={horizon}")
        self.logger.info(f"  SELL (0): {class_counts.get(0, 0):>6} ({class_counts.get(0, 0)/total*100:>5.1f}%)")
        self.logger.info(f"  HOLD (1): {class_counts.get(1, 0):>6} ({class_counts.get(1, 0)/total*100:>5.1f}%)")
        self.logger.info(f"  BUY  (2): {class_counts.get(2, 0):>6} ({class_counts.get(2, 0)/total*100:>5.1f}%)")
        self.logger.info(f"  Unknown:  {labels.isna().sum():>6}")
        
        # Warn if severe class imbalance
        class_percentages = class_counts / total * 100
        if (class_percentages > 70).any():
            max_class = class_percentages.idxmax()
            self.logger.warning(
                f"⚠ Severe class imbalance! Class {int(max_class)} has {class_percentages[max_class]:.1f}%. "
                f"Consider adjusting threshold or using class weights."
            )
        
        return labels
    
    
    def prepare_sequences(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None,
        fit_scaler: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare sequences for LSTM with strict validation
        
        Args:
            data: Feature DataFrame
            target: Target labels (optional, for training/validation)
            fit_scaler: Whether to fit scaler (True for training, False for val/pred)
            
        Returns:
            X array if target is None, else (X, y) tuple
        """
        # Validate input
        if data.shape[0] < self.lookback:
            raise ValueError(
                f"Data length ({data.shape[0]}) must be >= lookback ({self.lookback})"
            )
        
        # Handle NaN values
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"Found {nan_count} NaN values, applying fill strategy")
            # Forward fill, then backward fill, then zero
            data = data.ffill().bfill().fillna(0)
        
        # ═══════════════════════════════════════════════════════════════
        # SCALING (fit on train, transform on val/test)
        # ═══════════════════════════════════════════════════════════════
        if fit_scaler:
            # TRAINING MODE: Fit scaler
            self.feature_names = list(data.columns)
            self.n_features = len(self.feature_names)
            scaled_data = self.scaler.fit_transform(data)
            
            self.logger.info(
                f"✓ Scaler FITTED on {self.n_features} features"
            )
            self.logger.info(f"  Features: {self.feature_names[:3]}... (showing first 3)")
            
        else:
            # VALIDATION/PREDICTION MODE: Use existing scaler
            if not hasattr(self.scaler, 'mean_'):
                raise ValueError(
                    "Scaler not fitted. Call prepare_sequences with fit_scaler=True first."
                )
            
            if not hasattr(self, 'feature_names') or self.feature_names is None:
                raise ValueError("Feature names not found. Train model first.")
            
            # Strict feature validation
            if list(data.columns) != self.feature_names:
                raise ValueError(
                    f"Feature mismatch!\n"
                    f"  Expected ({len(self.feature_names)}): {self.feature_names[:3]}...\n"
                    f"  Got ({len(data.columns)}): {list(data.columns)[:3]}..."
                )
            
            if data.shape[1] != self.scaler.mean_.shape[0]:
                raise ValueError(
                    f"Feature count mismatch! Expected {self.scaler.mean_.shape[0]}, "
                    f"got {data.shape[1]}"
                )
            
            scaled_data = self.scaler.transform(data)
            
            mode = "VALIDATION" if target is not None else "PREDICTION"
            self.logger.info(f"✓ Scaler TRANSFORMED {mode} data")
        
        # ═══════════════════════════════════════════════════════════════
        # CREATE SEQUENCES
        # ═══════════════════════════════════════════════════════════════
        X = []
        y_list = [] if target is not None else None
        
        # ✅ FIX: Reset target index to avoid alignment issues
        if target is not None:
            target = target.reset_index(drop=True)
        
        for i in range(self.lookback, len(scaled_data)):
            # Input sequence: [i-lookback, i)
            X.append(scaled_data[i - self.lookback:i])
            
            # Target at position i
            if target is not None:
                y_list.append(target.iloc[i])
        
        X = np.array(X, dtype=np.float32)
        
        if X.shape[0] == 0:
            raise ValueError(
                f"No sequences created. Data length: {len(scaled_data)}, "
                f"Lookback: {self.lookback}"
            )
        
        self.logger.info(f"Created {X.shape[0]} sequences with shape {X.shape}")
        
        # ═══════════════════════════════════════════════════════════════
        # PROCESS TARGET
        # ═══════════════════════════════════════════════════════════════
        if y_list is not None:
            y = np.array(y_list, dtype=np.int32)
            
            # Validate classes
            unique_classes = np.unique(y[~np.isnan(y)])
            if not np.all(np.isin(unique_classes, [0, 1, 2])):
                raise ValueError(
                    f"Target values must be 0, 1, or 2. Found: {unique_classes}"
                )
            
            # One-hot encode
            y = keras.utils.to_categorical(y, num_classes=3)
            
            class_dist = np.sum(y, axis=0).astype(int)
            self.logger.info(
                f"Target shape: {y.shape}, "
                f"Distribution: SELL={class_dist[0]}, HOLD={class_dist[1]}, BUY={class_dist[2]}"
            )
            
            return X, y
        
        return X
    
    
    def add_circular_time_features(self,df):
        # 1. Get the total minutes passed in the day (0 - 1439)
        # This is better for M15 than just using hours
        minutes_in_day = df.index.hour * 60 + df.index.minute
        
        # 2. Max minutes in a day
        max_minutes = 24 * 60
        
        # 3. Apply Sine and Cosine transformations
        df['day_sin'] = np.sin(2 * np.pi * minutes_in_day / max_minutes)
        df['day_cos'] = np.cos(2 * np.pi * minutes_in_day / max_minutes)
        
        # 4. Optional: Add Day of the Week (0 = Monday, 6 = Sunday)
        day_of_week = df.index.dayofweek
        df['week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        return df
    
    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        label_threshold: float = 0.0005,
        label_horizon: int = 5,
        use_class_weights: bool = True,
        early_stopping_patience: int = 15,
        reduce_lr_patience: int = 8
    ) -> keras.callbacks.History:
        """
        Train the LSTM model with data leakage prevention
        
        Args:
            df: DataFrame with features
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            label_threshold: Price change threshold for labels
            label_horizon: Periods to look ahead for labels
            use_class_weights: Whether to use class balancing
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            
        Returns:
            Training history
        """
        # ═══════════════════════════════════════════════════════════════
        # VALIDATE INPUT
        # ═══════════════════════════════════════════════════════════════
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        
        min_required = self.lookback * 2 + label_horizon
        if len(df) < min_required:
            raise ValueError(
                f"Insufficient data: {len(df)} rows. "
                f"Need at least {min_required} rows "
                f"(lookback={self.lookback}, horizon={label_horizon})"
            )
        
        self.logger.info("=" * 70)
        self.logger.info(f"Starting training with {len(df)} rows")
        self.logger.info("=" * 70)
        
        # ═══════════════════════════════════════════════════════════════
        # CREATE LABELS
        # ═══════════════════════════════════════════════════════════════
        labels = self.create_labels(df, threshold=label_threshold, horizon=label_horizon)
        
        # Remove rows with unknown labels
        valid_idx = ~labels.isna()
        df_clean = df[valid_idx].copy()
        labels_clean = labels[valid_idx]
        
        self.logger.info(f"After label filtering: {len(df_clean)} rows remain")
        
        # ═══════════════════════════════════════════════════════════════
        # FEATURE SELECTION
        # ═══════════════════════════════════════════════════════════════
        # Exclude raw OHLCV from features (use only engineered features)
        feature_cols = [
            col for col in df_clean.columns
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Datetime', 'Date']
        ]
        
        if not feature_cols:
            raise ValueError("No feature columns found after excluding OHLCV")
        
        df_features = df_clean[feature_cols].copy()
        
        self.n_features = len(feature_cols)
        self.logger.info(f"Using {self.n_features} features")
        
        # ═══════════════════════════════════════════════════════════════
        # TRAIN/VAL SPLIT (time-series aware)
        # ═══════════════════════════════════════════════════════════════
        total_rows = len(df_features)
        split_idx = int(total_rows * (1 - validation_split))
        
        # Ensure minimum rows for both sets
        min_train = self.lookback + 1
        min_val = self.lookback + 1
        
        split_idx = max(min_train, min(split_idx, total_rows - min_val))
        
        train_data = df_features.iloc[:split_idx].copy()
        val_data = df_features.iloc[split_idx:].copy()
        train_target = labels_clean.iloc[:split_idx].copy()
        val_target = labels_clean.iloc[split_idx:].copy()
        
        self.logger.info(f"Split: Train={len(train_data)} rows, Val={len(val_data)} rows")
        
        # ═══════════════════════════════════════════════════════════════
        # PREPARE SEQUENCES (fit scaler on train only)
        # ═══════════════════════════════════════════════════════════════
        X_train, y_train = self.prepare_sequences(train_data, train_target, fit_scaler=True)
        X_val, y_val = self.prepare_sequences(val_data, val_target, fit_scaler=False)
        
        self.logger.info(f"Training sequences: {X_train.shape}")
        self.logger.info(f"Validation sequences: {X_val.shape}")
        
        # ═══════════════════════════════════════════════════════════════
        # COMPUTE CLASS WEIGHTS (handle imbalance)
        # ═══════════════════════════════════════════════════════════════
        class_weight_dict = None
        if use_class_weights:
            y_train_classes = y_train.argmax(axis=1)
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train_classes),
                y=y_train_classes
            )
            class_weight_dict = dict(enumerate(class_weights))
            
            self.logger.info(f"Class weights: {class_weight_dict}")
        
        # ═══════════════════════════════════════════════════════════════
        # BUILD MODEL
        # ═══════════════════════════════════════════════════════════════
        if self.model is None:
            self.build_model()
        
        # ═══════════════════════════════════════════════════════════════
        # CALLBACKS
        # ═══════════════════════════════════════════════════════════════
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                mode='min'
            ),
            callbacks.CSVLogger(
                os.path.join(self.log_dir, 'training_log.csv'),
                append=True
            )
        ]
        
        # ═══════════════════════════════════════════════════════════════
        # TRAIN MODEL
        # ═══════════════════════════════════════════════════════════════
        self.logger.info("Starting model training...")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            class_weight=class_weight_dict,
            shuffle=False,  # Important for time series
            verbose=1
        )
        
        # ═══════════════════════════════════════════════════════════════
        # TRAINING SUMMARY
        # ═══════════════════════════════════════════════════════════════
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_val_loss = np.min(history.history['val_loss'])
        best_val_acc = history.history['val_accuracy'][best_epoch - 1]
        final_train_acc = history.history['accuracy'][-1]
        
        # Check if early stopping was triggered
        actual_epochs = len(history.history['loss'])
        early_stopped = actual_epochs < epochs
        
        self.logger.info("=" * 70)
        self.logger.info("TRAINING COMPLETE!")
        self.logger.info("=" * 70)
        self.logger.info(f"Epochs Run:          {actual_epochs}/{epochs}" + 
                        (" (Early Stopped)" if early_stopped else ""))
        self.logger.info(f"Best Epoch:          {best_epoch}")
        self.logger.info(f"Best Val Loss:       {best_val_loss:.6f}")
        self.logger.info(f"Best Val Accuracy:   {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        self.logger.info(f"Final Train Accuracy:{final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        
        # Check for overfitting
        acc_gap = final_train_acc - best_val_acc
        if acc_gap > 0.1:
            self.logger.warning(
                f"⚠ Possible overfitting! Train-Val gap: {acc_gap:.4f}"
            )
        else:
            self.logger.info(f"✓ Train-Val gap: {acc_gap:.4f} (healthy)")
        
        self.logger.info("=" * 70)
        
        return history
    
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        epochs: int = 30,
        batch_size: int = 32,
        **kwargs
    ) -> Dict[str, float]:
        """
        Perform time series cross-validation
        
        Args:
            df: DataFrame with features
            n_splits: Number of CV splits
            epochs: Epochs per fold
            batch_size: Batch size
            **kwargs: Additional arguments for train()
            
        Returns:
            Dictionary with mean and std of metrics
        """
        self.logger.info(f"Starting {n_splits}-fold cross-validation")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        val_losses = []
        val_accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(df), 1):
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"FOLD {fold}/{n_splits}")
            self.logger.info(f"{'='*70}")
            
            # Reset model for each fold
            self.model = None
            self.scaler = StandardScaler()
            
            # Split data
            df_fold = df.copy()
            
            # Train on this fold
            history = self.train(
                df_fold,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=len(val_idx) / len(df),
                **kwargs
            )
            
            # Record best validation metrics
            best_val_loss = np.min(history.history['val_loss'])
            best_epoch = np.argmin(history.history['val_loss'])
            best_val_acc = history.history['val_accuracy'][best_epoch]
            
            val_losses.append(best_val_loss)
            val_accuracies.append(best_val_acc)
            
            self.logger.info(f"Fold {fold} - Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f}")
        
        # Calculate statistics
        results = {
            'mean_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses),
            'mean_val_accuracy': np.mean(val_accuracies),
            'std_val_accuracy': np.std(val_accuracies),
            'fold_losses': val_losses,
            'fold_accuracies': val_accuracies
        }
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info("CROSS-VALIDATION RESULTS")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Mean Val Loss:     {results['mean_val_loss']:.4f} ± {results['std_val_loss']:.4f}")
        self.logger.info(f"Mean Val Accuracy: {results['mean_val_accuracy']:.4f} ± {results['std_val_accuracy']:.4f}")
        self.logger.info(f"{'='*70}")
        
        return results
    
    
    def predict(
        self,
        df: pd.DataFrame,
        min_confidence: float = 0.0
    ) -> Tuple[np.ndarray, int, str, float]:
        """
        Make prediction on new data with confidence threshold
        
        Args:
            df: DataFrame with features
            min_confidence: Minimum confidence threshold (0-1)
            
        Returns:
            Tuple of (probabilities, predicted_class, signal, confidence)
        """
        # ═══════════════════════════════════════════════════════════════
        # VALIDATE MODEL STATE
        # ═══════════════════════════════════════════════════════════════
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler not fitted. Train or load model first.")
        
        if not hasattr(self, 'feature_names') or self.feature_names is None:
            raise ValueError("Feature names not found. Train or load model first.")
        
        # ═══════════════════════════════════════════════════════════════
        # VALIDATE INPUT
        # ═══════════════════════════════════════════════════════════════
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")
        
        if len(df) < self.lookback:
            raise ValueError(
                f"Insufficient data: {len(df)} rows. "
                f"Need at least {self.lookback} rows"
            )
        
        # ═══════════════════════════════════════════════════════════════
        # EXTRACT AND ALIGN FEATURES
        # ═══════════════════════════════════════════════════════════════
        feature_cols = [
            col for col in df.columns
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Datetime', 'Date']
        ]
        
        if not feature_cols:
            raise ValueError("No feature columns found")
        
        df_features = df[feature_cols].copy()
        
        # Handle NaN
        nan_count = df_features.isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"Found {nan_count} NaN values, filling...")
            df_features = df_features.ffill().bfill().fillna(0)
        
        # Feature alignment
        current_features = list(df_features.columns)
        expected_features = self.feature_names
        
        missing = set(expected_features) - set(current_features)
        extra = set(current_features) - set(expected_features)
        
        if missing:
            self.logger.warning(f"Missing {len(missing)} features, adding zeros")
            for feat in missing:
                df_features[feat] = 0.0
        
        if extra:
            self.logger.warning(f"Ignoring {len(extra)} extra features")
        
        # Reorder to match training
        try:
            df_features = df_features[self.feature_names]
        except KeyError as e:
            raise ValueError(f"Cannot align features: {e}")
        
        self.logger.info(f"Features aligned: {df_features.shape}")
        
        # ═══════════════════════════════════════════════════════════════
        # CREATE SEQUENCES
        # ═══════════════════════════════════════════════════════════════
        X = self.prepare_sequences(df_features, target=None, fit_scaler=False)
        
        if X is None or len(X) == 0:
            raise ValueError("No sequences created")
        
        # ═══════════════════════════════════════════════════════════════
        # PREDICT (on most recent sequence)
        # ═══════════════════════════════════════════════════════════════
        probabilities = self.model.predict(X[-1:], verbose=0)
        
        predicted_class = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(np.max(probabilities))
        
        prob_sell = float(probabilities[0][0])
        prob_hold = float(probabilities[0][1])
        prob_buy = float(probabilities[0][2])
        
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        signal = signal_map[predicted_class]
        
        # ═══════════════════════════════════════════════════════════════
        # APPLY CONFIDENCE THRESHOLD
        # ═══════════════════════════════════════════════════════════════
        if confidence < min_confidence:
            self.logger.warning(
                f"Low confidence: {confidence:.2%} < {min_confidence:.2%}. "
                f"Defaulting to HOLD."
            )
            signal = 'HOLD'
            predicted_class = 1
        
        # ═══════════════════════════════════════════════════════════════
        # LOG RESULTS
        # ═══════════════════════════════════════════════════════════════
        self.logger.info("=" * 70)
        self.logger.info("PREDICTION RESULTS")
        self.logger.info("=" * 70)
        self.logger.info(f"Signal:       {signal}")
        self.logger.info(f"Confidence:   {confidence:.2%}")
        self.logger.info(f"SELL Prob:    {prob_sell:.2%}")
        self.logger.info(f"HOLD Prob:    {prob_hold:.2%}")
        self.logger.info(f"BUY  Prob:    {prob_buy:.2%}")
        self.logger.info(f"Sequences:    {len(X)} (predicting on last)")
        self.logger.info("=" * 70)
        
        return probabilities[0], predicted_class, signal, confidence
    
    
    def save(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        version: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save model with versioning
        
        Args:
            model_path: Custom model path (optional)
            scaler_path: Custom scaler path (optional)
            version: Version string (auto-generated if None)
            
        Returns:
            Dictionary with saved paths
        """
        # ═══════════════════════════════════════════════════════════════
        # VALIDATE STATE
        # ═══════════════════════════════════════════════════════════════
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler not fitted. Train first.")
        
        if not hasattr(self, 'feature_names') or self.feature_names is None:
            raise ValueError("Feature names not found. Train first.")
        
        # ═══════════════════════════════════════════════════════════════
        # GENERATE VERSION
        # ═══════════════════════════════════════════════════════════════
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # ═══════════════════════════════════════════════════════════════
        # PATHS
        # ═══════════════════════════════════════════════════════════════
        if model_path is None:
            model_path = os.path.join(
                self.trained_dir,
                f'lstm_model_v{version}.keras'
            )
        
        if scaler_path is None:
            scaler_path = os.path.join(
                self.trained_dir,
                f'scaler_v{version}.pkl'
            )
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        # ═══════════════════════════════════════════════════════════════
        # SAVE MODEL
        # ═══════════════════════════════════════════════════════════════
        try:
            self.model.save(model_path)
            self.logger.info(f"✓ Model saved: {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
        
        # ═══════════════════════════════════════════════════════════════
        # SAVE SCALER + METADATA
        # ═══════════════════════════════════════════════════════════════
        metadata = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'lookback': self.lookback,
            'lstm_units': self.lstm_units,
            'dropout': self.dropout,
            'version': version,
            'saved_at': datetime.now().isoformat(),
            'model_params': self.model.count_params(),
            'architecture': {
                'lookback': self.lookback,
                'features': self.n_features,
                'lstm_units': self.lstm_units,
                'dropout': self.dropout,
                'dense_units': self.dense_units
            }
        }
        
        try:
            joblib.dump(metadata, scaler_path)
            self.logger.info(f"✓ Scaler + metadata saved: {scaler_path}")
        except Exception as e:
            self.logger.error(f"Failed to save scaler: {e}")
            raise
        
        # ═══════════════════════════════════════════════════════════════
        # SUMMARY
        # ═══════════════════════════════════════════════════════════════
        self.logger.info("=" * 70)
        self.logger.info("SAVE SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Version:      {version}")
        self.logger.info(f"Model:        {model_path}")
        self.logger.info(f"Scaler:       {scaler_path}")
        self.logger.info(f"Features:     {self.n_features}")
        self.logger.info(f"Lookback:     {self.lookback}")
        self.logger.info(f"Parameters:   {self.model.count_params():,}")
        self.logger.info("=" * 70)
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'version': version
        }
    
    
    def load(
        self,
        model_path: str,
        scaler_path: str
    ) -> None:
        """
        Load model with strict validation
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
        """
        # ═══════════════════════════════════════════════════════════════
        # VALIDATE PATHS
        # ═══════════════════════════════════════════════════════════════
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        try:
            # ═══════════════════════════════════════════════════════════════
            # LOAD MODEL
            # ═══════════════════════════════════════════════════════════════
            self.model = keras.models.load_model(model_path)
            self.logger.info(f"✓ Model loaded: {model_path}")
            
            # ═══════════════════════════════════════════════════════════════
            # LOAD SCALER + METADATA
            # ═══════════════════════════════════════════════════════════════
            metadata = joblib.load(scaler_path)
            
            if isinstance(metadata, dict):
                # New format
                self.scaler = metadata['scaler']
                self.feature_names = metadata.get('feature_names')
                self.n_features = metadata.get('n_features')
                saved_lookback = metadata.get('lookback')
                
                if self.feature_names is None:
                    raise ValueError("Metadata missing feature_names")
                
                # Validate lookback
                if saved_lookback and saved_lookback != self.lookback:
                    self.logger.warning(
                        f"Lookback mismatch! Saved: {saved_lookback}, "
                        f"Current: {self.lookback}. Updating to saved value."
                    )
                    self.lookback = saved_lookback
                
                self.logger.info(f"✓ Loaded metadata: {self.n_features} features")
                
            else:
                # Old format (backward compatibility)
                self.scaler = metadata
                self.feature_names = None
                self.logger.warning("⚠ Old format detected. Limited validation.")
            
            # ═══════════════════════════════════════════════════════════════
            # VALIDATE MODEL ARCHITECTURE
            # ═══════════════════════════════════════════════════════════════
            actual_input = self.model.input_shape
            
            if actual_input[1] != self.lookback:
                raise ValueError(
                    f"Model lookback mismatch! "
                    f"Expected {self.lookback}, got {actual_input[1]}"
                )
            
            if self.feature_names and actual_input[2] != len(self.feature_names):
                raise ValueError(
                    f"Model feature count mismatch! "
                    f"Expected {len(self.feature_names)}, got {actual_input[2]}"
                )
            
            # Validate scaler
            if not hasattr(self.scaler, 'mean_'):
                raise ValueError("Loaded scaler is not fitted")
            
            if self.feature_names:
                if self.scaler.mean_.shape[0] != len(self.feature_names):
                    raise ValueError(
                        f"Scaler feature count mismatch! "
                        f"Scaler: {self.scaler.mean_.shape[0]}, "
                        f"Features: {len(self.feature_names)}"
                    )
            
            # ═══════════════════════════════════════════════════════════════
            # SUMMARY
            # ═══════════════════════════════════════════════════════════════
            self.logger.info("=" * 70)
            self.logger.info("LOAD SUMMARY")
            self.logger.info("=" * 70)
            self.logger.info(f"Model:        {model_path}")
            self.logger.info(f"Scaler:       {scaler_path}")
            self.logger.info(f"Features:     {self.n_features}")
            self.logger.info(f"Lookback:     {self.lookback}")
            self.logger.info(f"Input Shape:  {self.model.input_shape}")
            self.logger.info(f"Output Shape: {self.model.output_shape}")
            self.logger.info(f"Parameters:   {self.model.count_params():,}")
            if self.feature_names:
                self.logger.info(f"Feature Names: {self.feature_names[:3]}... (first 3)")
            self.logger.info("=" * 70)
            
            self.logger.info("✓ Model loaded and validated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            # Reset state on failure
            self.model = None
            self.scaler = StandardScaler()
            self.feature_names = None
            raise


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

