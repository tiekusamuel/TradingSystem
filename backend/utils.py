import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import logging
import os
from typing import Tuple, Optional, Dict, List, Union
from datetime import datetime

import matplotlib.pyplot as plt

from models.trend_model.trend_model import LSTMPredictor

from models.momentum_model.main import MomentumFeatureEngine
from models.regime_model.main import XGBoostRegimeDetector




def plot_training_history(history: keras.callbacks.History, save_path: Optional[str] = None):
    """
    Plot training history
    
    Args:
        history: Keras training history
        save_path: Path to save plot (optional)
    
    
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    
        
        # Ensure directories exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    
    
    

def evaluate_model(
    model: LSTMPredictor,
    df: pd.DataFrame,
    label_threshold: float = 0.0005,
    label_horizon: int = 5
) -> Dict[str, float]:
    """
    Evaluate model on test data
    
    Args:
        model: Trained LSTMPredictor
        df: Test DataFrame
        label_threshold: Label threshold
        label_horizon: Label horizon
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    # Create labels
    labels = model.create_labels(df, threshold=label_threshold, horizon=label_horizon)
    valid_idx = ~labels.isna()
    df_clean = df[valid_idx].copy()
    labels_clean = labels[valid_idx]
    
    # Get features
    feature_cols = [
        col for col in df_clean.columns
        if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Datetime', 'Date']
    ]
    df_features = df_clean[feature_cols].copy()
    
    # Prepare sequences
    X_test, y_test = model.prepare_sequences(df_features, labels_clean, fit_scaler=False)
    
    # Predict
    y_pred = model.model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=['SELL', 'HOLD', 'BUY']
    ))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_classes, y_pred_classes))
    print("=" * 70)
    
    return {
        'accuracy': accuracy,
        'y_true': y_true_classes,
        'y_pred': y_pred_classes,
        'y_pred_proba': y_pred
    }
    
    
    

     

def prepare_data1(df_1,df_2):
    
    df_1=df_1.copy()
    
    df_2=df_2.copy()
   
    df_1.columns = df_1.columns.str.strip()
    df_2.columns = df_2.columns.str.strip()
    
   
    
    
    
   
    df_2 = df_2.shift(1)
    
    keep_col =['H1_Directional_Bias','H1_Efficiency_Ratio_20','H1_Momentum_Strength','H1_ROC_20','H1_ROC_50','H1_Momentum_Acceleration','H1_EMA20_Slope']
    
    df_2=df_2[[col for col in keep_col if col in df_2.columns]]
    
   
    
    
   
    df_1['Time'] = pd.to_datetime(df_1['Time'])
    df_1 = df_1.set_index('Time')
    
    df_1 = df_1.sort_index()
    
    
    
    
    #df.columns = df.columns.str.strip()
    
   

# Now check the output
   

    
    
    
    return df_1
     


def analyze_percentile_impact(
        self,
        df: pd.DataFrame,
        percentile_range: List[float] = [0.05, 0.10, 0.15, 0.20, 0.25],
        **label_kwargs
    ) -> pd.DataFrame:
        """
        Analyze how different percentile thresholds affect class balance.
        
        Use this to choose optimal top_percentile before training.
        
        Args:
            df: OHLCV DataFrame
            percentile_range: List of percentiles to test
            **label_kwargs: Passed to create_percentile_expansion_labels
            
        Returns:
            DataFrame with percentile analysis results
            
        Example:
            analysis = model.analyze_percentile_impact(
                df,
                percentile_range=[0.10, 0.15, 0.20, 0.25]
            )
            print(analysis)
        """
        self.logger.info("=" * 70)
        self.logger.info("PERCENTILE IMPACT ANALYSIS")
        self.logger.info("=" * 70)
        
        results = []
        
        for pct in percentile_range:
            self.logger.info(f"\nTesting top_percentile = {pct:.2%}...")
            
            # Remove top_percentile from kwargs if present
            kwargs = {k: v for k, v in label_kwargs.items() if k != 'top_percentile'}
            
            # Create labels
            labels = XGBoostRegimeDetector.create_percentile_expansion_labels(
                df,
                top_percentile=pct,
                **kwargs
            )
            
            # Stats
            valid = labels.dropna()
            if len(valid) > 0:
                n_favorable = (valid == 1).sum()
                pct_favorable = n_favorable / len(valid) * 100
                
                # Run label diagnostics
                diagnostics = XGBoostRegimeDetector.diagnose_label_quality(self, labels, plot=False)
                
                results.append({
                    'percentile': pct,
                    'n_favorable': n_favorable,
                    'pct_favorable': pct_favorable,
                    'favorable_streak_mean': diagnostics['favorable_streak_mean'],
                    'autocorr_lag1': diagnostics['autocorr_lag1'],
                    'clustering_risk': diagnostics['clustering_risk'],
                    'recommendation': diagnostics['recommendation'],
                })
            else:
                results.append({
                    'percentile': pct,
                    'n_favorable': 0,
                    'pct_favorable': 0.0,
                    'favorable_streak_mean': np.nan,
                    'autocorr_lag1': np.nan,
                    'clustering_risk': 'UNKNOWN',
                    'recommendation': 'NO_DATA',
                })
        
        results_df = pd.DataFrame(results)
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("SUMMARY")
        self.logger.info("=" * 70)
        print(results_df.to_string(index=False))
        
        # Save
        output_path = os.path.join('backend', 'percentile_analysis.csv')
        results_df.to_csv(output_path, index=False)
        self.logger.info(f"\n✓ Analysis saved: {output_path}")
        
        # Recommendation
        self.logger.info("\n" + "=" * 70)
        self.logger.info("RECOMMENDATION")
        self.logger.info("=" * 70)
        
        # Filter acceptable options
        acceptable = results_df[
            (results_df['pct_favorable'] >= 10) &
            (results_df['pct_favorable'] <= 25) &
            (results_df['clustering_risk'].isin(['LOW', 'MODERATE']))
        ]
        
        if len(acceptable) > 0:
            # Prefer lowest autocorrelation
            best = acceptable.loc[acceptable['autocorr_lag1'].idxmin()]
            
            self.logger.info(f"  ✓ Recommended percentile: {best['percentile']:.2%}")
            self.logger.info(f"    Favorable rate: {best['pct_favorable']:.1f}%")
            self.logger.info(f"    Autocorrelation: {best['autocorr_lag1']:.3f}")
            self.logger.info(f"    Clustering risk: {best['clustering_risk']}")
        else:
            self.logger.warning("  ⚠ No percentile meets criteria. Try adjusting range.")
        
        self.logger.info("=" * 70)
        
        return results_df

    
    
    