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