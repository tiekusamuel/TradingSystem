


# main.py - COMPLETE FIXED VERSION

import time
import pandas as pd
import yaml # type: ignore
import logging
from datetime import datetime, time as dt_time
from typing import Dict
import os
import sys

import utils

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger import setup_logger
from features.features import TechnicalFeatures
from models.trend_model.trend_model import LSTMPredictor  



class MultiPairTradingSystem:
    """
    Enhanced trading system with multi-pair support
    """
    
    def __init__(self, config_path='settings/config.yaml'):
        # Load configurations
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
       
        
        
        # Setup logging
        self.logger = setup_logger(self.config['logging'])
        
        self.symbol= "EURUSDm"
        
    
        
        self.technical_indicators = TechnicalFeatures()
        self.trend_model = LSTMPredictor()  
        
       
       
        
    def initialize(self):
       
        self.path =f'EURUSDm_M15_2000_2024.csv'
        self._train_model_for_pair(self.path)
        
        
   
    def _train_model_for_pair(self, csv_file_path=None):
        """Train ML model for specific pair"""
        self.logger.info(f"  📊 Training model for {self.symbol}...")
        
        try:
           
            
        # Load data from CSV
            if not os.path.exists(csv_file_path):
             raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
            df = pd.read_csv(csv_file_path, sep='\t')
            if df.empty:
                raise ValueError(f"No historical data retrieved for {self.symbol}")
            
            

            # 2. Combine Date and Time into one column
            df['time_combined'] = pd.to_datetime(df['date'] + ' ' + df['time'])

            # 3. Set as index and drop the old separate columns
            df.set_index('time_combined', inplace=True)
            
            
            
            df.drop(['date', 'time'], axis=1, inplace=True)

            # 4. Now perform your renames
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            df=self.trend_model.add_circular_time_features(df)
            
            # Add technical features
            self.logger.info(f"  🔧 Adding technical indicators...")
            df = self.technical_indicators.create_trend_features(df)
            if df is None or df.empty:
                raise ValueError(f"Feature engineering failed for {self.symbol}")
        
            self.logger.info(f"  🔧 Features added: {len(df.columns)} columns")
            
            # Prepare ML features
            self.logger.info(f"  🧮 Preparing ML features...")
            
            ml_features = self.technical_indicators.create_trend_ml_features(df)
            
            if ml_features is None or ml_features.empty:
                raise ValueError(f"ML feature creation failed for {self.symbol}")
        
            self.logger.info(f"  🧮 ML features created: {len(ml_features.columns)} features")
            
            # Combine features
            df_train = df.copy()
            for col in ml_features.columns:
                if col not in df_train.columns:
                    df_train[col] = ml_features[col]
            
            # Train model
            start_time = pd.Timestamp.now()
            self.logger.info(f"  🧠 Training LSTM model (this may take a few minutes)...")
            history= self.trend_model.train(df_train, epochs=30, batch_size=32, validation_split=0.2)
            
            #log training results
            if history is not None:
                best_val_loss = min(history.history['val_loss'])
                best_val_acc  = max(history.history['val_accuracy'])
                total_epochs  = len(history.history['val_loss'])
                
                self.logger.info(f"  📉 Best Val Loss:     {best_val_loss:.6f}")
                self.logger.info(f"  🎯 Best Val Accuracy: {best_val_acc*100:.2f}%")
                self.logger.info(f"  🔄 Epochs Completed:  {total_epochs}/50")
            
            # Save model
            model_path = f'backend/trend_model/EURUSDm_M15_lstm.keras'
            scaler_path = f'backend/trend_model/EURUSDm_M15_scaler.pkl'
            self.trend_model.save(model_path, scaler_path)
            
            path= f'trend_model/EURUSDm_M15.png'
            
            utils.plot_training_history(history, save_path=path)
            
            elapsed_time = (pd.Timestamp.now() - start_time).seconds
            minutes      = elapsed_time // 60
            seconds      = elapsed_time % 60
            
            self.logger.info(f"  ✅ Training Complete for {self.symbol}!")
            self.logger.info(f"  ⏱️  Time Taken:    {minutes}m {seconds}s")
            self.logger.info(f"  💾 Model Path:    {model_path}")
            self.logger.info(f"  💾 Scaler Path:   {scaler_path}")
            self.logger.info(f"{'='*60}")
            
            self.logger.info(f"  ✅ Model trained and saved for {self.symbol}")
            
        except Exception as e:
            self.logger.error(f"  ❌ Failed to train model for {self.symbol}: {e}", exc_info=True)
            raise
    
    
    def run(self):
        """Main execution loop"""
        self.initialize()
        
       
        
        
    



if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║   Multi-Pair Algorithmic Trading System v1.0         ║
    ║   Python 3.12 | MT5 Integration                      ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    #print(f"Model probs:Down:{probabilities[0]:.2%} | Neutral:{probabilities[1]:.2%} | Up:{probabilities[2]:.2%}")
    try:
        system = MultiPairTradingSystem()
        system.run()
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
