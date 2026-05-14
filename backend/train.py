


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

from models.trend_model.trend_model import LSTMPredictor  

#from models.momentum_model.momentum import XGBoostMomentumDetector
from models.momentum_model.main import XGBoostMomentumDetector
from models.regime_model.main import XGBoostRegimeDetector



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
        
    
        
        
        self.trend_model = LSTMPredictor()  
        self.momentum_model= XGBoostMomentumDetector()
        self.regime_model = XGBoostRegimeDetector()
        
        self.path= f'historicalData/EURUSD_15M_2010_2024.csv'
        
        
        self.M15 = pd.read_csv ('historicalData/EURUSD_15M_2015_2024.csv',sep=',')
        self.H1 = pd.read_csv ('historicalData/EURUSD_1H_2015_2024.csv',sep=',')
       
        
        
        
    def initialize(self):
       
        
        #df = utils.prepare_data1(self.M15,self.H1)
        
        df_1 = self.M15.copy()
        df_1.columns = df_1.columns.str.strip()
        
        df_1['Time'] = pd.to_datetime(df_1['Time'])
        df_1 = df_1.set_index('Time')
        
        df_1 = df_1.sort_index()
        
       
        
        self._train_model_for_pair(df_1)
        
        
   
    def _train_model_for_pair(self, df):
        """Train ML model for specific pair"""
        self.logger.info(f"  📊 Training model for {self.symbol}...")
        
        try:
           
            
           
           

            # 2. Combine Date and Time into one column
            #df['time_combined'] = pd.to_datetime(df['date'] + ' ' + df['time'])

            # 3. Set as index and drop the old separate columns
            #df.set_index('time_combined', inplace=True)
            
            
            
            #df.drop(['date', 'time'], axis=1, inplace=True)

            # 4. Now perform your renames
           
            
            
            
            
            
           # df=self.momentum_model.add_circular_time_features(df)
            
            # Add technical features
            self.logger.info(f"  🔧 Adding technical indicators...")
           # df = self.technical_indicators.create_trend_features(df)
           
            if df is None or df.empty:
                raise ValueError(f"Feature engineering failed for {self.symbol}")
        
            self.logger.info(f"  🔧 Features added: {len(df.columns)} columns")
            
            # Prepare ML features
            self.logger.info(f"  🧮 Preparing ML features...")
            
            #ml_features = self.technical_indicators.create_momentum_ml_features(df)
            
            
        
           
            
           
            
            
            # Train model
            #df_train = df_train[[ col for col in kepp_columns if col in df_train.columns]]
            
            #print(df_train)
            print(df.columns.tolist())
            start_time = pd.Timestamp.now()
            self.logger.info(f"  🧠 Training LSTM model (this may take a few minutes)...")
            #history= self.momentum_model.train(df)
            history1 = self.regime_model.train(df)
            
            
            # Save model
            model_path = f"models/regime_model/trained/{self.symbol}_model.json"
            meta_path = f"models/regime_model/trained/{self.symbol}_metadata.pkl"
            
            
            self.regime_model.save(model_path, meta_path)
            #self.momentum_model.load()
            #self.momentum_model.predict(df)
            
            #path= f'models/momentum_model/charts/EURUSD_M15_2015_2024.png'
            
           # utils.plot_training_history(history, save_path=path)
            
            elapsed_time = (pd.Timestamp.now() - start_time).seconds
            minutes      = elapsed_time // 60
            seconds      = elapsed_time % 60
            
            self.logger.info(f"  ✅ Training Complete for {self.symbol}!")
            self.logger.info(f"  ⏱️  Time Taken:    {minutes}m {seconds}s")
            self.logger.info(f"  💾 Model Path:    {model_path}")
            self.logger.info(f"  💾 Scaler Path:   {meta_path}")
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
    
    
    try:
        system = MultiPairTradingSystem()
        system.run()
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
