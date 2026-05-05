# data/connectors/mt5_connector.py

from logging import config

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import time
import yaml # type: ignore
import os

class MT5Connector:
    """
    Handles all MT5 connections, data retrieval, and order execution
    """
    
    def __init__(self, account: int, password: str, server: str, path: str = ""):
        self.account = account
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize(self.path):
                self.logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False
            
            if not mt5.login(self.account, password=self.password, server=self.server):
                self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            self.connected = True
            self.logger.info(f"Connected to MT5 Account: {self.account}")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Safely disconnect from MT5"""
        mt5.shutdown()
        self.connected = False
        self.logger.info("Disconnected from MT5")
    
    def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        bars: int ,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        
        Args:
            symbol: Trading pair (e.g., 'EURUSD', 'BTCUSD')
            timeframe: MT5 timeframe (e.g., 'M1', 'M5', 'H1', 'D1')
            bars: Number of bars to retrieve
            start_date: Optional start date
        """
        if not self.connected:
            raise ConnectionError("MT5 not connected")
        
        # Map timeframe strings to MT5 constants
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
        }
        
        tf = timeframe_map.get(timeframe)
        if tf is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Fetch data
        if start_date:
            rates = mt5.copy_rates_from(symbol, tf, start_date, bars)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        
        if rates is None or len(rates) == 0:
            self.logger.warning(f"No data retrieved for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def get_live_tick(self, symbol: str) -> Dict:
        """Get current tick data"""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {}
        
        return {
            'symbol': symbol,
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'time': datetime.fromtimestamp(tick.time)
        }
    
    def get_account_info(self) -> Dict:
        """Retrieve account information"""
        account_info = mt5.account_info()
        if account_info is None:
            return {}
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'margin_level': account_info.margin_level,
            'profit': account_info.profit,
            'currency': account_info.currency
        }
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        return [
            {
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'time': datetime.fromtimestamp(pos.time)
            }
            for pos in positions
        ]
    
    def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "Algo Trade",
        magic: int = 234000
    ) -> Dict:
        """
        Place market order with SL/TP
        
        Returns:
            dict with 'success', 'ticket', 'message'
        """
        if not self.connected:
            return {'success': False, 'message': 'MT5 not connected'}
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {'success': False, 'message': f'Symbol {symbol} not found'}
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return {'success': False, 'message': f'Failed to select {symbol}'}
        
        # Round volume
        volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if order_type == 'BUY' else tick.bid
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            return {'success': False, 'message': 'Order send failed'}
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                'success': False,
                'message': f'Order failed: {result.comment}',
                'retcode': result.retcode
            }
        
        return {
            'success': True,
            'ticket': result.order,
            'volume': result.volume,
            'price': result.price,
            'message': 'Order executed successfully'
        }
    
    def close_position(self, ticket: int) -> Dict:
        """Close specific position by ticket"""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {'success': False, 'message': 'Position not found'}
        
        position = positions[0]
        
        # Opposite order type
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        price = mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close by algo",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {'success': True, 'message': f'Position {ticket} closed'}
        else:
            return {'success': False, 'message': result.comment}
    
    def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> Dict:
        """Modify SL/TP of existing position"""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {'success': False, 'message': 'Position not found'}
        
        position = positions[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": sl if sl is not None else position.sl,
            "tp": tp if tp is not None else position.tp,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {'success': True, 'message': 'Position modified'}
        else:
            return {'success': False, 'message': result.comment}
        
        
        
symbol = 'EURUSDm'
timeframe = 'M1'
bars = 100  

if __name__ == "__main__":
    with open('settings/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    connector =MT5Connector(config['mt5']['account'], config['mt5']['password'], config['mt5']['server'], config['mt5']['path'])
    if connector.connect():
        data = connector.get_historical_data(symbol, timeframe, bars)
        print(data)
        print(connector.get_live_tick(symbol))
        positions = connector.get_positions()
        
        ticket = positions[0]['ticket'] if positions else None
        print(connector.close_position(ticket))
        connector.disconnect()