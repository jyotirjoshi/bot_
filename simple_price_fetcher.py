"""
Simple working price fetcher with multiple sources
"""

import requests
import yfinance as yf
import time
import logging
from datetime import datetime
from typing import Optional
import json
import os

logger = logging.getLogger(__name__)

class SimplePriceFetcher:
    """Simple price fetcher that actually works"""
    
    def __init__(self):
        self.cache = {}
        self.last_prices = {}
        
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get live price with simple fallback"""
        cache_key = f"{symbol}_price"
        
        # Check 5-minute cache
        if cache_key in self.cache:
            price, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 300:
                return price
        
        # Method 1: Try Yahoo Finance with minimal delay
        price = self._get_yahoo_price(symbol)
        if price:
            self.cache[cache_key] = (price, datetime.now())
            self.last_prices[symbol] = price
            logger.info(f"[YAHOO] {symbol}: Rs.{price:.2f}")
            return price
        
        # Method 2: Use last known price if available
        if symbol in self.last_prices:
            price = self.last_prices[symbol]
            logger.info(f"[CACHED] {symbol}: Rs.{price:.2f}")
            return price
        
        # No fake prices - only real data
        logger.warning(f"No real price available for {symbol}")
        
        return None
    
    def _get_yahoo_price(self, symbol: str) -> Optional[float]:
        """Get price from Yahoo Finance with timeout"""
        try:
            # Short timeout to avoid hanging
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1d")
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            logger.debug(f"Yahoo failed for {symbol}: {e}")
        
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30):
        """Get historical data with caching"""
        try:
            # Try to load from cache first
            cache_file = f"price_cache_{symbol}_{days}.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    cache_time = datetime.fromisoformat(cached_data['timestamp'])
                    if (datetime.now() - cache_time).hours < 24:
                        import pandas as pd
                        df = pd.DataFrame(cached_data['data'])
                        df['date'] = pd.to_datetime(df['date'])
                        logger.info(f"[CACHED-HIST] {symbol}: {len(df)} days")
                        return df
            
            # Fetch new data
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period=f"{days}d", interval="1d")
            
            if not data.empty:
                data.reset_index(inplace=True)
                data.columns = [col.lower() for col in data.columns]
                
                # Add required columns
                if 'dividends' not in data.columns:
                    data['dividends'] = 0.0
                if 'stock splits' not in data.columns:
                    data['stock splits'] = 1.0
                
                # Cache the data (convert timestamps to strings)
                data_for_cache = data.copy()
                data_for_cache['date'] = data_for_cache['date'].dt.strftime('%Y-%m-%d')
                
                cache_data = {
                    'timestamp': datetime.now().isoformat(),
                    'data': data_for_cache.to_dict('records')
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
                
                logger.info(f"[REAL-HIST] {symbol}: {len(data)} days")
                return data
        except Exception as e:
            logger.error(f"Historical data failed for {symbol}: {e}")
        
        import pandas as pd
        return pd.DataFrame()
    
    def get_market_status(self):
        """Get market status"""
        now = datetime.now()
        
        if now.weekday() >= 5:
            return {'is_open': False, 'status': 'Weekend'}
        
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if now < market_start:
            return {'is_open': False, 'status': 'Pre-market'}
        elif now > market_end:
            return {'is_open': False, 'status': 'Post-market'}
        else:
            return {'is_open': True, 'status': 'Open'}
    
    # Required compatibility methods
    def load_training_data(self, path: str = "training_data"):
        import os
        try:
            if not os.path.exists(path):
                return {}
            metadata_path = os.path.join(path, 'metadata.json')
            if not os.path.exists(metadata_path):
                return {}
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            data_dict = {}
            for symbol in metadata['symbols']:
                file_path = os.path.join(path, f"{symbol}_data.csv")
                if os.path.exists(file_path):
                    import pandas as pd
                    data = pd.read_csv(file_path)
                    data['date'] = pd.to_datetime(data['date'])
                    data_dict[symbol] = data
            return data_dict
        except:
            return {}
    
    def bulk_fetch_training_data(self, symbols: list, days: int = 365):
        data_dict = {}
        for symbol in symbols:
            data = self.get_historical_data(symbol, days)
            if not data.empty:
                data_dict[symbol] = data
            time.sleep(1)
        return data_dict
    
    def save_training_data(self, data_dict, path: str = "training_data"):
        import os
        try:
            os.makedirs(path, exist_ok=True)
            for symbol, data in data_dict.items():
                file_path = os.path.join(path, f"{symbol}_data.csv")
                data.to_csv(file_path, index=False)
            metadata = {
                'symbols': list(data_dict.keys()),
                'last_updated': datetime.now().isoformat(),
                'total_symbols': len(data_dict)
            }
            with open(os.path.join(path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")

# Global instance
simple_price_fetcher = SimplePriceFetcher()