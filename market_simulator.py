"""
Market Simulator for Testing Trading Bot
Generates realistic synthetic market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class MarketSimulator:
    """Realistic market data simulator"""
    
    def __init__(self):
        self.symbols = ["SBIN", "TCS", "INFY", "ITC", "RELIANCE"]
        self.current_prices = {
            "SBIN": 620.0,
            "TCS": 3245.0,
            "INFY": 1456.0,
            "ITC": 412.0,
            "RELIANCE": 2456.0
        }
        self.volatility = {
            "SBIN": 0.025,
            "TCS": 0.02,
            "INFY": 0.03,
            "ITC": 0.022,
            "RELIANCE": 0.028
        }
        self.trends = {symbol: random.choice([-1, 0, 1]) for symbol in self.symbols}
        self.last_update = datetime.now()
        self.market_open = True
        
    def generate_realistic_price_movement(self, symbol: str) -> float:
        """Generate realistic price movement"""
        base_volatility = self.volatility[symbol]
        trend = self.trends[symbol]
        
        # Random walk with trend bias
        random_change = np.random.normal(0, base_volatility)
        trend_bias = trend * 0.001  # Small trend bias
        
        # Add some market events (5% chance)
        if random.random() < 0.05:
            event_magnitude = random.choice([0.02, -0.02, 0.03, -0.03])  # 2-3% moves
            random_change += event_magnitude
        
        # Calculate new price
        current_price = self.current_prices[symbol]
        price_change = current_price * (random_change + trend_bias)
        new_price = current_price + price_change
        
        # Ensure price doesn't go negative or too extreme
        min_price = current_price * 0.95
        max_price = current_price * 1.05
        new_price = max(min_price, min(max_price, new_price))
        
        return round(new_price, 2)
    
    def update_prices(self):
        """Update all prices"""
        for symbol in self.symbols:
            self.current_prices[symbol] = self.generate_realistic_price_movement(symbol)
        
        # Occasionally change trends (10% chance)
        if random.random() < 0.1:
            symbol = random.choice(self.symbols)
            self.trends[symbol] = random.choice([-1, 0, 1])
        
        self.last_update = datetime.now()
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get current live price"""
        if not self.market_open:
            return None
            
        # Update prices every call to simulate real-time
        self.update_prices()
        
        price = self.current_prices.get(symbol)
        if price:
            logger.info(f"[SIM PRICE] {symbol}: Rs.{price:.2f}")
        return price
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Alias for get_live_price"""
        return self.get_live_price(symbol)
    
    def generate_historical_data(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """Generate realistic historical data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate date range (only weekdays)
        dates = pd.bdate_range(start=start_date, end=end_date)
        
        # Starting price
        base_price = self.current_prices[symbol] * random.uniform(0.8, 1.2)
        volatility = self.volatility[symbol]
        
        data = []
        current_price = base_price
        
        for date in dates:
            # Generate OHLCV data
            daily_volatility = volatility * random.uniform(0.5, 1.5)
            
            # Open price (gap from previous close)
            gap = np.random.normal(0, daily_volatility * 0.3)
            open_price = current_price * (1 + gap)
            
            # Intraday range
            daily_range = open_price * daily_volatility * random.uniform(0.8, 2.0)
            high = open_price + daily_range * random.uniform(0.3, 0.7)
            low = open_price - daily_range * random.uniform(0.3, 0.7)
            
            # Close price
            close_bias = np.random.normal(0, daily_volatility)
            close_price = open_price * (1 + close_bias)
            close_price = max(low, min(high, close_price))
            
            # Volume (realistic patterns)
            base_volume = random.randint(1000000, 5000000)
            volume_multiplier = 1 + abs(close_price - open_price) / open_price * 10
            volume = int(base_volume * volume_multiplier)
            
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
            
            current_price = close_price
        
        return pd.DataFrame(data)
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data for analysis"""
        return self.generate_historical_data(symbol, days)
    
    def get_market_status(self) -> Dict:
        """Get market status"""
        now = datetime.now()
        
        # Simulate market hours (always open for testing)
        return {
            'is_open': True,
            'status': 'Open (Simulation)',
            'next_event': 'Continuous trading',
            'current_time': now.strftime('%H:%M:%S')
        }
    
    def bulk_fetch_training_data(self, symbols: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
        """Generate bulk training data"""
        logger.info(f"🎭 Generating {days} days of synthetic training data for {len(symbols)} symbols...")
        
        data_dict = {}
        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] Generating data for {symbol}...")
            
            data = self.generate_historical_data(symbol, days)
            if not data.empty:
                data_dict[symbol] = data
                logger.info(f"✅ {symbol}: {len(data)} days generated")
        
        logger.info(f"Synthetic training data generation complete: {len(data_dict)} symbols")
        return data_dict
    
    def save_training_data(self, data_dict: Dict[str, pd.DataFrame], path: str = "training_data"):
        """Save synthetic training data"""
        try:
            os.makedirs(path, exist_ok=True)
            
            for symbol, data in data_dict.items():
                file_path = os.path.join(path, f"{symbol}_data.csv")
                data.to_csv(file_path, index=False)
            
            # Save metadata
            metadata = {
                'symbols': list(data_dict.keys()),
                'data_points': {symbol: len(data) for symbol, data in data_dict.items()},
                'last_updated': datetime.now().isoformat(),
                'total_symbols': len(data_dict),
                'data_type': 'synthetic'
            }
            
            with open(os.path.join(path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"🎭 Synthetic training data saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save synthetic data: {e}")
    
    def load_training_data(self, path: str = "training_data") -> Dict[str, pd.DataFrame]:
        """Load training data (generate if not exists)"""
        try:
            if not os.path.exists(path):
                logger.info("No cached data found, generating synthetic data...")
                data_dict = self.bulk_fetch_training_data(self.symbols, 365)
                self.save_training_data(data_dict, path)
                return data_dict
            
            # Load metadata
            metadata_path = os.path.join(path, 'metadata.json')
            if not os.path.exists(metadata_path):
                logger.info("No metadata found, generating fresh synthetic data...")
                data_dict = self.bulk_fetch_training_data(self.symbols, 365)
                self.save_training_data(data_dict, path)
                return data_dict
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load data files
            data_dict = {}
            for symbol in metadata['symbols']:
                file_path = os.path.join(path, f"{symbol}_data.csv")
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path)
                    data['date'] = pd.to_datetime(data['date'])
                    data_dict[symbol] = data
            
            logger.info(f"🎭 Loaded synthetic training data for {len(data_dict)} symbols")
            return data_dict
            
        except Exception as e:
            logger.error(f"Failed to load synthetic data: {e}")
            # Generate fresh data as fallback
            data_dict = self.bulk_fetch_training_data(self.symbols, 365)
            self.save_training_data(data_dict, path)
            return data_dict
    
    def create_market_events(self):
        """Create random market events for testing"""
        events = [
            "📈 Market Rally: Tech stocks surge",
            "📉 Profit Booking: Banking stocks decline", 
            "⚡ Breakout: High volume breakout detected",
            "🔄 Consolidation: Sideways movement",
            "📊 Earnings Impact: Mixed results"
        ]
        
        if random.random() < 0.1:  # 10% chance
            event = random.choice(events)
            logger.info(f"🎭 [MARKET EVENT] {event}")
            
            # Affect random stocks
            affected_symbols = random.sample(self.symbols, random.randint(1, 3))
            for symbol in affected_symbols:
                # Create temporary trend change
                self.trends[symbol] = random.choice([-1, 1])

# Create global simulator instance
market_simulator = MarketSimulator()

# Test function
def test_simulator():
    """Test the market simulator"""
    print("🎭 Testing Market Simulator...")
    print("=" * 50)
    
    # Test live prices
    print("📊 Live Prices:")
    for symbol in market_simulator.symbols:
        price = market_simulator.get_live_price(symbol)
        print(f"   {symbol}: Rs.{price:.2f}")
    
    print("\n📈 Historical Data Sample:")
    data = market_simulator.get_historical_data("TCS", 5)
    print(data.tail())
    
    print(f"\n🕐 Market Status: {market_simulator.get_market_status()}")
    
    print("\n✅ Simulator ready for bot testing!")

if __name__ == "__main__":
    test_simulator()