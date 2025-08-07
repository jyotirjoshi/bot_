"""
Final Working Web Scraper for Real-time Stock Data
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import yfinance as yf
import json
import os

logger = logging.getLogger(__name__)

class FinalWorkingScraper:
    """Final working scraper with multiple data sources"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get live price using MoneyControl web scraper"""
        cache_key = f"{symbol}_live"
        
        # Check cache - 2 minutes for live prices
        if cache_key in self.cache:
            price, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 120:  # 2 minutes
                return price
        
        # Try MoneyControl web scraper first
        price = self._get_moneycontrol_price(symbol)
        if price:
            self.cache[cache_key] = (price, datetime.now())
            return price
        
        # Fallback to Yahoo Finance with rate limiting
        time.sleep(3)
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="2d", interval="1d")
            if not data.empty:
                price = float(data['Close'].iloc[-1])
                logger.info(f"[YAHOO] {symbol}: Rs.{price:.2f}")
                self.cache[cache_key] = (price, datetime.now())
                return price
        except Exception as e:
            logger.debug(f"Yahoo failed for {symbol}: {e}")
        
        logger.warning(f"Could not get live price for {symbol}")
        return None
    
    def _get_moneycontrol_price(self, symbol: str) -> Optional[float]:
        """Scrape price from MoneyControl"""
        try:
            # MoneyControl URLs for major stocks
            mc_urls = {
                'SBIN': 'https://www.moneycontrol.com/india/stockpricequote/banks-public-sector/statebankofIndia/SBI',
                'TCS': 'https://www.moneycontrol.com/india/stockpricequote/computers-software/tataconsultancyservices/TCS',
                'INFY': 'https://www.moneycontrol.com/india/stockpricequote/computers-software/infosys/IT',
                'ITC': 'https://www.moneycontrol.com/india/stockpricequote/diversified/itc/ITC',
                'RELIANCE': 'https://www.moneycontrol.com/india/stockpricequote/refineries/relianceindustries/RI'
            }
            
            if symbol not in mc_urls:
                return None
            
            # Rate limiting
            time.sleep(2)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            
            response = self.session.get(mc_urls[symbol], headers=headers, timeout=10)
            
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for price elements
                selectors = [
                    'div.inprice1 span',
                    '.inprice1',
                    'span.span_price_wrap',
                    '.price_overview_today_price'
                ]
                
                for selector in selectors:
                    elem = soup.select_one(selector)
                    if elem:
                        text = elem.get_text().strip().replace(',', '').replace('₹', '')
                        match = re.search(r'[\d.]+', text)
                        if match:
                            price = float(match.group())
                            if 10 <= price <= 50000:  # Validate price range
                                logger.info(f"[MONEYCONTROL] {symbol}: Rs.{price:.2f}")
                                return price
                                
        except Exception as e:
            logger.debug(f"MoneyControl failed for {symbol}: {e}")
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get REAL historical data with aggressive caching"""
        cache_key = f"{symbol}_hist_{days}"
        
        # Check cache - extend to 24 hours to reduce API calls
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 86400:  # 24 hours
                return data
        
        # Try to use cached training data first
        try:
            training_data = self.load_training_data()
            if symbol in training_data:
                data = training_data[symbol]
                if not data.empty and len(data) >= days:
                    # Use last N days from training data
                    recent_data = data.tail(days).copy()
                    self.cache[cache_key] = (recent_data, datetime.now())
                    logger.info(f"[CACHED-HIST] {symbol}: {len(recent_data)} days from cache")
                    return recent_data
        except Exception as e:
            logger.debug(f"Cached training data failed for {symbol}: {e}")
        
        # Rate limiting - wait 10 seconds before API call
        time.sleep(10)
        
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period=f"{days}d", interval="1d")
            
            if not data.empty:
                data.reset_index(inplace=True)
                data.columns = [col.lower() for col in data.columns]
                
                # Add required columns for ML compatibility
                if 'dividends' not in data.columns:
                    data['dividends'] = 0.0
                if 'stock splits' not in data.columns:
                    data['stock splits'] = 1.0
                
                # Cache the data for 24 hours
                self.cache[cache_key] = (data, datetime.now())
                logger.info(f"[REAL-HIST] {symbol}: {len(data)} days of REAL data")
                return data
            else:
                logger.warning(f"No real historical data available for {symbol}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Real historical data failed for {symbol}: {e}")
            # Return empty DataFrame if all methods fail
            return pd.DataFrame()
    
    def get_market_status(self) -> Dict:
        """Get market status"""
        now = datetime.now()
        
        # Weekend check
        if now.weekday() >= 5:
            return {'is_open': False, 'status': 'Weekend', 'next_open': 'Monday 9:15 AM'}
        
        # Market hours: 9:15 AM to 3:30 PM
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_open = market_start <= now <= market_end
        
        if is_open:
            status = 'Open'
            next_event = f"Closes at 3:30 PM"
        elif now < market_start:
            status = 'Pre-market'
            next_event = f"Opens at 9:15 AM"
        else:
            status = 'Closed'
            next_event = f"Opens tomorrow at 9:15 AM"
        
        return {
            'is_open': is_open,
            'status': status,
            'next_event': next_event,
            'current_time': now.strftime('%H:%M:%S')
        }
    
    def bulk_fetch_training_data(self, symbols: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
        """Bulk fetch training data"""
        logger.info(f"Fetching {days} days of training data for {len(symbols)} symbols...")
        
        data_dict = {}
        successful = 0
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"[{i+1}/{len(symbols)}] Fetching data for {symbol}...")
                
                # Get data with retry logic
                data = None
                for attempt in range(3):
                    try:
                        ticker = yf.Ticker(f"{symbol}.NS")
                        data = ticker.history(period=f"{days}d", interval="1d")
                        
                        if not data.empty:
                            break
                        else:
                            logger.warning(f"Empty data for {symbol}, attempt {attempt + 1}")
                            time.sleep(2)
                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                        time.sleep(2)
                
                if data is not None and not data.empty:
                    # Clean and format data
                    data.reset_index(inplace=True)
                    data.columns = [col.lower() for col in data.columns]
                    
                    # Validate data quality
                    if len(data) >= 100:
                        data_dict[symbol] = data
                        successful += 1
                        logger.info(f"✅ {symbol}: {len(data)} days collected")
                    else:
                        logger.warning(f"⚠️ {symbol}: Insufficient data ({len(data)} days)")
                else:
                    logger.error(f"❌ {symbol}: Failed to fetch data")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Data collection failed - {e}")
        
        logger.info(f"Training data collection complete: {successful}/{len(symbols)} successful")
        return data_dict
    
    def save_training_data(self, data_dict: Dict[str, pd.DataFrame], path: str = "training_data"):
        """Save training data to disk"""
        try:
            os.makedirs(path, exist_ok=True)
            
            for symbol, data in data_dict.items():
                file_path = os.path.join(path, f"{symbol}_data.csv")
                data.to_csv(file_path, index=False)
                logger.debug(f"Saved {symbol} data to {file_path}")
            
            # Save metadata
            metadata = {
                'symbols': list(data_dict.keys()),
                'data_points': {symbol: len(data) for symbol, data in data_dict.items()},
                'last_updated': datetime.now().isoformat(),
                'total_symbols': len(data_dict)
            }
            
            with open(os.path.join(path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Training data saved to {path} directory")
            
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
    
    def load_training_data(self, path: str = "training_data") -> Dict[str, pd.DataFrame]:
        """Load training data from disk"""
        try:
            if not os.path.exists(path):
                return {}
            
            # Load metadata
            metadata_path = os.path.join(path, 'metadata.json')
            if not os.path.exists(metadata_path):
                return {}
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if data is recent
            last_updated = datetime.fromisoformat(metadata['last_updated'])
            if (datetime.now() - last_updated).days > 7:
                logger.info("Cached training data is older than 7 days")
                return {}
            
            # Load data files
            data_dict = {}
            for symbol in metadata['symbols']:
                file_path = os.path.join(path, f"{symbol}_data.csv")
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path)
                    data['date'] = pd.to_datetime(data['date'])
                    data_dict[symbol] = data
            
            logger.info(f"Loaded cached training data for {len(data_dict)} symbols")
            return data_dict
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return {}

# Create global instance
final_working_scraper = FinalWorkingScraper()