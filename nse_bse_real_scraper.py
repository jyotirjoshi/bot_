"""
Real NSE/BSE Web Scraper - Gets actual live prices
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
import json
import os
import re
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class NSEBSERealScraper:
    """Real web scraper for NSE/BSE live prices"""
    
    def __init__(self):
        self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive'
        })
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get real live price from multiple sources"""
        cache_key = f"{symbol}_live"
        
        # Check 2-minute cache
        if cache_key in self.cache:
            price, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 120:
                return price
        
        # Try multiple sources
        sources = [
            self._get_nse_price,
            self._get_bse_price,
            self._get_moneycontrol_price,
            self._get_yahoo_price
        ]
        
        for source in sources:
            try:
                price = source(symbol)
                if price and 10 <= price <= 50000:  # Validate price range
                    self.cache[cache_key] = (price, datetime.now())
                    return price
            except Exception as e:
                logger.debug(f"Source failed for {symbol}: {e}")
                continue
        
        return None
    
    def _get_nse_price(self, symbol: str) -> Optional[float]:
        """Scrape from NSE website"""
        try:
            # NSE quote page
            url = f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}"
            
            # Get main page first for cookies
            self.session.get("https://www.nseindia.com", timeout=10)
            time.sleep(1)
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for price in various elements
                selectors = [
                    'span[id*="lastPrice"]',
                    '.quoteLtp',
                    '#lastPrice',
                    'span.number',
                    '.price-current'
                ]
                
                for selector in selectors:
                    elem = soup.select_one(selector)
                    if elem:
                        text = elem.get_text().strip().replace(',', '')
                        match = re.search(r'[\d.]+', text)
                        if match:
                            price = float(match.group())
                            logger.info(f"[NSE] {symbol}: Rs.{price:.2f}")
                            return price
                            
        except Exception as e:
            logger.debug(f"NSE scraping failed for {symbol}: {e}")
        return None
    
    def _get_bse_price(self, symbol: str) -> Optional[float]:
        """Scrape from BSE website"""
        try:
            # BSE codes
            bse_codes = {
                'SBIN': '500112',
                'TCS': '532540',
                'INFY': '500209',
                'ITC': '500875',
                'RELIANCE': '500325'
            }
            
            if symbol not in bse_codes:
                return None
            
            bse_code = bse_codes[symbol]
            url = f"https://www.bseindia.com/stock-share-price/{symbol.lower()}/{bse_code}/"
            
            time.sleep(2)  # Rate limiting
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                selectors = [
                    'span.CurrentRate',
                    '.quote-price',
                    'strong[id*="idcrval"]',
                    '.LTP'
                ]
                
                for selector in selectors:
                    elem = soup.select_one(selector)
                    if elem:
                        text = elem.get_text().strip().replace(',', '')
                        match = re.search(r'[\d.]+', text)
                        if match:
                            price = float(match.group())
                            logger.info(f"[BSE] {symbol}: Rs.{price:.2f}")
                            return price
                            
        except Exception as e:
            logger.debug(f"BSE scraping failed for {symbol}: {e}")
        return None
    
    def _get_moneycontrol_price(self, symbol: str) -> Optional[float]:
        """Scrape from MoneyControl"""
        try:
            # MoneyControl search API
            search_url = f"https://www.moneycontrol.com/mccode/common/autosuggestion_solr.php?classic=true&query={symbol}&type=1&format=json&callback=suggest1"
            
            time.sleep(1)
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                # Extract JSON from JSONP
                json_str = response.text.replace('suggest1(', '').replace(');', '')
                data = json.loads(json_str)
                
                if data and len(data) > 0:
                    stock_data = data[0]
                    if 'pricecurrent' in stock_data:
                        price = float(stock_data['pricecurrent'])
                        logger.info(f"[MONEYCONTROL] {symbol}: Rs.{price:.2f}")
                        return price
                        
        except Exception as e:
            logger.debug(f"MoneyControl failed for {symbol}: {e}")
        return None
    
    def _get_yahoo_price(self, symbol: str) -> Optional[float]:
        """Yahoo Finance as backup"""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="1d", interval="1d")
            
            if not data.empty:
                price = float(data['Close'].iloc[-1])
                logger.info(f"[YAHOO] {symbol}: Rs.{price:.2f}")
                return price
                
        except Exception as e:
            logger.debug(f"Yahoo failed for {symbol}: {e}")
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data"""
        try:
            import yfinance as yf
            
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
                
                logger.info(f"[HIST] {symbol}: {len(data)} days")
                return data
                
        except Exception as e:
            logger.error(f"Historical data failed for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def get_market_status(self) -> Dict:
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
nse_bse_real_scraper = NSEBSERealScraper()