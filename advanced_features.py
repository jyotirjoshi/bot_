"""
Advanced Trading Features - Professional Grade
Kelly Criterion, Overnight Swings, High-Frequency, Alternative Data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import requests
import yfinance as yf
from scipy import stats
from sklearn.ensemble import VotingRegressor
import tensorflow as tf
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class KellyCriterionSizer:
    """Kelly Criterion for optimal position sizing"""
    
    def __init__(self, lookback_trades: int = 50):
        self.lookback_trades = lookback_trades
        self.trade_history = []
    
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly fraction: f = (bp - q) / b"""
        if avg_loss == 0 or win_rate == 0:
            return 0.02  # Default 2%
        
        b = abs(avg_win / avg_loss)  # Win/loss ratio
        p = win_rate  # Win probability
        q = 1 - p     # Loss probability
        
        kelly_fraction = (b * p - q) / b
        
        # Cap at 25% for safety (fractional Kelly)
        return max(0.01, min(kelly_fraction * 0.25, 0.25))
    
    def get_optimal_position_size(self, capital: float, price: float, 
                                signal_confidence: float) -> int:
        """Get Kelly-optimized position size"""
        if len(self.trade_history) < 10:
            return int((capital * 0.02) / price)  # Default 2%
        
        recent_trades = self.trade_history[-self.lookback_trades:]
        wins = [t for t in recent_trades if t > 0]
        losses = [t for t in recent_trades if t < 0]
        
        if not wins or not losses:
            return int((capital * 0.02) / price)
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        # Adjust by signal confidence
        adjusted_fraction = kelly_fraction * signal_confidence
        
        quantity = int((capital * adjusted_fraction) / price)
        return max(1, quantity)

class OvernightSwingManager:
    """Overnight swing trades with leverage"""
    
    def __init__(self, capital: float):
        self.capital = capital
        self.overnight_allocation = 0.25  # 25% for overnight
        self.leverage = 2.5  # 2.5x leverage
        self.overnight_positions = {}
    
    def identify_gap_candidates(self, watchlist: List[str]) -> List[Dict]:
        """Identify stocks likely to gap"""
        candidates = []
        
        for symbol in watchlist:
            try:
                # Get after-hours data and news sentiment
                ticker = yf.Ticker(f"{symbol}.NS")
                data = ticker.history(period="5d")
                
                if len(data) < 3:
                    continue
                
                # Calculate volatility and momentum
                returns = data['Close'].pct_change()
                volatility = returns.std()
                momentum = (data['Close'].iloc[-1] / data['Close'].iloc[-3] - 1)
                
                # Volume spike detection
                avg_volume = data['Volume'].rolling(3).mean().iloc[-2]
                current_volume = data['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # Gap probability score
                gap_score = (abs(momentum) * 2 + volatility * 3 + 
                           min(volume_ratio, 3) * 0.5) / 5.5
                
                if gap_score > 0.6:  # High gap probability
                    candidates.append({
                        'symbol': symbol,
                        'gap_score': gap_score,
                        'momentum': momentum,
                        'volatility': volatility,
                        'volume_ratio': volume_ratio
                    })
            except:
                continue
        
        return sorted(candidates, key=lambda x: x['gap_score'], reverse=True)[:3]
    
    def place_overnight_trades(self, candidates: List[Dict]) -> List[Dict]:
        """Place overnight swing trades"""
        overnight_capital = self.capital * self.overnight_allocation * self.leverage
        trades = []
        
        for candidate in candidates:
            try:
                symbol = candidate['symbol']
                current_price = yf.Ticker(f"{symbol}.NS").history(period="1d")['Close'].iloc[-1]
                
                # Position size with leverage
                position_value = overnight_capital / len(candidates)
                quantity = int(position_value / current_price)
                
                if quantity > 0:
                    # Determine direction based on momentum
                    action = "BUY" if candidate['momentum'] > 0 else "SELL"
                    
                    # Set wider stops for overnight (5%)
                    stop_loss = current_price * (0.95 if action == "BUY" else 1.05)
                    target = current_price * (1.15 if action == "BUY" else 0.85)  # 15% target
                    
                    trade = {
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'price': current_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'leverage': self.leverage,
                        'type': 'overnight_swing'
                    }
                    
                    trades.append(trade)
                    logger.info(f"[OVERNIGHT] {action} {quantity} {symbol} @ Rs.{current_price:.2f}")
            except:
                continue
        
        return trades

class HighFrequencyEngine:
    """High-frequency trading with 5-second cycles"""
    
    def __init__(self):
        self.cycle_time = 5  # 5 seconds
        self.tick_data = {}
        self.order_book = {}
    
    def get_level2_data(self, symbol: str) -> Dict:
        """Simulate Level 2 order book data"""
        # In real implementation, connect to NSE/BSE Level 2 feeds
        try:
            current_price = yf.Ticker(f"{symbol}.NS").history(period="1d", interval="1m")['Close'].iloc[-1]
            
            # Simulate bid-ask spread and depth
            spread = current_price * 0.001  # 0.1% spread
            
            bid_ask = {
                'bid_price': current_price - spread/2,
                'ask_price': current_price + spread/2,
                'bid_size': np.random.randint(100, 1000),
                'ask_size': np.random.randint(100, 1000),
                'market_depth': np.random.randint(5000, 20000),
                'imbalance_ratio': np.random.uniform(0.3, 0.7)  # Bid/Ask imbalance
            }
            
            return bid_ask
        except:
            return {}
    
    def detect_institutional_orders(self, symbol: str) -> Dict:
        """Detect large institutional orders (iceberg detection)"""
        level2 = self.get_level2_data(symbol)
        
        if not level2:
            return {'detected': False}
        
        # Iceberg detection logic
        imbalance = level2.get('imbalance_ratio', 0.5)
        depth = level2.get('market_depth', 0)
        
        # Large order detected if significant imbalance + high depth
        institutional_signal = {
            'detected': imbalance < 0.3 or imbalance > 0.7,
            'direction': 'BUY' if imbalance > 0.7 else 'SELL',
            'strength': abs(imbalance - 0.5) * 2,
            'depth': depth
        }
        
        return institutional_signal
    
    def execute_hft_cycle(self, symbol: str, current_price: float) -> Dict:
        """Execute high-frequency trading cycle"""
        # Get institutional signals
        inst_signal = self.detect_institutional_orders(symbol)
        
        # Get micro-trend (1-minute momentum)
        try:
            data = yf.Ticker(f"{symbol}.NS").history(period="1d", interval="1m")
            if len(data) >= 3:
                micro_momentum = (data['Close'].iloc[-1] / data['Close'].iloc[-3] - 1) * 100
            else:
                micro_momentum = 0
        except:
            micro_momentum = 0
        
        # HFT signal generation
        hft_signal = 0
        if inst_signal['detected'] and abs(micro_momentum) > 0.1:
            direction = 1 if inst_signal['direction'] == 'BUY' else -1
            momentum_direction = 1 if micro_momentum > 0 else -1
            
            if direction == momentum_direction:  # Alignment
                hft_signal = direction * inst_signal['strength']
        
        return {
            'signal': hft_signal,
            'institutional': inst_signal,
            'micro_momentum': micro_momentum,
            'confidence': abs(hft_signal)
        }

class TransformerSentimentAnalyzer:
    """Advanced sentiment analysis using Transformer models"""
    
    def __init__(self):
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                             model="nlptown/bert-base-multilingual-uncased-sentiment")
            self.news_sources = [
                "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
                "https://feeds.finance.yahoo.com/rss/2.0/headline"
            ]
        except:
            self.sentiment_pipeline = None
    
    def get_news_sentiment(self, symbol: str) -> float:
        """Get news sentiment using BERT"""
        if not self.sentiment_pipeline:
            return 0.0
        
        try:
            # Get recent news
            ticker = yf.Ticker(f"{symbol}.NS")
            news = ticker.news
            
            if not news:
                return 0.0
            
            sentiments = []
            for article in news[:5]:
                title = article.get('title', '')
                if title:
                    result = self.sentiment_pipeline(title)[0]
                    
                    # Convert to -1 to +1 scale
                    if result['label'] == 'POSITIVE':
                        score = result['score']
                    else:
                        score = -result['score']
                    
                    sentiments.append(score)
            
            return np.mean(sentiments) if sentiments else 0.0
        except:
            return 0.0
    
    def get_social_sentiment(self, symbol: str) -> float:
        """Get social media sentiment (simulated)"""
        # In real implementation, connect to Twitter/StockTwits APIs
        return np.random.uniform(-0.3, 0.3)  # Simulated social sentiment

class CausalInferenceEngine:
    """Causal inference for trading patterns"""
    
    def __init__(self):
        self.causal_patterns = {}
    
    def analyze_gap_causality(self, data: pd.DataFrame) -> Dict:
        """Analyze if gap + volume predicts upside"""
        if len(data) < 30:
            return {'causal_strength': 0}
        
        # Create features
        data['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(5).mean()
        data['future_return'] = data['Close'].shift(-1) / data['Close'] - 1
        
        # Filter for significant gaps
        gap_days = data[abs(data['gap']) > 0.02].copy()
        
        if len(gap_days) < 10:
            return {'causal_strength': 0}
        
        # Causal analysis: Gap + Volume → Future Return
        X = gap_days[['gap', 'volume_ratio']].fillna(0)
        y = gap_days['future_return'].fillna(0)
        
        # Simple correlation-based causality
        correlation = np.corrcoef(X.mean(axis=1), y)[0, 1]
        
        return {
            'causal_strength': abs(correlation),
            'direction': 1 if correlation > 0 else -1,
            'confidence': min(abs(correlation) * 2, 1.0)
        }

class PPOPositionSizer:
    """Proximal Policy Optimization for position sizing"""
    
    def __init__(self, state_dim: int = 10, action_dim: int = 5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self._build_ppo_model()
        self.memory = []
    
    def _build_ppo_model(self):
        """Build PPO neural network"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model
    
    def get_state(self, market_data: Dict) -> np.ndarray:
        """Create state vector for PPO"""
        state = np.array([
            market_data.get('price_momentum', 0),
            market_data.get('volatility', 0),
            market_data.get('volume_ratio', 1),
            market_data.get('rsi', 50) / 100,
            market_data.get('signal_strength', 0),
            market_data.get('portfolio_heat', 0),
            market_data.get('time_of_day', 0.5),
            market_data.get('market_regime', 0),
            market_data.get('sentiment', 0),
            market_data.get('causal_strength', 0)
        ])
        return np.nan_to_num(state)
    
    def get_position_size(self, market_data: Dict, capital: float, price: float) -> int:
        """Get PPO-optimized position size"""
        state = self.get_state(market_data)
        
        # Get action probabilities
        action_probs = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        action = np.argmax(action_probs)
        
        # Convert action to position size (0%, 5%, 10%, 15%, 20%)
        size_percentages = [0, 0.05, 0.10, 0.15, 0.20]
        allocation = capital * size_percentages[action]
        
        return int(allocation / price) if price > 0 else 0

class EnsembleMetaLearner:
    """Meta-learner to combine all signals"""
    
    def __init__(self):
        self.meta_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(8,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh')
        ])
        self.meta_model.compile(optimizer='adam', loss='mse')
    
    def combine_signals(self, signals: Dict) -> Dict:
        """Combine all signals using meta-learner"""
        # Extract signal features
        features = np.array([
            signals.get('technical', 0),
            signals.get('lstm_prediction', 0),
            signals.get('xgboost_signal', 0),
            signals.get('rl_signal', 0),
            signals.get('sentiment', 0),
            signals.get('hft_signal', 0),
            signals.get('causal_signal', 0),
            signals.get('overnight_momentum', 0)
        ])
        
        # Get meta-prediction
        meta_signal = self.meta_model.predict(features.reshape(1, -1), verbose=0)[0][0]
        
        # Calculate confidence based on signal agreement
        signal_values = [v for v in features if abs(v) > 0.1]
        agreement = len([s for s in signal_values if np.sign(s) == np.sign(meta_signal)]) / max(len(signal_values), 1)
        
        return {
            'final_signal': meta_signal,
            'confidence': agreement,
            'signal_count': len(signal_values)
        }

class StatisticalArbitrage:
    """Pairs trading and statistical arbitrage"""
    
    def __init__(self):
        self.pairs = [
            ('SBIN', 'ICICIBANK'),
            ('TCS', 'INFY'),
            ('RELIANCE', 'ONGC')
        ]
    
    def find_cointegrated_pairs(self, data1: pd.Series, data2: pd.Series) -> Dict:
        """Test for cointegration using ADF test"""
        try:
            from statsmodels.tsa.stattools import coint
            
            # Cointegration test
            score, pvalue, _ = coint(data1, data2)
            
            # Calculate spread
            spread = data1 - data2
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            current_spread = spread.iloc[-1]
            z_score = (current_spread - spread_mean) / spread_std
            
            return {
                'cointegrated': pvalue < 0.05,
                'z_score': z_score,
                'signal': 'LONG_1_SHORT_2' if z_score < -2 else 'LONG_2_SHORT_1' if z_score > 2 else 'HOLD'
            }
        except:
            return {'cointegrated': False}

class CompoundingEngine:
    """Daily profit compounding"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_returns = []
    
    def compound_daily_pnl(self, daily_pnl: float) -> float:
        """Compound daily P&L"""
        daily_return = daily_pnl / self.current_capital
        self.daily_returns.append(daily_return)
        
        # Compound the capital
        self.current_capital += daily_pnl
        
        # Calculate cumulative return
        cumulative_return = (self.current_capital / self.initial_capital - 1) * 100
        
        logger.info(f"💰 Capital: Rs.{self.current_capital:,.2f} (+{cumulative_return:.2f}%)")
        
        return self.current_capital
    
    def get_compounding_stats(self) -> Dict:
        """Get compounding statistics"""
        if not self.daily_returns:
            return {}
        
        returns_array = np.array(self.daily_returns)
        
        return {
            'current_capital': self.current_capital,
            'total_return': (self.current_capital / self.initial_capital - 1) * 100,
            'daily_avg_return': np.mean(returns_array) * 100,
            'volatility': np.std(returns_array) * 100,
            'sharpe_ratio': np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown()
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.daily_returns) < 2:
            return 0
        
        cumulative = np.cumprod(1 + np.array(self.daily_returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown)) * 100