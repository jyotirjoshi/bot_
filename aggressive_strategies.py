"""
Aggressive Trading Strategies for Maximum Profit
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class AggressiveStrategies:
    """Multiple aggressive trading strategies"""
    
    def __init__(self):
        self.strategies = [
            self.momentum_breakout,
            self.volume_spike,
            self.gap_trading,
            self.reversal_scalping,
            self.trend_following,
            self.vwap_strategy,
            self.opening_range_breakout
        ]
    
    def momentum_breakout(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Momentum breakout strategy"""
        if len(data) < 20:
            return {'signal': 0, 'strength': 0, 'strategy': 'momentum_breakout'}
        
        # 20-day high/low breakout
        high_20 = data['high'].rolling(20).max().iloc[-1]
        low_20 = data['low'].rolling(20).min().iloc[-1]
        
        # Volume confirmation
        avg_volume = data['volume'].rolling(10).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        signal = 0
        strength = 0
        
        if current_price > high_20 * 1.001:  # Breakout above 20-day high
            signal = 0.8
            strength = 0.7 if current_volume > avg_volume * 1.5 else 0.5
        elif current_price < low_20 * 0.999:  # Breakdown below 20-day low
            signal = -0.8
            strength = 0.7 if current_volume > avg_volume * 1.5 else 0.5
        
        return {'signal': signal, 'strength': strength, 'strategy': 'momentum_breakout'}
    
    def volume_spike(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Volume spike strategy"""
        if len(data) < 10:
            return {'signal': 0, 'strength': 0, 'strategy': 'volume_spike'}
        
        avg_volume = data['volume'].rolling(10).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Price change with volume
        price_change = (current_price - data['close'].iloc[-2]) / data['close'].iloc[-2]
        
        signal = 0
        strength = 0
        
        if current_volume > avg_volume * 2:  # 2x volume spike
            if price_change > 0.01:  # 1% price increase
                signal = 0.7
                strength = min(current_volume / avg_volume / 3, 1.0)
            elif price_change < -0.01:  # 1% price decrease
                signal = -0.7
                strength = min(current_volume / avg_volume / 3, 1.0)
        
        return {'signal': signal, 'strength': strength, 'strategy': 'volume_spike'}
    
    def gap_trading(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Gap trading strategy"""
        if len(data) < 2:
            return {'signal': 0, 'strength': 0, 'strategy': 'gap_trading'}
        
        prev_close = data['close'].iloc[-2]
        gap_pct = (current_price - prev_close) / prev_close
        
        signal = 0
        strength = 0
        
        if gap_pct > 0.02:  # 2% gap up
            signal = 0.6  # Buy the gap
            strength = min(abs(gap_pct) * 10, 1.0)
        elif gap_pct < -0.02:  # 2% gap down
            signal = 0.6  # Buy the dip
            strength = min(abs(gap_pct) * 10, 1.0)
        
        return {'signal': signal, 'strength': strength, 'strategy': 'gap_trading'}
    
    def reversal_scalping(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Quick reversal scalping"""
        if len(data) < 5:
            return {'signal': 0, 'strength': 0, 'strategy': 'reversal_scalping'}
        
        # Short-term RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(5).mean()
        rs = gain / loss
        rsi_5 = 100 - (100 / (1 + rs))
        
        current_rsi = rsi_5.iloc[-1]
        
        signal = 0
        strength = 0
        
        if current_rsi < 20:  # Extremely oversold
            signal = 0.9
            strength = 0.8
        elif current_rsi > 80:  # Extremely overbought
            signal = -0.9
            strength = 0.8
        
        return {'signal': signal, 'strength': strength, 'strategy': 'reversal_scalping'}
    
    def trend_following(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Aggressive trend following"""
        if len(data) < 10:
            return {'signal': 0, 'strength': 0, 'strategy': 'trend_following'}
        
        # Multiple EMAs
        ema_5 = data['close'].ewm(span=5).mean().iloc[-1]
        ema_10 = data['close'].ewm(span=10).mean().iloc[-1]
        ema_20 = data['close'].ewm(span=20).mean().iloc[-1]
        
        signal = 0
        strength = 0
        
        # Strong uptrend
        if current_price > ema_5 > ema_10 > ema_20:
            signal = 0.8
            strength = 0.7
        # Strong downtrend
        elif current_price < ema_5 < ema_10 < ema_20:
            signal = -0.8
            strength = 0.7
        
        return {'signal': signal, 'strength': strength, 'strategy': 'trend_following'}
    
    def vwap_strategy(self, data: pd.DataFrame, current_price: float) -> Dict:
        """VWAP (Volume Weighted Average Price) strategy"""
        if len(data) < 5:
            return {'signal': 0, 'strength': 0, 'strategy': 'vwap_strategy'}
        
        # Calculate VWAP
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        current_vwap = vwap.iloc[-1]
        
        # Price distance from VWAP
        vwap_distance = (current_price - current_vwap) / current_vwap
        
        signal = 0
        strength = 0
        
        # Buy when price is below VWAP (value buying)
        if vwap_distance < -0.01:  # 1% below VWAP
            signal = 0.7
            strength = min(abs(vwap_distance) * 50, 0.9)
        # Sell when price is above VWAP (momentum selling)
        elif vwap_distance > 0.01:  # 1% above VWAP
            signal = -0.7
            strength = min(abs(vwap_distance) * 50, 0.9)
        
        return {'signal': signal, 'strength': strength, 'strategy': 'vwap_strategy'}
    
    def opening_range_breakout(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Opening Range Breakout - First 30 minutes range"""
        if len(data) < 2:
            return {'signal': 0, 'strength': 0, 'strategy': 'opening_range_breakout'}
        
        # Use first candle as opening range (simulated)
        opening_high = data['high'].iloc[0]
        opening_low = data['low'].iloc[0]
        opening_range = opening_high - opening_low
        
        # Recent high/low for breakout confirmation
        recent_high = data['high'].tail(5).max()
        recent_low = data['low'].tail(5).min()
        
        signal = 0
        strength = 0
        
        # Breakout above opening range
        if current_price > opening_high and current_price > recent_high:
            signal = 0.8
            strength = 0.8
        # Breakdown below opening range
        elif current_price < opening_low and current_price < recent_low:
            signal = 0.6  # Buy the dip on breakdown
            strength = 0.7
        
        return {'signal': signal, 'strength': strength, 'strategy': 'opening_range_breakout'}
    
    def get_combined_signals(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Combine all aggressive strategies"""
        all_signals = []
        
        for strategy in self.strategies:
            try:
                result = strategy(data, current_price)
                if result['strength'] > 0.3:  # Only consider strong signals
                    all_signals.append(result)
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} failed: {e}")
        
        if not all_signals:
            return {'signal': 0, 'strength': 0, 'strategy': 'no_aggressive_signal'}
        
        # Weighted combination
        total_signal = sum(s['signal'] * s['strength'] for s in all_signals)
        total_weight = sum(s['strength'] for s in all_signals)
        
        final_signal = total_signal / total_weight if total_weight > 0 else 0
        final_strength = total_weight / len(all_signals)
        
        strategies_used = ', '.join([s['strategy'] for s in all_signals])
        
        return {
            'signal': final_signal,
            'strength': final_strength,
            'strategy': f"Aggressive: {strategies_used}"
        }

class AdvancedSignalFusion:
    """Advanced signal fusion with intelligent scoring"""
    
    def __init__(self):
        self.signal_weights = {
            'momentum_breakout': 0.15,
            'volume_spike': 0.12,
            'gap_trading': 0.10,
            'reversal_scalping': 0.13,
            'trend_following': 0.15,
            'vwap_strategy': 0.20,
            'opening_range_breakout': 0.15
        }
        self.confidence_multipliers = {
            'high_volume': 1.3,
            'trend_alignment': 1.2,
            'multiple_timeframes': 1.4,
            'market_regime': 1.1
        }
    
    def calculate_signal_score(self, signals: Dict, market_data: pd.DataFrame) -> Dict:
        """Calculate advanced signal score with multiple factors"""
        if not signals:
            return {'score': 0, 'confidence': 0, 'quality': 'poor'}
        
        # Base weighted score
        weighted_score = 0
        total_weight = 0
        
        for strategy, signal_data in signals.items():
            if strategy in self.signal_weights:
                weight = self.signal_weights[strategy]
                signal_strength = signal_data.get('strength', 0)
                signal_value = signal_data.get('signal', 0)
                
                weighted_score += signal_value * signal_strength * weight
                total_weight += weight * signal_strength
        
        base_score = weighted_score / max(total_weight, 0.1)
        
        # Apply confidence multipliers
        confidence_boost = 1.0
        
        # High volume confirmation
        if len(market_data) > 0:
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].rolling(10).mean().iloc[-1]
            if current_volume > avg_volume * 1.5:
                confidence_boost *= self.confidence_multipliers['high_volume']
        
        # Multiple signal agreement
        bullish_signals = sum(1 for s in signals.values() if s.get('signal', 0) > 0.3)
        bearish_signals = sum(1 for s in signals.values() if s.get('signal', 0) < -0.3)
        
        if max(bullish_signals, bearish_signals) >= 3:
            confidence_boost *= self.confidence_multipliers['multiple_timeframes']
        
        final_score = np.clip(base_score * confidence_boost, -1, 1)
        final_confidence = min(abs(final_score) * confidence_boost, 1.0)
        
        # Quality assessment
        if final_confidence > 0.8:
            quality = 'excellent'
        elif final_confidence > 0.6:
            quality = 'good'
        elif final_confidence > 0.4:
            quality = 'fair'
        else:
            quality = 'poor'
        
        return {
            'score': final_score,
            'confidence': final_confidence,
            'quality': quality,
            'signal_count': len(signals),
            'agreement': max(bullish_signals, bearish_signals) / len(signals)
        }

class IntelligentPositionSizing:
    """Intelligent position sizing based on multiple factors"""
    
    def __init__(self, capital: float):
        self.capital = capital
        self.base_risk = 0.02  # 2% base risk
        self.max_risk = 0.05   # 5% maximum risk
        self.volatility_adjustment = True
        self.signal_scaling = True
    
    def calculate_intelligent_size(self, price: float, signal_data: Dict, 
                                 market_data: pd.DataFrame, current_positions: int) -> Dict:
        """Calculate intelligent position size with multiple factors"""
        
        # Base position size calculation
        signal_score = signal_data.get('score', 0)
        signal_confidence = signal_data.get('confidence', 0)
        signal_quality = signal_data.get('quality', 'poor')
        
        # Risk adjustment based on signal quality
        quality_multipliers = {
            'excellent': 1.5,
            'good': 1.2,
            'fair': 1.0,
            'poor': 0.7
        }
        
        risk_multiplier = quality_multipliers.get(signal_quality, 1.0)
        
        # Confidence scaling
        confidence_multiplier = 0.5 + (signal_confidence * 1.5)  # 0.5x to 2.0x
        
        # Volatility adjustment
        volatility_multiplier = 1.0
        if len(market_data) >= 20:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            
            if volatility > 0.03:  # High volatility
                volatility_multiplier = 0.8
            elif volatility < 0.015:  # Low volatility
                volatility_multiplier = 1.3
        
        # Position concentration limit
        max_positions = 6
        concentration_multiplier = max(0.5, 1 - (current_positions / max_positions) * 0.3)
        
        # Calculate final risk percentage
        final_risk = self.base_risk * risk_multiplier * confidence_multiplier * volatility_multiplier * concentration_multiplier
        final_risk = np.clip(final_risk, 0.005, self.max_risk)  # 0.5% to 5%
        
        # Calculate position size
        risk_amount = self.capital * final_risk
        
        # Estimate stop loss (2% default)
        stop_distance = price * 0.02
        if len(market_data) >= 10:
            atr = self._calculate_atr(market_data)
            stop_distance = max(atr * 1.5, price * 0.015)  # ATR-based or 1.5% minimum
        
        quantity = int(risk_amount / stop_distance)
        
        # Minimum and maximum position limits
        min_trade_value = 3000
        max_allocation = self.capital * 0.25  # 25% max per trade
        
        if quantity * price < min_trade_value:
            quantity = 0
        elif quantity * price > max_allocation:
            quantity = int(max_allocation / price)
        
        return {
            'quantity': quantity,
            'risk_amount': risk_amount,
            'risk_percentage': final_risk * 100,
            'stop_distance': stop_distance,
            'trade_value': quantity * price,
            'quality_factor': risk_multiplier,
            'confidence_factor': confidence_multiplier,
            'volatility_factor': volatility_multiplier
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(true_range).rolling(period).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else data['close'].iloc[-1] * 0.02

class ProfitMaximizer:
    """Enhanced profit maximization with intelligent systems"""
    
    def __init__(self, capital: float):
        self.capital = capital
        self.signal_fusion = AdvancedSignalFusion()
        self.position_sizer = IntelligentPositionSizing(capital)
    
    def calculate_aggressive_position_size(self, price: float, signal_data: Dict, 
                                         market_data: pd.DataFrame, current_positions: int) -> int:
        """Calculate intelligent aggressive position size"""
        sizing_result = self.position_sizer.calculate_intelligent_size(
            price, signal_data, market_data, current_positions
        )
        return sizing_result['quantity']
    
    def get_enhanced_signals(self, all_signals: Dict, market_data: pd.DataFrame) -> Dict:
        """Get enhanced signals with advanced fusion"""
        return self.signal_fusion.calculate_signal_score(all_signals, market_data)
    
    def dynamic_stop_loss(self, entry_price: float, current_price: float, action: str, volatility: float) -> float:
        """Dynamic stop loss based on market conditions"""
        # Base stop distance
        base_stop = entry_price * max(volatility, 0.015)  # 1.5% minimum
        
        # Tighter stops in low volatility
        if volatility < 0.02:
            base_stop *= 0.8
        
        if action == "BUY":
            return entry_price - base_stop
        else:
            return entry_price + base_stop
    
    def profit_taking_levels(self, entry_price: float, action: str, signal_strength: float) -> List[Dict]:
        """Multiple profit taking levels"""
        levels = []
        
        # Base profit targets
        targets = [0.02, 0.04, 0.06]  # 2%, 4%, 6%
        
        # Adjust based on signal strength
        if signal_strength > 0.8:
            targets = [0.015, 0.03, 0.05, 0.08]  # More aggressive targets
        
        for i, target_pct in enumerate(targets):
            if action == "BUY":
                target_price = entry_price * (1 + target_pct)
            else:
                target_price = entry_price * (1 - target_pct)
            
            levels.append({
                'level': i + 1,
                'price': target_price,
                'percentage': int(100 / len(targets)),  # Equal distribution
                'target_pct': target_pct * 100
            })
        
        return levels
    
    def should_trade_intelligent(self, signal_data: Dict, market_conditions: Dict) -> bool:
        """Intelligent trading decision with multiple factors"""
        score = signal_data.get('score', 0)
        confidence = signal_data.get('confidence', 0)
        quality = signal_data.get('quality', 'poor')
        
        # Base thresholds
        min_score = 0.3
        min_confidence = 0.4
        
        # Quality-based adjustments
        if quality == 'excellent':
            min_score = 0.2
            min_confidence = 0.3
        elif quality == 'poor':
            min_score = 0.5
            min_confidence = 0.6
        
        # Market condition adjustments
        if market_conditions.get('high_volatility', False):
            min_confidence += 0.1
        
        return abs(score) >= min_score and confidence >= min_confidence