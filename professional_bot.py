"""
Professional Trading Bot - Institutional Grade
All Advanced Features Integrated
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_trading_bot import EnhancedTradingBot
from advanced_features import *
import logging
import time
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class ProfessionalTradingBot(EnhancedTradingBot):
    """Professional-grade trading bot with all advanced features"""
    
    def __init__(self, capital: float = 50000):
        super().__init__(capital)
        
        # Use market simulator to avoid rate limits
        from market_simulator import market_simulator
        self.data_provider = market_simulator
        
        # Initialize all advanced components
        self.kelly_sizer = KellyCriterionSizer()
        self.overnight_manager = OvernightSwingManager(capital)
        self.hft_engine = HighFrequencyEngine()
        self.sentiment_analyzer = TransformerSentimentAnalyzer()
        self.causal_engine = CausalInferenceEngine()
        self.ppo_sizer = PPOPositionSizer()
        self.meta_learner = EnsembleMetaLearner()
        self.stat_arb = StatisticalArbitrage()
        self.compounding_engine = CompoundingEngine(capital)
        
        # Professional settings
        self.cycle_time = 5  # 5-second cycles
        self.max_risk = 0.15  # 15% max risk during high volatility
        self.overnight_active = True
        
        logger.info("🏛️ Professional Trading Bot initialized with institutional features")
        logger.info("⚡ 5-second cycles | 🌙 Overnight swings | 🧠 AI ensemble | 💎 Kelly sizing")
    
    def run_professional_cycle(self):
        """Run professional trading cycle with all features"""
        current_time = datetime.now()
        
        # 1. High-frequency signals (every 5 seconds)
        hft_signals = {}
        for symbol in self.watchlist:
            try:
                current_price = self.data_provider.get_current_price(symbol)
                if current_price:
                    hft_data = self.hft_engine.execute_hft_cycle(symbol, current_price)
                    hft_signals[symbol] = hft_data
            except:
                continue
        
        # 2. Enhanced sentiment analysis
        sentiment_scores = {}
        for symbol in self.watchlist:
            news_sentiment = self.sentiment_analyzer.get_news_sentiment(symbol)
            social_sentiment = self.sentiment_analyzer.get_social_sentiment(symbol)
            sentiment_scores[symbol] = (news_sentiment + social_sentiment) / 2
        
        # 3. Causal inference signals
        causal_signals = {}
        for symbol in self.watchlist:
            try:
                hist_data = self.data_provider.get_historical_data(symbol, 30)
                if not hist_data.empty:
                    causal_data = self.causal_engine.analyze_gap_causality(hist_data)
                    causal_signals[symbol] = causal_data
            except:
                causal_signals[symbol] = {'causal_strength': 0}
        
        # 4. Statistical arbitrage opportunities
        arb_signals = self.check_arbitrage_opportunities()
        
        # 5. Process each symbol with ensemble approach
        for symbol in self.watchlist:
            if symbol in self.positions:
                continue  # Skip if already have position
            
            try:
                # Collect all signals
                all_signals = self.collect_all_signals(symbol, hft_signals, 
                                                     sentiment_scores, causal_signals)
                
                # Meta-learner fusion
                meta_result = self.meta_learner.combine_signals(all_signals)
                
                if abs(meta_result['final_signal']) > 0.3 and meta_result['confidence'] > 0.6:
                    # Professional position sizing
                    position_size = self.calculate_professional_position_size(
                        symbol, meta_result, all_signals
                    )
                    
                    if position_size > 0:
                        self.execute_professional_trade(symbol, meta_result, position_size)
            
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
        
        # 6. Update positions with advanced exit management
        self.update_positions()
        
        # 7. Overnight swing management (after 3 PM)
        if current_time.hour >= 15 and self.overnight_active:
            self.manage_overnight_swings()
        
        # 8. Daily compounding (end of day)
        if current_time.hour >= 15 and current_time.minute >= 30:
            self.compound_daily_profits()
    
    def collect_all_signals(self, symbol: str, hft_signals: Dict, 
                          sentiment_scores: Dict, causal_signals: Dict) -> Dict:
        """Collect all available signals for a symbol"""
        signals = {}
        
        try:
            # Technical signals
            hist_data = self.data_provider.get_historical_data(symbol, 90)
            if not hist_data.empty:
                tech_signals = self.technical_analyzer.generate_signals(hist_data)
                signals['technical'] = tech_signals['signal']
            
            # ML signals
            if self.ml_trained and self.ml_engine:
                current_price = self.data_provider.get_current_price(symbol)
                if current_price:
                    portfolio_value = self.get_portfolio_value()
                    ml_signals = self.ml_engine.get_ml_signals(
                        symbol, hist_data, current_price, portfolio_value, len(self.positions)
                    )
                    signals['lstm_prediction'] = ml_signals.get('signal', 0)
                    signals['xgboost_signal'] = ml_signals.get('signal', 0)
            
            # High-frequency signals
            if symbol in hft_signals:
                signals['hft_signal'] = hft_signals[symbol]['signal']
            
            # Sentiment signals
            if symbol in sentiment_scores:
                signals['sentiment'] = sentiment_scores[symbol]
            
            # Causal signals
            if symbol in causal_signals:
                causal_data = causal_signals[symbol]
                signals['causal_signal'] = (causal_data.get('direction', 0) * 
                                          causal_data.get('causal_strength', 0))
            
            # Overnight momentum
            signals['overnight_momentum'] = self.calculate_overnight_momentum(symbol)
            
        except Exception as e:
            logger.debug(f"Error collecting signals for {symbol}: {e}")
        
        return signals
    
    def calculate_professional_position_size(self, symbol: str, meta_result: Dict, 
                                           all_signals: Dict) -> int:
        """Calculate position size using multiple methods"""
        try:
            current_price = self.data_provider.get_current_price(symbol)
            if not current_price:
                return 0
            
            # Method 1: Kelly Criterion
            kelly_size = self.kelly_sizer.get_optimal_position_size(
                self.compounding_engine.current_capital, current_price, 
                meta_result['confidence']
            )
            
            # Method 2: PPO Reinforcement Learning
            market_data = {
                'signal_strength': meta_result['confidence'],
                'volatility': self.calculate_volatility(symbol),
                'portfolio_heat': len(self.positions) / 6,
                'sentiment': all_signals.get('sentiment', 0),
                'causal_strength': all_signals.get('causal_signal', 0)
            }
            
            ppo_size = self.ppo_sizer.get_position_size(
                market_data, self.compounding_engine.current_capital, current_price
            )
            
            # Ensemble sizing (average of methods)
            final_size = int((kelly_size + ppo_size) / 2)
            
            # Risk limits
            max_position_value = self.compounding_engine.current_capital * 0.25
            max_size = int(max_position_value / current_price)
            
            return min(final_size, max_size)
            
        except Exception as e:
            logger.debug(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    def execute_professional_trade(self, symbol: str, meta_result: Dict, quantity: int):
        """Execute trade with professional features"""
        try:
            current_price = self.data_provider.get_current_price(symbol)
            if not current_price or quantity <= 0:
                return
            
            action = "BUY" if meta_result['final_signal'] > 0 else "SELL"
            
            # ATR-based stop loss
            hist_data = self.data_provider.get_historical_data(symbol, 20)
            if not hist_data.empty:
                atr = self.calculate_atr(hist_data)
                stop_distance = max(atr * 1.5, current_price * 0.015)
            else:
                stop_distance = current_price * 0.02
            
            stop_loss = (current_price - stop_distance if action == "BUY" 
                        else current_price + stop_distance)
            
            # Dynamic target based on signal strength
            risk_reward = 2 + (meta_result['confidence'] * 3)  # 2:1 to 5:1
            target = (current_price + stop_distance * risk_reward if action == "BUY"
                     else current_price - stop_distance * risk_reward)
            
            # Execute the trade
            opportunity = {
                'symbol': symbol,
                'action': action,
                'price': current_price,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'target': target,
                'signal_strength': meta_result['confidence'],
                'strategy': 'Professional_Ensemble'
            }
            
            success = self.execute_trade(opportunity)
            if success:
                logger.info(f"🏛️ [PROFESSIONAL] {action} {quantity} {symbol} @ Rs.{current_price:.2f}")
                logger.info(f"📊 Confidence: {meta_result['confidence']:.2f} | R:R {risk_reward:.1f}:1")
        
        except Exception as e:
            logger.error(f"Error executing professional trade: {e}")
    
    def manage_overnight_swings(self):
        """Manage overnight swing positions"""
        try:
            # Identify gap candidates
            candidates = self.overnight_manager.identify_gap_candidates(self.watchlist)
            
            if candidates:
                overnight_trades = self.overnight_manager.place_overnight_trades(candidates)
                
                for trade in overnight_trades:
                    logger.info(f"🌙 [OVERNIGHT] {trade['action']} {trade['quantity']} "
                              f"{trade['symbol']} @ Rs.{trade['price']:.2f} (Leverage: {trade['leverage']}x)")
        
        except Exception as e:
            logger.error(f"Error managing overnight swings: {e}")
    
    def compound_daily_profits(self):
        """Compound daily profits"""
        try:
            # Calculate daily P&L
            summary = self.get_portfolio_summary()
            daily_pnl = summary.get('daily_pnl', 0)
            
            if daily_pnl != 0:
                new_capital = self.compounding_engine.compound_daily_pnl(daily_pnl)
                
                # Update bot capital
                self.capital = new_capital
                
                # Get compounding stats
                stats = self.compounding_engine.get_compounding_stats()
                
                logger.info(f"📈 DAILY COMPOUND: +{stats.get('total_return', 0):.2f}% "
                          f"(Sharpe: {stats.get('sharpe_ratio', 0):.2f})")
        
        except Exception as e:
            logger.error(f"Error compounding profits: {e}")
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = pd.Series(true_range).rolling(period).mean().iloc[-1]
            
            return atr if not np.isnan(atr) else data['close'].iloc[-1] * 0.02
        except:
            return data['close'].iloc[-1] * 0.02 if not data.empty else 100
    
    def calculate_volatility(self, symbol: str) -> float:
        """Calculate recent volatility"""
        try:
            hist_data = self.data_provider.get_historical_data(symbol, 20)
            if not hist_data.empty:
                returns = hist_data['close'].pct_change().dropna()
                return returns.std() * np.sqrt(252)  # Annualized
            return 0.25  # Default 25%
        except:
            return 0.25
    
    def calculate_overnight_momentum(self, symbol: str) -> float:
        """Calculate overnight momentum signal"""
        try:
            hist_data = self.data_provider.get_historical_data(symbol, 5)
            if len(hist_data) >= 2:
                # Gap from previous close to current open
                gap = (hist_data['open'].iloc[-1] - hist_data['close'].iloc[-2]) / hist_data['close'].iloc[-2]
                return np.clip(gap * 10, -1, 1)  # Scale to -1 to 1
            return 0
        except:
            return 0
    
    def check_arbitrage_opportunities(self) -> Dict:
        """Check statistical arbitrage opportunities"""
        arb_signals = {}
        
        try:
            for pair in self.stat_arb.pairs:
                symbol1, symbol2 = pair
                
                # Get data for both symbols
                data1 = self.data_provider.get_historical_data(symbol1, 30)
                data2 = self.data_provider.get_historical_data(symbol2, 30)
                
                if not data1.empty and not data2.empty:
                    # Test cointegration
                    result = self.stat_arb.find_cointegrated_pairs(
                        data1['close'], data2['close']
                    )
                    
                    if result.get('cointegrated', False):
                        arb_signals[f"{symbol1}_{symbol2}"] = result
        
        except Exception as e:
            logger.debug(f"Error checking arbitrage: {e}")
        
        return arb_signals
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        return (self.available_cash + 
                sum(pos.current_price * pos.quantity for pos in self.positions.values()))
    
    def start_professional_trading(self):
        """Start professional trading with 5-second cycles"""
        self.is_running = True
        logger.info("🏛️ Professional Trading Bot started with 5-second cycles")
        
        while self.is_running:
            try:
                self.run_professional_cycle()
                time.sleep(self.cycle_time)  # 5-second cycles
            except KeyboardInterrupt:
                logger.info("Professional trading stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in professional cycle: {e}")
                time.sleep(10)
        
        self.is_running = False

def main():
    """Main function for professional bot"""
    print("🏛️ PROFESSIONAL TRADING BOT")
    print("=" * 60)
    print("Features:")
    print("⚡ 5-second high-frequency cycles")
    print("🧠 AI ensemble with 8+ signals")
    print("💎 Kelly Criterion position sizing")
    print("🌙 Overnight swing trades (2.5x leverage)")
    print("📊 Transformer sentiment analysis")
    print("🔬 Causal inference engine")
    print("🤖 PPO reinforcement learning")
    print("📈 Daily profit compounding")
    print("=" * 60)
    
    try:
        bot = ProfessionalTradingBot(capital=50000)
        bot.start_professional_trading()
    except KeyboardInterrupt:
        print("\n🛑 Professional bot stopped")

if __name__ == "__main__":
    main()