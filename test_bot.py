"""
Test Trading Bot with Market Simulator
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_trading_bot import EnhancedTradingBot
from market_simulator import market_simulator
import logging
import time
from datetime import datetime, timedelta

# Setup logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestTradingBot(EnhancedTradingBot):
    """Test version of trading bot with simulator"""
    
    def __init__(self, capital: float = 50000):
        # Initialize with simulator as data provider
        super().__init__(capital)
        
        # Replace data provider with simulator
        self.data_provider = market_simulator
        
        # Keep ML but mark as trained to skip training
        self.ml_trained = True
        
        logger.info("🎭 Test Trading Bot initialized with Market Simulator")
        logger.info(f"💰 Starting Capital: Rs.{capital:,}")
        logger.info("🚀 Ready for aggressive testing!")
    
    def run_test_session(self, duration_minutes: int = 30):
        """Run a test trading session"""
        logger.info(f"🎬 Starting {duration_minutes}-minute test session...")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        cycle_count = 0
        self.is_running = True  # Ensure bot is running
        
        while datetime.now() < end_time and self.is_running:
            try:
                cycle_count += 1
                logger.info(f"🔄 Test Cycle #{cycle_count}")
                
                # Create random market events
                market_simulator.create_market_events()
                
                # Run trading cycle
                self.run_trading_cycle()
                
                # Show current status
                self.show_test_status()
                
                # Wait 1 minute (or less for faster testing)
                time.sleep(10)  # 10 seconds for faster testing
                
            except KeyboardInterrupt:
                logger.info("🛑 Test session stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in test cycle: {e}")
                time.sleep(5)
        
        # Final results
        self.show_final_results(cycle_count)
    
    def show_test_status(self):
        """Show current test status"""
        summary = self.get_portfolio_summary()
        
        logger.info("📊 CURRENT STATUS:")
        logger.info(f"   💰 Available Cash: Rs.{summary['available_cash']:,.2f}")
        logger.info(f"   📈 Invested: Rs.{summary['invested_amount']:,.2f}")
        logger.info(f"   💹 Total P&L: Rs.{summary['total_pnl']:,.2f}")
        logger.info(f"   📋 Positions: {summary['positions_count']}")
        logger.info(f"   🔄 Trades: {summary['trades_count']}")
        
        if self.positions:
            logger.info("   🎯 Active Positions:")
            for symbol, pos in self.positions.items():
                pnl_pct = (pos.unrealized_pnl / (pos.avg_price * pos.quantity)) * 100
                status = "🟢" if pos.unrealized_pnl > 0 else "🔴" if pos.unrealized_pnl < 0 else "⚪"
                logger.info(f"      {status} {symbol}: Rs.{pos.avg_price:.2f} -> Rs.{pos.current_price:.2f} ({pnl_pct:+.1f}%)")
    
    def show_final_results(self, cycles: int):
        """Show final test results"""
        summary = self.get_portfolio_summary()
        
        logger.info("🏁 FINAL TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"🔄 Total Cycles: {cycles}")
        logger.info(f"💰 Starting Capital: Rs.{self.capital:,}")
        logger.info(f"💹 Final P&L: Rs.{summary['total_pnl']:,.2f}")
        logger.info(f"📊 Return: {(summary['total_pnl'] / self.capital) * 100:+.2f}%")
        logger.info(f"🔄 Total Trades: {summary['trades_count']}")
        
        if summary['trades_count'] > 0:
            avg_trade = summary['total_pnl'] / summary['trades_count']
            logger.info(f"📈 Avg P&L per Trade: Rs.{avg_trade:.2f}")
        
        # Win rate calculation
        winning_trades = len([t for t in self.trades if hasattr(t, 'pnl') and getattr(t, 'pnl', 0) > 0])
        if summary['trades_count'] > 0:
            win_rate = (winning_trades / summary['trades_count']) * 100
            logger.info(f"🎯 Win Rate: {win_rate:.1f}%")
        
        logger.info("=" * 60)
        
        # Performance assessment
        if summary['total_pnl'] > 0:
            logger.info("🎉 TEST SUCCESSFUL - Bot made profit!")
        elif summary['total_pnl'] == 0:
            logger.info("⚖️ TEST NEUTRAL - No profit or loss")
        else:
            logger.info("⚠️ TEST LOSS - Bot needs optimization")

def main():
    """Main test function"""
    print("🎭 TRADING BOT SIMULATOR TEST")
    print("=" * 60)
    print("This will test your aggressive trading bot with synthetic market data")
    print("The bot will trade for 30 minutes with 10-second cycles")
    print("=" * 60)
    
    # Test the simulator first
    print("🧪 Testing Market Simulator...")
    for symbol in ["SBIN", "TCS", "INFY"]:
        price = market_simulator.get_live_price(symbol)
        print(f"   📊 {symbol}: Rs.{price:.2f}")
    
    print("\n🤖 Initializing Test Trading Bot...")
    
    # Create test bot
    test_bot = TestTradingBot(capital=50000)
    
    try:
        # Run test session
        test_bot.run_test_session(duration_minutes=30)
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    finally:
        test_bot.stop_trading()
        print("\n✅ Test completed!")

if __name__ == "__main__":
    main()