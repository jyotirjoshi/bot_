@echo off
echo ========================================
echo    TRADING BOT SIMULATOR TEST
echo ========================================
echo.

cd /d "%~dp0"

echo Testing Market Simulator...
python market_simulator.py
echo.

echo Starting Trading Bot Test...
echo This will run for 30 minutes with synthetic data
echo Press Ctrl+C to stop early
echo.
pause

python test_bot.py

echo.
echo Test completed! Check test_trading_bot.log for details
pause