"""
Test different price sources to see what works
"""

import requests
import yfinance as yf
import time
from datetime import datetime

def test_yahoo_finance():
    """Test Yahoo Finance"""
    print("🧪 Testing Yahoo Finance...")
    try:
        ticker = yf.Ticker("SBIN.NS")
        data = ticker.history(period="1d", interval="1d")
        if not data.empty:
            price = data['Close'].iloc[-1]
            print(f"✅ Yahoo Finance: SBIN = Rs.{price:.2f}")
            return True
        else:
            print("❌ Yahoo Finance: No data")
            return False
    except Exception as e:
        print(f"❌ Yahoo Finance: {e}")
        return False

def test_simple_request():
    """Test simple HTTP request"""
    print("\n🧪 Testing Simple HTTP Request...")
    try:
        response = requests.get("https://httpbin.org/json", timeout=5)
        if response.status_code == 200:
            print("✅ HTTP requests working")
            return True
        else:
            print("❌ HTTP requests failed")
            return False
    except Exception as e:
        print(f"❌ HTTP requests: {e}")
        return False

def test_nse_connection():
    """Test NSE website connection"""
    print("\n🧪 Testing NSE Connection...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get("https://www.nseindia.com", headers=headers, timeout=10)
        print(f"NSE Response: {response.status_code}")
        if response.status_code == 200:
            print("✅ NSE website accessible")
            return True
        else:
            print("❌ NSE website not accessible")
            return False
    except Exception as e:
        print(f"❌ NSE connection: {e}")
        return False

def main():
    print("🔍 TESTING PRICE DATA SOURCES")
    print("=" * 40)
    
    results = []
    
    # Test each source
    results.append(("Yahoo Finance", test_yahoo_finance()))
    results.append(("HTTP Requests", test_simple_request()))
    results.append(("NSE Connection", test_nse_connection()))
    
    print("\n📊 RESULTS:")
    print("-" * 20)
    for name, success in results:
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{name:15}: {status}")
    
    working_count = sum(1 for _, success in results if success)
    print(f"\nWorking sources: {working_count}/{len(results)}")
    
    if working_count > 0:
        print("\n💡 RECOMMENDATION:")
        if results[0][1]:  # Yahoo Finance working
            print("✅ Use Yahoo Finance for price data")
        elif results[1][1]:  # HTTP working
            print("✅ Use alternative web scraping")
        else:
            print("⚠️  Limited connectivity - use cached data")
    else:
        print("\n❌ No working data sources found")

if __name__ == "__main__":
    main()