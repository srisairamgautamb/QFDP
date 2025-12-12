"""
Alpha Vantage Market Data Connector
====================================

Fetches real-time stock prices and historical data for quantum option pricing.

API Documentation: https://www.alphavantage.co/documentation/

Author: QFDP Research
"""

import os
import json
import numpy as np
import requests
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class StockData:
    """Real-time stock market data."""
    symbol: str
    current_price: float
    open_price: float
    high: float
    low: float
    volume: int
    previous_close: float
    change_percent: float
    timestamp: datetime


@dataclass
class HistoricalData:
    """Historical price data for volatility estimation."""
    symbol: str
    dates: np.ndarray
    prices: np.ndarray
    returns: np.ndarray
    volatility: float  # Annualized


class AlphaVantageConnector:
    """
    Fetches stock data from Alpha Vantage API.
    
    Features:
    - Real-time quotes
    - Historical daily prices
    - Volatility estimation (rolling window)
    - Rate limit handling (5 calls/min for free tier)
    """
    
    def __init__(self, api_key: str):
        """
        Initialize connector.
        
        Args:
            api_key: Alpha Vantage API key
        """
        import os
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.last_call_time = None
        self.rate_limit_delay = 12.0  # 5 calls/min = 12s between calls
        # On-disk cache to reduce rate-limit hits
        self.cache_dir = os.path.join('.cache', 'alphavantage')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _rate_limit(self):
        """Enforce rate limiting (5 API calls per minute)."""
        if self.last_call_time is not None:
            elapsed = (datetime.now() - self.last_call_time).total_seconds()
            if elapsed < self.rate_limit_delay:
                wait_time = self.rate_limit_delay - elapsed
                print(f"  ⏱ Rate limit: waiting {wait_time:.1f}s...")
                import time
                time.sleep(wait_time)
        self.last_call_time = datetime.now()
    
    def _make_request(self, params: Dict) -> Dict:
        """
        Make API request with error handling.
        
        Args:
            params: Query parameters
            
        Returns:
            JSON response
            
        Raises:
            RuntimeError: If API call fails
        """
        self._rate_limit()
        
        params['apikey'] = self.api_key
        response = requests.get(self.base_url, params=params)
        
        if response.status_code != 200:
            raise RuntimeError(f"API call failed: HTTP {response.status_code}")
        
        data = response.json()
        
        # Check for API error messages
        if "Error Message" in data:
            raise RuntimeError(f"API error: {data['Error Message']}")
        if "Note" in data:
            raise RuntimeError(f"API rate limit exceeded: {data['Note']}")
        
        return data
    
    def _cache_path(self, function: str, symbol: str) -> str:
        import os
        fname = f"{function}_{symbol}.json"
        return os.path.join(self.cache_dir, fname)

    def _read_cache(self, function: str, symbol: str, ttl_seconds: int) -> Optional[Dict]:
        import os, json, time
        path = self._cache_path(function, symbol)
        if not os.path.exists(path):
            return None
        age = time.time() - os.path.getmtime(path)
        if age > ttl_seconds:
            return None
        with open(path, 'r') as f:
            return json.load(f)

    def _write_cache(self, function: str, symbol: str, data: Dict) -> None:
        import json
        path = self._cache_path(function, symbol)
        with open(path, 'w') as f:
            json.dump(data, f)

    def get_quote(self, symbol: str) -> StockData:
        """
        Fetch real-time stock quote.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'TSLA')
            
        Returns:
            StockData with current price and metadata
            
        Example:
            >>> connector = AlphaVantageConnector(api_key)
            >>> data = connector.get_quote('AAPL')
            >>> print(f"AAPL: ${data.current_price:.2f}")
        """
        print(f"Fetching quote for {symbol}...")
        
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol
        }
        # Try cache (1 minute TTL)
        cached = self._read_cache('GLOBAL_QUOTE', symbol, ttl_seconds=60)
        if cached is None:
            response = self._make_request(params)
            self._write_cache('GLOBAL_QUOTE', symbol, response)
        else:
            response = cached
        quote = response.get('Global Quote', {})
        
        if not quote:
            raise RuntimeError(f"No data returned for {symbol}")
        
        return StockData(
            symbol=symbol,
            current_price=float(quote.get('05. price', 0)),
            open_price=float(quote.get('02. open', 0)),
            high=float(quote.get('03. high', 0)),
            low=float(quote.get('04. low', 0)),
            volume=int(quote.get('06. volume', 0)),
            previous_close=float(quote.get('08. previous close', 0)),
            change_percent=float(quote.get('10. change percent', '0%').rstrip('%')),
            timestamp=datetime.now()
        )
    
    def get_historical_daily(
        self,
        symbol: str,
        days: int = 252
    ) -> HistoricalData:
        """
        Fetch historical daily prices.
        
        Args:
            symbol: Stock ticker
            days: Number of trading days (default: 252 = 1 year)
            
        Returns:
            HistoricalData with prices and computed volatility
            
        Example:
            >>> hist = connector.get_historical_daily('AAPL', days=252)
            >>> print(f"Volatility: {hist.volatility*100:.1f}%")
        """
        print(f"Fetching {days} days of historical data for {symbol}...")
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'full'  # Get all available data
        }
        # Try cache (1 day TTL)
        cached = self._read_cache('TIME_SERIES_DAILY', symbol, ttl_seconds=24*3600)
        if cached is None:
            response = self._make_request(params)
            self._write_cache('TIME_SERIES_DAILY', symbol, response)
        else:
            response = cached
        time_series = response.get('Time Series (Daily)', {})
        
        if not time_series:
            raise RuntimeError(f"No historical data for {symbol}")
        
        # Parse data (sorted by date descending)
        dates = []
        prices = []
        
        for date_str, values in sorted(time_series.items(), reverse=True):
            dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
            prices.append(float(values['4. close']))  # Closing price
            
            if len(dates) >= days:
                break
        
        dates = np.array(dates[::-1])  # Oldest to newest
        prices = np.array(prices[::-1])
        
        # Compute log returns
        returns = np.diff(np.log(prices))
        
        # Annualized volatility (252 trading days/year)
        volatility = np.std(returns) * np.sqrt(252)
        
        return HistoricalData(
            symbol=symbol,
            dates=dates,
            prices=prices,
            returns=returns,
            volatility=volatility
        )
    
    def estimate_parameters(
        self,
        symbol: str,
        risk_free_rate: float = 0.03
    ) -> Tuple[float, float, float]:
        """
        Estimate (S0, r, σ) for quantum option pricing.
        
        Args:
            symbol: Stock ticker
            risk_free_rate: Risk-free rate (default: 3%)
            
        Returns:
            (S0, r, sigma) tuple:
            - S0: Current spot price
            - r: Risk-free rate (input)
            - sigma: Annualized volatility (historical)
            
        Example:
            >>> S0, r, sigma = connector.estimate_parameters('AAPL')
            >>> print(f"Parameters: S0=${S0:.2f}, σ={sigma*100:.1f}%")
        """
        # Get current price
        quote = self.get_quote(symbol)
        S0 = quote.current_price
        
        # Get historical volatility
        hist = self.get_historical_daily(symbol, days=252)
        sigma = hist.volatility
        
        return S0, risk_free_rate, sigma


def quick_test(api_key: str, symbol: str = 'AAPL'):
    """
    Quick test of Alpha Vantage connector.
    
    Args:
        api_key: Alpha Vantage API key
        symbol: Stock ticker to test
    """
    print(f"\n{'='*60}")
    print(f"Alpha Vantage Connector Test: {symbol}")
    print(f"{'='*60}\n")
    
    connector = AlphaVantageConnector(api_key)
    
    # Test 1: Real-time quote
    print("[1/2] Fetching real-time quote...")
    quote = connector.get_quote(symbol)
    print(f"  ✓ {quote.symbol}: ${quote.current_price:.2f}")
    print(f"    Open: ${quote.open_price:.2f}, High: ${quote.high:.2f}, Low: ${quote.low:.2f}")
    print(f"    Change: {quote.change_percent:+.2f}%")
    
    # Test 2: Historical volatility
    print(f"\n[2/2] Estimating volatility (252 days)...")
    S0, r, sigma = connector.estimate_parameters(symbol)
    print(f"  ✓ Spot price (S0): ${S0:.2f}")
    print(f"  ✓ Volatility (σ): {sigma*100:.1f}%")
    print(f"  ✓ Risk-free rate (r): {r*100:.1f}%")
    
    print(f"\n{'='*60}")
    print(f"Test complete! Ready for quantum pricing.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test with API key from environment variable
    api_key = os.getenv('ALPHAVANTAGE_API_KEY', 'demo')
    
    if api_key == 'demo':
        print("⚠ Using demo API key. Set ALPHAVANTAGE_API_KEY environment variable.")
    
    quick_test(api_key, symbol='IBM')  # Use IBM for demo (free tier)
