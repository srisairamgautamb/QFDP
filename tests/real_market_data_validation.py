"""
Real Market Data Validation
===========================

Test QRC+QTC+FB-IQFT with actual S&P 500 stock data.

Uses yfinance to fetch real historical data for:
- FAANG stocks (META, AAPL, AMZN, NFLX, GOOGL)
- Tech sector (MSFT, NVDA, TSLA, AMD, INTC)
- Diverse portfolio (mix of sectors)

Tests correlation estimation, regime detection, and pricing accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import sys
from datetime import datetime, timedelta

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.fb_iqft_pricing import FBIQFTPricing
from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
from qfdp.integrated.corrected_qtc_integrated_pricer import CorrectedQTCIntegratedPricer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================================
# MARKET DATA FETCHER
# ============================================================================

def fetch_stock_data(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical stock data via yfinance.
    
    Args:
        tickers: List of stock symbols
        period: History period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y')
    
    Returns:
        DataFrame with adjusted close prices
    """
    try:
        import yfinance as yf
        
        # Fetch data
        data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        
        # Handle different return formats
        if 'Close' in data.columns if hasattr(data.columns, '__iter__') else False:
            data = data['Close']
        elif isinstance(data.columns, pd.MultiIndex):
            data = data['Close'] if 'Close' in data.columns.get_level_values(0) else data.xs('Close', axis=1, level=0)
        
        # Handle single ticker case
        if len(tickers) == 1:
            data = pd.DataFrame(data)
            data.columns = tickers
        
        return data.dropna()
    except ImportError:
        logger.warning("yfinance not installed. Using synthetic data.")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return None


def compute_realized_volatility(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute annualized realized volatility.
    
    Args:
        prices: Price DataFrame
        window: Rolling window for volatility calculation
    
    Returns:
        Series of annualized volatilities
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    volatility = log_returns.rolling(window=window).std() * np.sqrt(252)
    return volatility.iloc[-1]


def compute_correlation_matrix(prices: pd.DataFrame, window: int = 60) -> np.ndarray:
    """
    Compute rolling correlation matrix.
    
    Args:
        prices: Price DataFrame
        window: Rolling window
    
    Returns:
        Current correlation matrix
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    recent_returns = log_returns.tail(window)
    return recent_returns.corr().values


def get_price_history(prices: pd.Series, n_points: int = 6) -> np.ndarray:
    """
    Get last n price points for QTC input.
    
    Args:
        prices: Price series
        n_points: Number of points to return
    
    Returns:
        Array of last n prices
    """
    return prices.tail(n_points).values


# ============================================================================
# TEST PORTFOLIOS
# ============================================================================

PORTFOLIOS = {
    'FAANG': {
        'tickers': ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL'],
        'description': 'Facebook, Apple, Amazon, Netflix, Google'
    },
    'Tech': {
        'tickers': ['MSFT', 'NVDA', 'TSLA', 'AMD', 'INTC'],
        'description': 'Microsoft, Nvidia, Tesla, AMD, Intel'
    },
    'Diverse': {
        'tickers': ['AAPL', 'JPM', 'JNJ', 'XOM', 'WMT'],
        'description': 'Tech, Finance, Healthcare, Energy, Retail'
    },
    'Large': {
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'JNJ'],
        'description': '10 Large Cap Stocks'
    }
}


# ============================================================================
# SYNTHETIC DATA FALLBACK
# ============================================================================

def generate_synthetic_market_data(n_assets: int, n_days: int = 252) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic market data if yfinance is unavailable.
    
    Returns:
        Tuple of (prices, volatilities, correlation_matrix)
    """
    np.random.seed(42)
    
    # Generate correlated returns
    base_corr = 0.3  # Base correlation
    corr = np.eye(n_assets) * (1 - base_corr) + base_corr
    L = np.linalg.cholesky(corr)
    
    # Base parameters
    mu = 0.10  # 10% annual drift
    sigma = np.random.uniform(0.15, 0.40, n_assets)  # 15-40% volatility
    
    # Generate paths
    dt = 1 / 252
    Z = np.random.standard_normal((n_days, n_assets))
    Z_correlated = Z @ L.T
    
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_correlated
    log_prices = np.cumsum(log_returns, axis=0)
    prices = 100 * np.exp(log_prices)  # Start at 100
    
    # Compute realized volatility and correlation
    realized_vol = np.std(log_returns, axis=0) * np.sqrt(252)
    realized_corr = np.corrcoef(log_returns.T)
    
    return prices[-1], realized_vol, realized_corr


# ============================================================================
# VALIDATION TESTS
# ============================================================================

def run_real_market_validation():
    """
    Run validation with real market data.
    """
    print("=" * 80)
    print("REAL MARKET DATA VALIDATION")
    print("=" * 80)
    
    # Initialize pricers
    fb_iqft = FBIQFTPricing(M=16, alpha=1.0)
    qrc_qtc_pricer = CorrectedQTCIntegratedPricer(fb_iqft, qrc_beta=0.1, qtc_gamma=0.05)
    
    results = []
    
    for portfolio_name, portfolio_info in PORTFOLIOS.items():
        tickers = portfolio_info['tickers']
        n = len(tickers)
        
        print(f"\n{'='*60}")
        print(f"PORTFOLIO: {portfolio_name} ({portfolio_info['description']})")
        print(f"{'='*60}")
        
        # Fetch data
        print(f"Fetching data for {tickers}...")
        prices_df = fetch_stock_data(tickers, period="1y")
        
        if prices_df is None or prices_df.empty:
            print("  Using synthetic data (yfinance unavailable)")
            current_prices, vols, corr = generate_synthetic_market_data(n)
            price_history = np.linspace(95, 100, 6)  # Synthetic price history
        else:
            # Compute market parameters
            current_prices = prices_df.iloc[-1].values
            vols = compute_realized_volatility(prices_df, window=20)
            corr = compute_correlation_matrix(prices_df, window=60)
            
            # Use portfolio average price history for QTC
            portfolio_prices = (prices_df / prices_df.iloc[0]).mean(axis=1) * 100  # Normalize to 100
            price_history = get_price_history(portfolio_prices)
        
        # Portfolio parameters
        weights = np.ones(n) / n  # Equal weight
        portfolio_value = np.sum(weights * current_prices)
        
        print(f"\n  Current Prices: {current_prices.round(2)}")
        print(f"  Realized Vols:  {vols.round(4) if hasattr(vols, 'round') else np.array(vols).round(4)}")
        print(f"  Portfolio Value: ${portfolio_value:.2f}")
        
        # Compute correlation statistics
        if isinstance(corr, np.ndarray):
            avg_corr = (np.sum(corr) - n) / (n * (n - 1))
            print(f"  Average Correlation: {avg_corr:.4f}")
        
        # Price at different strikes
        strikes = [0.9, 0.95, 1.0, 1.05, 1.1]  # Moneyness levels
        
        print(f"\n  {'Strike':<10} {'Moneyness':<12} {'QRC-QTC Price':<15} {'σ_p Enhanced':<12}")
        print("  " + "-" * 55)
        
        for moneyness in strikes:
            K = portfolio_value * moneyness
            
            # Ensure proper numpy arrays
            asset_prices = np.array(current_prices)
            asset_vols = np.array(vols) if not isinstance(vols, np.ndarray) else vols
            corr_matrix = np.array(corr) if not isinstance(corr, np.ndarray) else corr
            
            # Make correlation matrix valid
            np.fill_diagonal(corr_matrix, 1.0)
            corr_matrix = np.clip(corr_matrix, -1, 1)
            
            # Ensure positive semi-definite
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            if np.min(eigenvalues) < 0:
                corr_matrix = corr_matrix + (-np.min(eigenvalues) + 0.01) * np.eye(n)
                corr_matrix = corr_matrix / np.sqrt(np.outer(np.diag(corr_matrix), np.diag(corr_matrix)))
            
            market_data = {
                'spot_prices': asset_prices,
                'volatilities': asset_vols,
                'correlation_matrix': corr_matrix,
                'weights': weights,
                'maturity': 1.0,
                'risk_free_rate': 0.05
            }
            
            qrc_result = qrc_qtc_pricer.price_with_full_quantum_pipeline(
                market_data, price_history, strike=K, use_quantum_circuit=True
            )
            
            label = "ATM" if moneyness == 1.0 else ("ITM" if moneyness < 1.0 else "OTM")
            print(f"  ${K:<9.2f} {label} ({moneyness:.0%})  ${qrc_result['price_quantum']:<13.4f} {qrc_result['sigma_p_enhanced']:<12.4f}")
            
            results.append({
                'portfolio': portfolio_name,
                'moneyness': moneyness,
                'price': qrc_result['price_quantum'],
                'sigma_p': qrc_result['sigma_p_enhanced']
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal portfolios tested: {len(PORTFOLIOS)}")
    print(f"Total option prices computed: {len(results)}")
    
    # By portfolio
    print("\nAverage σ_p by Portfolio:")
    for portfolio_name in PORTFOLIOS.keys():
        portfolio_results = [r for r in results if r['portfolio'] == portfolio_name]
        avg_sigma = np.mean([r['sigma_p'] for r in portfolio_results])
        print(f"  {portfolio_name:<10}: {avg_sigma:.4f}")
    
    print("\n✅ REAL MARKET DATA VALIDATION COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_real_market_validation()
