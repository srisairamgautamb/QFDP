"""
Data Pipeline for QML-IQFT
===========================

Stock data collection and preprocessing for QML-enhanced pricing.

Provides:
- Download 5-year daily close prices from yfinance
- Compute log returns
- Correlation analysis

Author: QFDP Research Team
Date: January 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Tuple, Optional, List

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# Default tickers (tech sector for high correlation)
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']


@dataclass
class StockData:
    """
    Result from stock data collection.
    
    Attributes
    ----------
    prices : pd.DataFrame
        Daily close prices (T × N)
    returns : pd.DataFrame
        Log returns (T-1 × N)
    correlation_matrix : np.ndarray
        Correlation matrix (N × N)
    covariance_matrix : np.ndarray
        Covariance matrix (N × N)
    tickers : List[str]
        Ticker symbols
    start_date : str
        Data start date
    end_date : str
        Data end date
    """
    prices: pd.DataFrame
    returns: pd.DataFrame
    correlation_matrix: np.ndarray
    covariance_matrix: np.ndarray
    tickers: List[str]
    start_date: str
    end_date: str


def collect_stock_data(
    tickers: Optional[List[str]] = None,
    years: int = 5,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download historical stock prices from Yahoo Finance.
    
    Parameters
    ----------
    tickers : List[str], optional
        Stock ticker symbols. Default: AAPL, MSFT, GOOGL, NVDA, META
    years : int
        Number of years of historical data
    end_date : str, optional
        End date in YYYY-MM-DD format. Default: today
        
    Returns
    -------
    prices : pd.DataFrame
        Daily close prices (T × N)
    returns : pd.DataFrame
        Log returns (T-1 × N)
        
    Raises
    ------
    ImportError
        If yfinance is not installed
        
    Examples
    --------
    >>> prices, returns = collect_stock_data()
    >>> print(f"Downloaded {len(prices)} days of data")
    >>> print(f"Computed {len(returns)} return observations")
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError(
            "yfinance not installed. Run: pip install yfinance"
        )
    
    if tickers is None:
        tickers = DEFAULT_TICKERS
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    start_date = (
        datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=years*365)
    ).strftime('%Y-%m-%d')
    
    print(f"📊 Downloading stock data...")
    print(f"   Tickers: {tickers}")
    print(f"   Period: {start_date} to {end_date}")
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Extract close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = tickers
    
    # Handle any missing tickers
    prices = prices.dropna(axis=1, how='all')
    
    # Compute log returns
    returns = compute_log_returns(prices)
    
    print(f"✅ Downloaded {len(prices)} days of data")
    print(f"✅ Computed {len(returns)} return observations")
    
    return prices, returns


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from price data.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Daily close prices (T × N)
        
    Returns
    -------
    returns : pd.DataFrame
        Log returns: r_t = log(P_t / P_{t-1})
        
    Notes
    -----
    Log returns are preferred for:
    - Additivity across time
    - Better normality approximation
    - Symmetric treatment of gains/losses
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def analyze_correlations(
    returns: pd.DataFrame,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute and analyze correlation structure.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Log returns (T × N)
    verbose : bool
        Print analysis summary
        
    Returns
    -------
    corr_matrix : np.ndarray
        Correlation matrix (N × N)
    cov_matrix : np.ndarray
        Covariance matrix (N × N)
        
    Examples
    --------
    >>> corr, cov = analyze_correlations(returns)
    >>> print(f"Mean correlation: {mean_corr:.3f}")
    """
    corr_matrix = returns.corr().values
    cov_matrix = returns.cov().values
    
    if verbose:
        # Extract upper triangular (excluding diagonal)
        N = corr_matrix.shape[0]
        upper_tri = corr_matrix[np.triu_indices(N, k=1)]
        
        print("\n📊 Correlation Analysis:")
        print(f"   Number of assets: {N}")
        print(f"   Mean pairwise correlation: {upper_tri.mean():.3f}")
        print(f"   Min correlation: {upper_tri.min():.3f}")
        print(f"   Max correlation: {upper_tri.max():.3f}")
        print(f"   Std correlation: {upper_tri.std():.3f}")
        
        # Eigenvalue analysis
        eigenvalues = np.linalg.eigvalsh(corr_matrix)[::-1]
        variance_explained = eigenvalues / eigenvalues.sum()
        cumulative_var = np.cumsum(variance_explained)
        
        print(f"\n   Eigenvalue spectrum:")
        for i, (ev, ve, cv) in enumerate(zip(
            eigenvalues[:5], variance_explained[:5], cumulative_var[:5]
        )):
            print(f"     λ_{i+1} = {ev:.4f} ({ve*100:.1f}%, cumulative: {cv*100:.1f}%)")
    
    return corr_matrix, cov_matrix


def prepare_full_dataset(
    tickers: Optional[List[str]] = None,
    years: int = 5,
    save_to_disk: bool = False,
    data_dir: str = 'data'
) -> StockData:
    """
    Complete data preparation pipeline.
    
    Parameters
    ----------
    tickers : List[str], optional
        Stock ticker symbols
    years : int
        Number of years of historical data
    save_to_disk : bool
        Save data to CSV files
    data_dir : str
        Directory for saved files
        
    Returns
    -------
    StockData
        Complete dataset with prices, returns, and statistics
        
    Examples
    --------
    >>> data = prepare_full_dataset()
    >>> print(f"Correlation shape: {data.correlation_matrix.shape}")
    """
    import os
    
    if tickers is None:
        tickers = DEFAULT_TICKERS
    
    # Collect data
    prices, returns = collect_stock_data(tickers, years)
    
    # Analyze correlations
    corr_matrix, cov_matrix = analyze_correlations(returns)
    
    # Create result
    result = StockData(
        prices=prices,
        returns=returns,
        correlation_matrix=corr_matrix,
        covariance_matrix=cov_matrix,
        tickers=list(prices.columns),
        start_date=prices.index[0].strftime('%Y-%m-%d'),
        end_date=prices.index[-1].strftime('%Y-%m-%d')
    )
    
    # Save to disk if requested
    if save_to_disk:
        os.makedirs(data_dir, exist_ok=True)
        prices.to_csv(os.path.join(data_dir, 'raw_prices.csv'))
        returns.to_csv(os.path.join(data_dir, 'log_returns.csv'))
        np.save(os.path.join(data_dir, 'correlation_matrix.npy'), corr_matrix)
        print(f"✅ Data saved to {data_dir}/")
    
    return result


if __name__ == '__main__':
    # Test the pipeline
    data = prepare_full_dataset(save_to_disk=False)
    print(f"\n✅ Data pipeline test complete")
    print(f"   Shape: {data.returns.shape}")
