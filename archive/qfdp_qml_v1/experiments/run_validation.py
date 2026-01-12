"""
QAE vs PCA Validation Experiment
================================
Validates quantum autoencoder against classical PCA for factor extraction.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp_qml.quantum_autoencoder import QuantumFactorAutoencoder, train_qae
from qfdp_qml.hybrid_integration import QAE_FB_IQFT_Pricer


def run_3asset_experiment():
    """3-asset portfolio: compare QAE vs PCA pricing."""
    print("=" * 60)
    print("EXPERIMENT: 3-Asset Portfolio (QAE vs PCA)")
    print("=" * 60)
    
    returns = pd.read_csv('data/qml/log_returns.csv', index_col=0)
    prices = pd.read_csv('data/qml/raw_prices.csv', index_col=0)
    
    tickers = ['AAPL', 'GOOGL', 'META']
    returns_3 = returns[tickers].values
    prices_3 = prices[tickers].iloc[-1].values
    
    print(f"\nAssets: {tickers}")
    print(f"Latest prices: ${prices_3.round(2)}")
    
    pricer = QAE_FB_IQFT_Pricer(n_factors=2, n_layers=2)
    print("\nTraining QAE...")
    pricer.train(returns_3, max_iter=50)
    
    weights = np.ones(3) / 3
    sigma = returns[tickers].std().values * np.sqrt(252)
    corr = returns[tickers].corr().values
    K = np.sum(prices_3 * weights)
    
    print(f"\nPricing ATM call (K=${K:.2f})...")
    result = pricer.price_option(
        S0=prices_3,
        sigma=sigma,
        corr=corr,
        weights=weights,
        K=K,
        T=1.0,
        r=0.05,
        n_mc=100000
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Monte Carlo (reference): ${result.price_mc:.4f}")
    print(f"PCA-based pricing:       ${result.price_pca:.4f} (error: {result.error_pca:.2f}%)")
    print(f"QAE-based pricing:       ${result.price_qae:.4f} (error: {result.error_qae:.2f}%)")
    print(f"PCA explained variance:  {result.pca_explained:.1%}")
    print(f"QAE factors:             {result.qae_factors.round(4)}")
    print(f"PCA factors:             {result.pca_factors.round(4)}")
    
    improvement = result.error_pca - result.error_qae
    if improvement > 0:
        print(f"\n✅ QAE beats PCA by {improvement:.2f}%")
    else:
        print(f"\n⚠️ PCA beats QAE by {-improvement:.2f}%")
    
    return result


def run_5asset_experiment():
    """5-asset portfolio: full validation."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: 5-Asset Portfolio (Full System)")
    print("=" * 60)
    
    returns = pd.read_csv('data/qml/log_returns.csv', index_col=0)
    prices = pd.read_csv('data/qml/raw_prices.csv', index_col=0)
    
    tickers = list(returns.columns)
    returns_5 = returns.values
    prices_5 = prices.iloc[-1].values
    
    print(f"\nAssets: {tickers}")
    print(f"Latest prices: ${prices_5.round(2)}")
    
    pricer = QAE_FB_IQFT_Pricer(n_factors=3, n_layers=2)
    print("\nTraining QAE...")
    pricer.train(returns_5, max_iter=50)
    
    weights = np.ones(5) / 5
    sigma = returns.std().values * np.sqrt(252)
    corr = returns.corr().values
    K = np.sum(prices_5 * weights)
    
    print(f"\nPricing ATM call (K=${K:.2f})...")
    result = pricer.price_option(
        S0=prices_5,
        sigma=sigma,
        corr=corr,
        weights=weights,
        K=K,
        T=1.0,
        r=0.05,
        n_mc=100000
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Monte Carlo (reference): ${result.price_mc:.4f}")
    print(f"PCA-based pricing:       ${result.price_pca:.4f} (error: {result.error_pca:.2f}%)")
    print(f"QAE-based pricing:       ${result.price_qae:.4f} (error: {result.error_qae:.2f}%)")
    print(f"PCA explained variance:  {result.pca_explained:.1%}")
    
    improvement = result.error_pca - result.error_qae
    if improvement > 0:
        print(f"\n✅ QAE beats PCA by {improvement:.2f}%")
    else:
        print(f"\n⚠️ PCA beats QAE by {-improvement:.2f}%")
    
    return result


if __name__ == '__main__':
    result_3 = run_3asset_experiment()
    result_5 = run_5asset_experiment()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"3-Asset: QAE={result_3.error_qae:.2f}% vs PCA={result_3.error_pca:.2f}%")
    print(f"5-Asset: QAE={result_5.error_qae:.2f}% vs PCA={result_5.error_pca:.2f}%")
