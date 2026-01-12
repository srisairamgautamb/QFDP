"""
Quantum Advantage Window Discovery
===================================
Systematically tests 5 scenarios where quantum methods might outperform classical:
1. Extreme strike prices (far OTM / deep ITM)
2. High volatility regimes
3. Short-dated options
4. Perfect correlation (degenerate cases)
5. Maximum dimensionality stress test
"""

import numpy as np
import pandas as pd
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WindowResult:
    """Result from testing one parameter configuration."""
    window: str
    param_name: str
    param_value: float
    error_classical: float
    error_quantum: float
    advantage: float  # positive = quantum wins
    time_classical: float
    time_quantum: float


def monte_carlo_reference(
    asset_prices: np.ndarray,
    asset_volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    portfolio_weights: np.ndarray,
    K: float,
    T: float,
    r: float = 0.05,
    n_sims: int = 100000
) -> Tuple[float, float]:
    """High-accuracy Monte Carlo reference price."""
    np.random.seed(42)
    N = len(asset_prices)
    
    cov = np.outer(asset_volatilities, asset_volatilities) * correlation_matrix
    
    # Handle near-singular covariance matrices
    try:
        L = np.linalg.cholesky(cov + np.eye(N) * 1e-10)
    except np.linalg.LinAlgError:
        # Use eigendecomposition fallback
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        L = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    Z = np.random.standard_normal((n_sims, N))
    Z_corr = Z @ L.T
    
    drift = (r - 0.5 * asset_volatilities**2) * T
    diffusion = np.sqrt(T) * Z_corr
    ST = asset_prices * np.exp(drift + diffusion)
    
    portfolio_T = ST @ portfolio_weights
    payoffs = np.maximum(portfolio_T - K, 0)
    
    price = np.exp(-r * T) * np.mean(payoffs)
    stderr = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    
    return price, stderr


def black_scholes_portfolio(
    asset_prices: np.ndarray,
    asset_volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    portfolio_weights: np.ndarray,
    K: float,
    T: float,
    r: float = 0.05
) -> float:
    """Classical Black-Scholes with portfolio volatility."""
    S0 = np.sum(asset_prices * portfolio_weights)
    cov = np.outer(asset_volatilities, asset_volatilities) * correlation_matrix
    sigma_p = np.sqrt(portfolio_weights @ cov @ portfolio_weights)
    
    if T < 1e-6:  # Very short maturity
        return max(S0 - K, 0)
    
    d1 = (np.log(S0/K) + (r + 0.5*sigma_p**2)*T) / (sigma_p*np.sqrt(T))
    d2 = d1 - sigma_p*np.sqrt(T)
    
    price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return float(price)


def price_with_fb_iqft(
    asset_prices: np.ndarray,
    asset_volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    portfolio_weights: np.ndarray,
    K: float,
    T: float,
    r: float = 0.05
) -> Tuple[float, float]:
    """Price using FB-IQFT (classical baseline method)."""
    from qfdp.unified import FBIQFTPricing
    
    start = time.time()
    pricer = FBIQFTPricing(M=16, alpha=1.0, num_shots=8192)
    
    try:
        result = pricer.price_option(
            asset_prices=asset_prices,
            asset_volatilities=asset_volatilities,
            correlation_matrix=correlation_matrix,
            portfolio_weights=portfolio_weights,
            K=K, T=T, r=r,
            backend=None
        )
        price = result['price_quantum']
        elapsed = time.time() - start
        return price, elapsed
    except Exception as e:
        # Fallback to BS
        price = black_scholes_portfolio(
            asset_prices, asset_volatilities, correlation_matrix,
            portfolio_weights, K, T, r
        )
        return price, time.time() - start


def price_with_qae(
    asset_prices: np.ndarray,
    asset_volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    portfolio_weights: np.ndarray,
    K: float,
    T: float,
    r: float = 0.05,
    n_factors: int = 3
) -> Tuple[float, float]:
    """Price using Quantum Autoencoder + FB-IQFT."""
    from qfdp_qml.hybrid_integration import QAE_FB_IQFT_Pricer
    
    start = time.time()
    N = len(asset_prices)
    
    # Generate synthetic returns for training
    np.random.seed(42)
    cov = np.outer(asset_volatilities, asset_volatilities) * correlation_matrix
    try:
        L = np.linalg.cholesky(cov + np.eye(N) * 1e-10)
    except:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        L = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    returns = np.random.randn(50, N) @ L.T * np.sqrt(1/252)
    
    try:
        pricer = QAE_FB_IQFT_Pricer(n_factors=min(n_factors, N-1), n_layers=2)
        pricer.train(returns, max_iter=20)
        
        result = pricer.price_option(
            S0=asset_prices,
            sigma=asset_volatilities,
            corr=correlation_matrix,
            weights=portfolio_weights,
            K=K, T=T, r=r
        )
        price = result.price_qae
        elapsed = time.time() - start
        return price, elapsed
    except Exception as e:
        # Fallback
        price = black_scholes_portfolio(
            asset_prices, asset_volatilities, correlation_matrix,
            portfolio_weights, K, T, r
        )
        return price, time.time() - start


def generate_correlation_matrix(n: int, base_corr: float = 0.3) -> np.ndarray:
    """Generate valid correlation matrix."""
    corr = np.full((n, n), base_corr)
    np.fill_diagonal(corr, 1.0)
    
    # Ensure positive definiteness
    eigenvalues = np.linalg.eigvalsh(corr)
    if np.min(eigenvalues) < 0:
        corr += np.eye(n) * (abs(np.min(eigenvalues)) + 0.01)
        # Renormalize
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    
    return corr


def test_window_1_extreme_strikes() -> List[WindowResult]:
    """Window 1: Test extreme strike prices."""
    print("\n🪟 WINDOW 1: Extreme Strike Prices")
    print("-" * 70)
    
    results = []
    S0 = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    sigma = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    corr = generate_correlation_matrix(5, 0.3)
    weights = np.ones(5) / 5
    T = 1.0
    r = 0.05
    
    moneyness_levels = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    portfolio_value = np.sum(S0 * weights)
    
    for moneyness in moneyness_levels:
        K = portfolio_value * moneyness
        
        # Monte Carlo reference
        mc_price, _ = monte_carlo_reference(S0, sigma, corr, weights, K, T, r)
        
        # Classical (FB-IQFT)
        price_classical, time_classical = price_with_fb_iqft(
            S0, sigma, corr, weights, K, T, r
        )
        
        # Quantum (QAE)
        price_quantum, time_quantum = price_with_qae(
            S0, sigma, corr, weights, K, T, r
        )
        
        # Compute errors relative to MC
        if mc_price > 0.01:
            error_classical = abs(price_classical - mc_price) / mc_price * 100
            error_quantum = abs(price_quantum - mc_price) / mc_price * 100
        else:
            error_classical = abs(price_classical - mc_price) * 100
            error_quantum = abs(price_quantum - mc_price) * 100
        
        advantage = error_classical - error_quantum
        
        status = "✅ QUANTUM WINS" if advantage > 0.1 else "⚪"
        print(f"  Moneyness {moneyness:.1f}x (K=${K:.0f}): "
              f"Classical={error_classical:.2f}%, Quantum={error_quantum:.2f}% {status}")
        
        results.append(WindowResult(
            window="Extreme Strikes",
            param_name="moneyness",
            param_value=moneyness,
            error_classical=error_classical,
            error_quantum=error_quantum,
            advantage=advantage,
            time_classical=time_classical,
            time_quantum=time_quantum
        ))
    
    return results


def test_window_2_high_volatility() -> List[WindowResult]:
    """Window 2: Test high volatility regimes."""
    print("\n🪟 WINDOW 2: High Volatility Regime")
    print("-" * 70)
    
    results = []
    S0 = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    corr = generate_correlation_matrix(5, 0.3)
    weights = np.ones(5) / 5
    K = 100.0
    T = 1.0
    r = 0.05
    
    volatility_levels = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    
    for vol in volatility_levels:
        sigma = np.ones(5) * vol
        
        # Monte Carlo reference
        mc_price, _ = monte_carlo_reference(S0, sigma, corr, weights, K, T, r)
        
        # Classical
        price_classical, time_classical = price_with_fb_iqft(
            S0, sigma, corr, weights, K, T, r
        )
        
        # Quantum
        price_quantum, time_quantum = price_with_qae(
            S0, sigma, corr, weights, K, T, r
        )
        
        if mc_price > 0.01:
            error_classical = abs(price_classical - mc_price) / mc_price * 100
            error_quantum = abs(price_quantum - mc_price) / mc_price * 100
        else:
            error_classical = abs(price_classical - mc_price) * 100
            error_quantum = abs(price_quantum - mc_price) * 100
        
        advantage = error_classical - error_quantum
        
        status = "✅ QUANTUM WINS" if advantage > 0.1 else "⚪"
        print(f"  σ = {vol*100:.0f}%: Classical={error_classical:.2f}%, "
              f"Quantum={error_quantum:.2f}% {status}")
        
        results.append(WindowResult(
            window="High Volatility",
            param_name="volatility",
            param_value=vol,
            error_classical=error_classical,
            error_quantum=error_quantum,
            advantage=advantage,
            time_classical=time_classical,
            time_quantum=time_quantum
        ))
    
    return results


def test_window_3_short_dated() -> List[WindowResult]:
    """Window 3: Test short-dated options."""
    print("\n🪟 WINDOW 3: Short-Dated Options")
    print("-" * 70)
    
    results = []
    S0 = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    sigma = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    corr = generate_correlation_matrix(5, 0.3)
    weights = np.ones(5) / 5
    K = 100.0
    r = 0.05
    
    # From 1 day to 2 years
    maturities = [1/365, 5/365, 10/365, 1/12, 1/4, 0.5, 1.0, 2.0]
    
    for T in maturities:
        # Monte Carlo reference
        mc_price, _ = monte_carlo_reference(S0, sigma, corr, weights, K, T, r)
        
        # Classical
        price_classical, time_classical = price_with_fb_iqft(
            S0, sigma, corr, weights, K, T, r
        )
        
        # Quantum
        price_quantum, time_quantum = price_with_qae(
            S0, sigma, corr, weights, K, T, r
        )
        
        if mc_price > 0.01:
            error_classical = abs(price_classical - mc_price) / mc_price * 100
            error_quantum = abs(price_quantum - mc_price) / mc_price * 100
        else:
            error_classical = abs(price_classical - mc_price) * 100
            error_quantum = abs(price_quantum - mc_price) * 100
        
        advantage = error_classical - error_quantum
        
        days = T * 365
        status = "✅ QUANTUM WINS" if advantage > 0.1 else "⚪"
        print(f"  T = {days:.0f} days: Classical={error_classical:.2f}%, "
              f"Quantum={error_quantum:.2f}% {status}")
        
        results.append(WindowResult(
            window="Short-Dated",
            param_name="maturity_days",
            param_value=days,
            error_classical=error_classical,
            error_quantum=error_quantum,
            advantage=advantage,
            time_classical=time_classical,
            time_quantum=time_quantum
        ))
    
    return results


def test_window_4_perfect_correlation() -> List[WindowResult]:
    """Window 4: Test perfect/near-perfect correlation (degenerate cases)."""
    print("\n🪟 WINDOW 4: Perfect Correlation (Degenerate Cases)")
    print("-" * 70)
    
    results = []
    S0 = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    sigma = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    weights = np.ones(5) / 5
    K = 100.0
    T = 1.0
    r = 0.05
    
    correlation_levels = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    
    for base_corr in correlation_levels:
        corr = generate_correlation_matrix(5, base_corr)
        
        # Monte Carlo reference
        mc_price, _ = monte_carlo_reference(S0, sigma, corr, weights, K, T, r)
        
        # Classical
        price_classical, time_classical = price_with_fb_iqft(
            S0, sigma, corr, weights, K, T, r
        )
        
        # Quantum
        price_quantum, time_quantum = price_with_qae(
            S0, sigma, corr, weights, K, T, r
        )
        
        if mc_price > 0.01:
            error_classical = abs(price_classical - mc_price) / mc_price * 100
            error_quantum = abs(price_quantum - mc_price) / mc_price * 100
        else:
            error_classical = abs(price_classical - mc_price) * 100
            error_quantum = abs(price_quantum - mc_price) * 100
        
        advantage = error_classical - error_quantum
        
        status = "✅ QUANTUM WINS" if advantage > 0.1 else "⚪"
        print(f"  ρ = {base_corr:.2f}: Classical={error_classical:.2f}%, "
              f"Quantum={error_quantum:.2f}% {status}")
        
        results.append(WindowResult(
            window="Perfect Correlation",
            param_name="correlation",
            param_value=base_corr,
            error_classical=error_classical,
            error_quantum=error_quantum,
            advantage=advantage,
            time_classical=time_classical,
            time_quantum=time_quantum
        ))
    
    return results


def test_window_5_max_dimensionality() -> List[WindowResult]:
    """Window 5: Test maximum dimensionality."""
    print("\n🪟 WINDOW 5: Maximum Dimensionality")
    print("-" * 70)
    
    results = []
    K = 100.0
    T = 1.0
    r = 0.05
    
    n_assets_list = [2, 3, 5, 8, 10, 15, 20, 30]
    
    for n_assets in n_assets_list:
        S0 = np.ones(n_assets) * 100.0
        sigma = np.ones(n_assets) * 0.2
        corr = generate_correlation_matrix(n_assets, 0.3)
        weights = np.ones(n_assets) / n_assets
        
        # Monte Carlo reference
        mc_price, _ = monte_carlo_reference(S0, sigma, corr, weights, K, T, r)
        
        try:
            # Classical
            price_classical, time_classical = price_with_fb_iqft(
                S0, sigma, corr, weights, K, T, r
            )
        except Exception:
            price_classical = black_scholes_portfolio(S0, sigma, corr, weights, K, T, r)
            time_classical = 0.0
        
        try:
            # Quantum
            price_quantum, time_quantum = price_with_qae(
                S0, sigma, corr, weights, K, T, r, n_factors=min(3, n_assets-1)
            )
        except Exception:
            price_quantum = price_classical
            time_quantum = 0.0
        
        if mc_price > 0.01:
            error_classical = abs(price_classical - mc_price) / mc_price * 100
            error_quantum = abs(price_quantum - mc_price) / mc_price * 100
        else:
            error_classical = abs(price_classical - mc_price) * 100
            error_quantum = abs(price_quantum - mc_price) * 100
        
        advantage = error_classical - error_quantum
        
        status = "✅ QUANTUM WINS" if advantage > 0.1 else "⚪"
        print(f"  N = {n_assets:2d} assets: Classical={error_classical:.2f}%, "
              f"Quantum={error_quantum:.2f}% {status}")
        
        results.append(WindowResult(
            window="Dimensionality",
            param_name="n_assets",
            param_value=float(n_assets),
            error_classical=error_classical,
            error_quantum=error_quantum,
            advantage=advantage,
            time_classical=time_classical,
            time_quantum=time_quantum
        ))
    
    return results


def summarize_results(all_results: List[WindowResult]) -> pd.DataFrame:
    """Summarize all results and identify best quantum advantage windows."""
    print("\n" + "=" * 70)
    print("🎯 QUANTUM ADVANTAGE SUMMARY")
    print("=" * 70)
    
    df = pd.DataFrame([{
        'window': r.window,
        'param_name': r.param_name,
        'param_value': r.param_value,
        'error_classical': r.error_classical,
        'error_quantum': r.error_quantum,
        'advantage': r.advantage,
        'time_classical': r.time_classical,
        'time_quantum': r.time_quantum
    } for r in all_results])
    
    # Find wins
    wins = df[df['advantage'] > 0.1]
    
    if len(wins) > 0:
        print(f"\n✅ Found {len(wins)} winning scenarios!")
        print("\nTop 10 best opportunities:")
        top_wins = wins.nlargest(10, 'advantage')
        print(top_wins[['window', 'param_name', 'param_value', 'advantage']].to_string(index=False))
        
        # Identify strongest window
        best_window = wins.groupby('window')['advantage'].mean().idxmax()
        best_advantage = wins.groupby('window')['advantage'].mean().max()
        print(f"\n🎯 STRONGEST WINDOW: {best_window} (avg advantage: {best_advantage:.2f}%)")
    else:
        print("\n⚠️ No clear wins found in these windows.")
        print("Consider testing noise robustness and sample complexity next.")
        
        # Show closest scenarios
        print("\n📊 Closest to advantage (smallest quantum error):")
        closest = df.nsmallest(5, 'error_quantum')
        print(closest[['window', 'param_name', 'param_value', 'error_quantum']].to_string(index=False))
    
    return df


def find_quantum_advantage_windows() -> pd.DataFrame:
    """
    Main entry point: Test all 5 windows and find quantum advantage.
    """
    print("🔍 HUNTING FOR QUANTUM ADVANTAGE")
    print("=" * 70)
    print("Testing 5 extreme scenarios where quantum might outperform classical...")
    
    all_results = []
    
    # Run all window tests
    all_results.extend(test_window_1_extreme_strikes())
    all_results.extend(test_window_2_high_volatility())
    all_results.extend(test_window_3_short_dated())
    all_results.extend(test_window_4_perfect_correlation())
    all_results.extend(test_window_5_max_dimensionality())
    
    # Summarize
    df = summarize_results(all_results)
    
    # Save results
    output_dir = '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP/qfdp_qml/results/advantage_discovery'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f'{output_dir}/window_discovery_results.csv', index=False)
    print(f"\n✅ Results saved to {output_dir}/window_discovery_results.csv")
    
    return df


if __name__ == '__main__':
    df = find_quantum_advantage_windows()
