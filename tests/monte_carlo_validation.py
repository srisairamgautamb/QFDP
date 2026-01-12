"""
Monte Carlo Validation Suite
============================

Compares QRC+QTC+FB-IQFT pricing against Monte Carlo simulation.

Tests:
1. Single-asset option (Black-Scholes benchmark)
2. Multi-asset portfolio option (correlation effects)
3. Various moneyness levels (ITM, ATM, OTM)
4. Different maturities (short, medium, long)
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import norm
import logging
import sys
import time

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.fb_iqft_pricing import FBIQFTPricing
from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
from qfdp.integrated.corrected_qtc_integrated_pricer import CorrectedQTCIntegratedPricer

logging.basicConfig(level=logging.WARNING)


# ============================================================================
# MONTE CARLO PRICER
# ============================================================================

class MonteCarloOptionPricer:
    """
    Monte Carlo simulation for option pricing.
    Serves as ground truth for validation.
    """
    
    def __init__(self, n_paths: int = 100000, n_steps: int = 252):
        """
        Args:
            n_paths: Number of simulation paths
            n_steps: Number of time steps per path (252 = daily for 1 year)
        """
        self.n_paths = n_paths
        self.n_steps = n_steps
    
    def price_european_call(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        seed: int = None
    ) -> Dict:
        """
        Price European call option via Monte Carlo.
        
        Uses geometric Brownian motion:
        dS = μSdt + σSdW
        
        Returns:
            Dict with price, std_error, confidence interval
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / self.n_steps
        
        # Generate paths
        Z = np.random.standard_normal((self.n_paths, self.n_steps))
        
        # GBM simulation
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        log_returns = drift + diffusion * Z
        log_S = np.log(S0) + np.cumsum(log_returns, axis=1)
        S_T = np.exp(log_S[:, -1])  # Terminal prices
        
        # Payoffs
        payoffs = np.maximum(S_T - K, 0)
        
        # Discounted expected payoff
        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        
        # Standard error
        std_error = discount * np.std(payoffs) / np.sqrt(self.n_paths)
        
        # 95% confidence interval
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error
        
        return {
            'price': price,
            'std_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_paths': self.n_paths,
            'S_T_mean': np.mean(S_T),
            'S_T_std': np.std(S_T)
        }
    
    def price_portfolio_option(
        self,
        asset_prices: np.ndarray,
        asset_volatilities: np.ndarray,
        correlation_matrix: np.ndarray,
        portfolio_weights: np.ndarray,
        K: float,
        T: float,
        r: float,
        seed: int = None
    ) -> Dict:
        """
        Price portfolio option via correlated Monte Carlo.
        
        Simulates correlated asset paths using Cholesky decomposition.
        
        Returns:
            Dict with portfolio option price and statistics
        """
        if seed is not None:
            np.random.seed(seed)
        
        N = len(asset_prices)
        dt = T / self.n_steps
        
        # Cholesky decomposition for correlated normals
        L = np.linalg.cholesky(correlation_matrix)
        
        # Generate independent standard normals
        Z_independent = np.random.standard_normal((self.n_paths, self.n_steps, N))
        
        # Correlate them
        Z_correlated = np.einsum('ijk,lk->ijl', Z_independent, L)
        
        # Initialize log prices
        log_S = np.log(asset_prices)  # (N,)
        log_S = np.tile(log_S, (self.n_paths, 1))  # (n_paths, N)
        
        # Simulate paths
        for t in range(self.n_steps):
            drift = (r - 0.5 * asset_volatilities**2) * dt
            diffusion = asset_volatilities * np.sqrt(dt) * Z_correlated[:, t, :]
            log_S += drift + diffusion
        
        S_T = np.exp(log_S)  # Terminal prices (n_paths, N)
        
        # Portfolio value at maturity
        portfolio_T = np.sum(portfolio_weights * S_T, axis=1)  # (n_paths,)
        
        # Portfolio value at t=0
        portfolio_0 = np.sum(portfolio_weights * asset_prices)
        
        # Call option payoff
        payoffs = np.maximum(portfolio_T - K, 0)
        
        # Discounted expected payoff
        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        
        # Standard error
        std_error = discount * np.std(payoffs) / np.sqrt(self.n_paths)
        
        # Compute portfolio volatility from simulation
        portfolio_returns = np.log(portfolio_T / portfolio_0)
        sigma_p_simulated = np.std(portfolio_returns) / np.sqrt(T)
        
        return {
            'price': price,
            'std_error': std_error,
            'ci_lower': price - 1.96 * std_error,
            'ci_upper': price + 1.96 * std_error,
            'portfolio_0': portfolio_0,
            'portfolio_T_mean': np.mean(portfolio_T),
            'sigma_p_simulated': sigma_p_simulated,
            'n_paths': self.n_paths
        }


# ============================================================================
# VALIDATION TESTS
# ============================================================================

def generate_correlation_matrix(n: int, rho: float) -> np.ndarray:
    """Generate N×N correlation matrix with uniform off-diagonal correlation."""
    return np.eye(n) * (1 - rho) + rho


def generate_price_history(trend: str = 'up', base_price: float = 100.0) -> np.ndarray:
    """Generate 6-point price history based on trend."""
    if trend == 'up':
        return base_price + np.array([-2, -1, 0, 1, 2, 3])
    elif trend == 'down':
        return base_price + np.array([3, 2, 1, -1, -3, -5])
    elif trend == 'volatile':
        return base_price + np.array([0, 3, -2, 4, -3, 2])
    else:
        return np.full(6, base_price)


def run_monte_carlo_validation():
    """
    Comprehensive Monte Carlo validation.
    """
    print("=" * 80)
    print("MONTE CARLO VALIDATION SUITE")
    print("=" * 80)
    print("Comparing QRC+QTC+FB-IQFT against Monte Carlo (100,000 paths)")
    print("=" * 80)
    
    # Initialize pricers
    fb_iqft = FBIQFTPricing(M=16, alpha=1.0)
    qrc_qtc_pricer = CorrectedQTCIntegratedPricer(fb_iqft, qrc_beta=0.1, qtc_gamma=0.05)
    mc_pricer = MonteCarloOptionPricer(n_paths=100000, n_steps=252)
    
    results = []
    
    # =========================================================================
    # TEST 1: SINGLE-ASSET (BLACK-SCHOLES BENCHMARK)
    # =========================================================================
    print("\n" + "-" * 60)
    print("TEST 1: Single-Asset Option (Black-Scholes Benchmark)")
    print("-" * 60)
    
    test_cases = [
        {'S': 100, 'K': 100, 'T': 1.0, 'sigma': 0.20, 'label': 'ATM σ=20%'},
        {'S': 100, 'K': 90, 'T': 1.0, 'sigma': 0.20, 'label': 'ITM (K=90)'},
        {'S': 100, 'K': 110, 'T': 1.0, 'sigma': 0.20, 'label': 'OTM (K=110)'},
        {'S': 100, 'K': 100, 'T': 0.25, 'sigma': 0.20, 'label': 'Short T=0.25'},
        {'S': 100, 'K': 100, 'T': 1.0, 'sigma': 0.40, 'label': 'High σ=40%'},
    ]
    
    print(f"\n{'Test Case':<20} {'MC Price':<12} {'QRC-QTC':<12} {'Carr-Madan':<12} {'MC Error':<10}")
    print("-" * 70)
    
    for tc in test_cases:
        # Monte Carlo
        mc_result = mc_pricer.price_european_call(
            tc['S'], tc['K'], tc['T'], 0.05, tc['sigma'], seed=42
        )
        
        # Classical Carr-Madan
        cm_result = price_call_option_corrected(
            tc['S'], tc['K'], tc['T'], 0.05, tc['sigma']
        )
        
        # QRC+QTC (single asset = N=1 portfolio)
        market_data = {
            'spot_prices': np.array([tc['S']]),
            'volatilities': np.array([tc['sigma']]),
            'correlation_matrix': np.array([[1.0]]),
            'weights': np.array([1.0]),
            'maturity': tc['T'],
            'risk_free_rate': 0.05
        }
        price_history = generate_price_history('up', tc['S'])
        qrc_result = qrc_qtc_pricer.price_with_full_quantum_pipeline(
            market_data, price_history, strike=tc['K'], use_quantum_circuit=True
        )
        
        mc_error = abs(qrc_result['price_quantum'] - mc_result['price']) / mc_result['price'] * 100
        
        print(f"{tc['label']:<20} ${mc_result['price']:<10.4f} ${qrc_result['price_quantum']:<10.4f} ${cm_result['price']:<10.4f} {mc_error:<9.2f}%")
        
        results.append({
            'test': 'single_asset',
            'label': tc['label'],
            'mc_price': mc_result['price'],
            'qrc_price': qrc_result['price_quantum'],
            'cm_price': cm_result['price'],
            'mc_error': mc_error
        })
    
    # =========================================================================
    # TEST 2: MULTI-ASSET PORTFOLIO
    # =========================================================================
    print("\n" + "-" * 60)
    print("TEST 2: Multi-Asset Portfolio Option")
    print("-" * 60)
    
    portfolio_tests = [
        {'n': 2, 'rho': 0.3, 'label': 'N=2, ρ=0.3'},
        {'n': 2, 'rho': 0.7, 'label': 'N=2, ρ=0.7'},
        {'n': 5, 'rho': 0.3, 'label': 'N=5, ρ=0.3'},
        {'n': 5, 'rho': 0.7, 'label': 'N=5, ρ=0.7'},
        {'n': 10, 'rho': 0.5, 'label': 'N=10, ρ=0.5'},
    ]
    
    print(f"\n{'Portfolio':<15} {'MC Price':<12} {'QRC-QTC':<12} {'σ_p(MC)':<10} {'σ_p(QRC)':<10} {'Error':<10}")
    print("-" * 75)
    
    for pt in portfolio_tests:
        n = pt['n']
        rho = pt['rho']
        
        asset_prices = np.full(n, 100.0)
        asset_vols = np.full(n, 0.20)
        weights = np.ones(n) / n
        corr = generate_correlation_matrix(n, rho)
        K = 100.0
        
        # Monte Carlo
        mc_result = mc_pricer.price_portfolio_option(
            asset_prices, asset_vols, corr, weights,
            K=K, T=1.0, r=0.05, seed=42
        )
        
        # QRC+QTC
        market_data = {
            'spot_prices': asset_prices,
            'volatilities': asset_vols,
            'correlation_matrix': corr,
            'weights': weights,
            'maturity': 1.0,
            'risk_free_rate': 0.05
        }
        price_history = generate_price_history('up')
        qrc_result = qrc_qtc_pricer.price_with_full_quantum_pipeline(
            market_data, price_history, strike=K, use_quantum_circuit=True
        )
        
        mc_error = abs(qrc_result['price_quantum'] - mc_result['price']) / mc_result['price'] * 100
        
        print(f"{pt['label']:<15} ${mc_result['price']:<10.4f} ${qrc_result['price_quantum']:<10.4f} {mc_result['sigma_p_simulated']:<9.4f} {qrc_result['sigma_p_enhanced']:<9.4f} {mc_error:<9.2f}%")
        
        results.append({
            'test': 'portfolio',
            'label': pt['label'],
            'mc_price': mc_result['price'],
            'qrc_price': qrc_result['price_quantum'],
            'sigma_p_mc': mc_result['sigma_p_simulated'],
            'sigma_p_qrc': qrc_result['sigma_p_enhanced'],
            'mc_error': mc_error
        })
    
    # =========================================================================
    # TEST 3: REGIME STRESS TEST
    # =========================================================================
    print("\n" + "-" * 60)
    print("TEST 3: Regime Stress Test (Calm → Crisis)")
    print("-" * 60)
    
    regimes = [
        {'rho': 0.3, 'trend': 'up', 'label': 'Calm'},
        {'rho': 0.5, 'trend': 'flat', 'label': 'Medium'},
        {'rho': 0.7, 'trend': 'volatile', 'label': 'High Vol'},
        {'rho': 0.85, 'trend': 'down', 'label': 'Crisis'},
    ]
    
    n = 5
    asset_prices = np.full(n, 100.0)
    asset_vols = np.full(n, 0.20)
    weights = np.ones(n) / n
    K = 100.0
    
    print(f"\n{'Regime':<12} {'MC Price':<12} {'QRC-QTC':<12} {'Error':<10}")
    print("-" * 50)
    
    for regime in regimes:
        corr = generate_correlation_matrix(n, regime['rho'])
        
        # Monte Carlo
        mc_result = mc_pricer.price_portfolio_option(
            asset_prices, asset_vols, corr, weights,
            K=K, T=1.0, r=0.05, seed=42
        )
        
        # QRC+QTC
        market_data = {
            'spot_prices': asset_prices,
            'volatilities': asset_vols,
            'correlation_matrix': corr,
            'weights': weights,
            'maturity': 1.0,
            'risk_free_rate': 0.05
        }
        price_history = generate_price_history(regime['trend'])
        qrc_result = qrc_qtc_pricer.price_with_full_quantum_pipeline(
            market_data, price_history, strike=K, use_quantum_circuit=True
        )
        
        mc_error = abs(qrc_result['price_quantum'] - mc_result['price']) / mc_result['price'] * 100
        
        print(f"{regime['label']:<12} ${mc_result['price']:<10.4f} ${qrc_result['price_quantum']:<10.4f} {mc_error:<9.2f}%")
        
        results.append({
            'test': 'regime',
            'label': regime['label'],
            'mc_price': mc_result['price'],
            'qrc_price': qrc_result['price_quantum'],
            'mc_error': mc_error
        })
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_errors = [r['mc_error'] for r in results]
    print(f"\nMean Error vs MC: {np.mean(all_errors):.2f}%")
    print(f"Max Error vs MC:  {np.max(all_errors):.2f}%")
    print(f"Min Error vs MC:  {np.min(all_errors):.2f}%")
    
    # By test type
    single_errors = [r['mc_error'] for r in results if r['test'] == 'single_asset']
    portfolio_errors = [r['mc_error'] for r in results if r['test'] == 'portfolio']
    regime_errors = [r['mc_error'] for r in results if r['test'] == 'regime']
    
    print(f"\nBy Test Type:")
    print(f"  Single-Asset: Mean = {np.mean(single_errors):.2f}%")
    print(f"  Portfolio:    Mean = {np.mean(portfolio_errors):.2f}%")
    print(f"  Regime:       Mean = {np.mean(regime_errors):.2f}%")
    
    # Pass/Fail
    threshold = 5.0  # 5% error threshold
    passed = sum(1 for e in all_errors if e < threshold)
    total = len(all_errors)
    
    print(f"\nPASS/FAIL (threshold = {threshold}% error):")
    print(f"  {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL MONTE CARLO VALIDATION TESTS PASSED!")
    else:
        failed = [r for r in results if r['mc_error'] >= threshold]
        print(f"\n⚠️  {len(failed)} test(s) exceeded {threshold}% error threshold")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_monte_carlo_validation()
