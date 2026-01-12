"""
Hyperparameter Optimization
===========================

Optimize β (QRC modulation) and γ (QTC modulation) for minimal error.

Uses scipy.optimize to find optimal parameters.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple
import logging
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.fb_iqft_pricing import FBIQFTPricing
from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
from qfdp.integrated.corrected_qtc_integrated_pricer import CorrectedQTCIntegratedPricer

logging.basicConfig(level=logging.WARNING)


# ============================================================================
# OPTIMIZATION OBJECTIVE
# ============================================================================

def generate_correlation_matrix(n: int, rho: float) -> np.ndarray:
    return np.eye(n) * (1 - rho) + rho


def generate_price_history(trend: str = 'up', base: float = 100.0) -> np.ndarray:
    trends = {
        'up': np.array([-2, -1, 0, 1, 2, 3]),
        'down': np.array([3, 2, 1, -1, -3, -5]),
        'volatile': np.array([0, 3, -2, 4, -3, 2]),
        'flat': np.array([0, 0.1, -0.1, 0.2, 0, 0.1])
    }
    return base + trends.get(trend, trends['up'])


# Test scenarios for optimization
TEST_SCENARIOS = [
    {'n': 5, 'rho': 0.3, 'trend': 'up', 'label': 'Calm'},
    {'n': 5, 'rho': 0.5, 'trend': 'flat', 'label': 'Medium'},
    {'n': 5, 'rho': 0.7, 'trend': 'volatile', 'label': 'High'},
    {'n': 5, 'rho': 0.85, 'trend': 'down', 'label': 'Crisis'},
    {'n': 10, 'rho': 0.5, 'trend': 'up', 'label': 'N=10 Medium'},
    {'n': 10, 'rho': 0.7, 'trend': 'down', 'label': 'N=10 High'},
]


def compute_error(beta: float, gamma: float, fb_iqft) -> float:
    """
    Compute mean error across all test scenarios.
    
    Lower is better.
    """
    pricer = CorrectedQTCIntegratedPricer(fb_iqft, qrc_beta=beta, qtc_gamma=gamma)
    
    errors = []
    
    for scenario in TEST_SCENARIOS:
        n = scenario['n']
        rho = scenario['rho']
        trend = scenario['trend']
        
        asset_prices = np.full(n, 100.0)
        asset_vols = np.full(n, 0.20)
        weights = np.ones(n) / n
        corr = generate_correlation_matrix(n, rho)
        price_history = generate_price_history(trend)
        
        # True σ_p
        vol_matrix = np.diag(asset_vols)
        cov = vol_matrix @ corr @ vol_matrix
        sigma_p_true = float(np.sqrt(weights.T @ cov @ weights))
        price_true = price_call_option_corrected(100, 100, 1.0, 0.05, sigma_p_true)['price']
        
        # QRC+QTC price
        market_data = {
            'spot_prices': asset_prices,
            'volatilities': asset_vols,
            'correlation_matrix': corr,
            'weights': weights,
            'maturity': 1.0,
            'risk_free_rate': 0.05
        }
        
        result = pricer.price_with_full_quantum_pipeline(
            market_data, price_history, strike=100.0, use_quantum_circuit=True
        )
        
        error = abs(result['price_quantum'] - price_true) / price_true * 100
        errors.append(error)
    
    return np.mean(errors)


def objective_function(params, fb_iqft):
    """Optimization objective (to minimize)."""
    beta, gamma = params
    return compute_error(beta, gamma, fb_iqft)


# ============================================================================
# OPTIMIZATION
# ============================================================================

def run_hyperparameter_optimization():
    """
    Find optimal β and γ using multiple methods.
    """
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print("Finding optimal β (QRC) and γ (QTC) for minimal pricing error")
    print("=" * 80)
    
    fb_iqft = FBIQFTPricing(M=16, alpha=1.0)
    
    # Current defaults
    current_beta = 0.1
    current_gamma = 0.05
    current_error = compute_error(current_beta, current_gamma, fb_iqft)
    
    print(f"\nCurrent parameters: β={current_beta}, γ={current_gamma}")
    print(f"Current mean error: {current_error:.4f}%")
    
    # Method 1: Grid Search
    print("\n" + "-" * 60)
    print("Method 1: Grid Search")
    print("-" * 60)
    
    beta_range = np.arange(0.05, 0.25, 0.05)
    gamma_range = np.arange(0.02, 0.12, 0.02)
    
    best_grid = {'beta': 0.1, 'gamma': 0.05, 'error': current_error}
    
    print(f"\n{'β':<8} {'γ':<8} {'Error':<10}")
    print("-" * 30)
    
    for beta in beta_range:
        for gamma in gamma_range:
            error = compute_error(beta, gamma, fb_iqft)
            if error < best_grid['error']:
                best_grid = {'beta': beta, 'gamma': gamma, 'error': error}
            print(f"{beta:<8.3f} {gamma:<8.3f} {error:<9.4f}%")
    
    print(f"\nBest (Grid): β={best_grid['beta']:.3f}, γ={best_grid['gamma']:.3f}, Error={best_grid['error']:.4f}%")
    
    # Method 2: Scipy Minimize (Nelder-Mead)
    print("\n" + "-" * 60)
    print("Method 2: Nelder-Mead Optimization")
    print("-" * 60)
    
    result_nm = minimize(
        lambda p: objective_function(p, fb_iqft),
        x0=[best_grid['beta'], best_grid['gamma']],
        method='Nelder-Mead',
        options={'maxiter': 50, 'xatol': 0.01, 'fatol': 0.01}
    )
    
    best_nm_beta, best_nm_gamma = result_nm.x
    best_nm_error = result_nm.fun
    
    print(f"Best (NM): β={best_nm_beta:.4f}, γ={best_nm_gamma:.4f}, Error={best_nm_error:.4f}%")
    
    # Method 3: Differential Evolution (global optimization)
    print("\n" + "-" * 60)
    print("Method 3: Differential Evolution (Global)")
    print("-" * 60)
    
    bounds = [(0.01, 0.30), (0.01, 0.15)]  # β, γ bounds
    
    result_de = differential_evolution(
        lambda p: objective_function(p, fb_iqft),
        bounds=bounds,
        seed=42,
        maxiter=30,
        tol=0.01,
        workers=1,
        disp=False
    )
    
    best_de_beta, best_de_gamma = result_de.x
    best_de_error = result_de.fun
    
    print(f"Best (DE): β={best_de_beta:.4f}, γ={best_de_gamma:.4f}, Error={best_de_error:.4f}%")
    
    # Final Recommendation
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    
    all_results = [
        {'method': 'Current', 'beta': current_beta, 'gamma': current_gamma, 'error': current_error},
        {'method': 'Grid', 'beta': best_grid['beta'], 'gamma': best_grid['gamma'], 'error': best_grid['error']},
        {'method': 'Nelder-Mead', 'beta': best_nm_beta, 'gamma': best_nm_gamma, 'error': best_nm_error},
        {'method': 'Diff Evolution', 'beta': best_de_beta, 'gamma': best_de_gamma, 'error': best_de_error}
    ]
    
    print(f"\n{'Method':<15} {'β':<10} {'γ':<10} {'Error':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for r in all_results:
        improvement = current_error - r['error']
        print(f"{r['method']:<15} {r['beta']:<10.4f} {r['gamma']:<10.4f} {r['error']:<11.4f}% {improvement:+.4f}%")
    
    # Best overall
    best = min(all_results, key=lambda x: x['error'])
    improvement = current_error - best['error']
    
    print(f"\n🏆 RECOMMENDED: β={best['beta']:.4f}, γ={best['gamma']:.4f}")
    print(f"   Error: {best['error']:.4f}% (improvement: {improvement:+.4f}%)")
    
    print("\n" + "=" * 80)
    print("✅ HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    return best


if __name__ == "__main__":
    run_hyperparameter_optimization()
