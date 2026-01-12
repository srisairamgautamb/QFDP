"""
Greeks Computation Module
=========================

Computes option Greeks (Delta, Gamma, Vega, Theta, Rho) using the
QRC+QTC+FB-IQFT quantum pricing engine.

Greeks are computed via finite differences on the quantum pricer.
"""

import numpy as np
from typing import Dict, Tuple
import logging
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.fb_iqft_pricing import FBIQFTPricing
from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
from qfdp.integrated.corrected_qtc_integrated_pricer import CorrectedQTCIntegratedPricer
from scipy.stats import norm

logging.basicConfig(level=logging.WARNING)


# ============================================================================
# BLACK-SCHOLES GREEKS (REFERENCE)
# ============================================================================

def bs_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Dict:
    """
    Compute Black-Scholes Greeks for reference.
    
    Returns:
        Dict with delta, gamma, vega, theta, rho
    """
    if sigma <= 0 or T <= 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # N(d1), N(d2), n(d1)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)
    
    delta = Nd1
    gamma = nd1 / (S * sigma * sqrt_T)
    vega = S * nd1 * sqrt_T / 100  # Per 1% change in vol
    theta = (-S * nd1 * sigma / (2 * sqrt_T) - r * K * np.exp(-r * T) * Nd2) / 252  # Daily
    rho = K * T * np.exp(-r * T) * Nd2 / 100  # Per 1% change in rate
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


# ============================================================================
# QUANTUM GREEKS CALCULATOR
# ============================================================================

class QuantumGreeksCalculator:
    """
    Compute Greeks using the QRC+QTC+FB-IQFT quantum pricer.
    
    Uses finite difference methods on the quantum pricing function.
    """
    
    def __init__(
        self, 
        qrc_beta: float = 0.01,  # Optimized value
        qtc_gamma: float = 0.018  # Optimized value
    ):
        self.fb_iqft = FBIQFTPricing(M=16, alpha=1.0)
        self.pricer = CorrectedQTCIntegratedPricer(
            self.fb_iqft, 
            qrc_beta=qrc_beta, 
            qtc_gamma=qtc_gamma
        )
        
        # Finite difference steps
        self.dS = 1.0  # For delta/gamma
        self.dsigma = 0.01  # For vega
        self.dT = 1/252  # For theta (1 day)
        self.dr = 0.0001  # For rho
    
    def _price(
        self, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float,
        n_assets: int = 5
    ) -> float:
        """
        Price using quantum pricer.
        """
        asset_prices = np.full(n_assets, S)
        asset_vols = np.full(n_assets, sigma)
        weights = np.ones(n_assets) / n_assets
        corr = np.eye(n_assets) * 0.7 + 0.3  # ρ=0.3
        
        market_data = {
            'spot_prices': asset_prices,
            'volatilities': asset_vols,
            'correlation_matrix': corr,
            'weights': weights,
            'maturity': max(T, 0.001),  # Avoid T=0
            'risk_free_rate': r
        }
        
        price_history = S + np.array([-2, -1, 0, 1, 2, 3])  # Up trend
        
        result = self.pricer.price_with_full_quantum_pipeline(
            market_data, price_history, strike=K, use_quantum_circuit=True
        )
        
        return result['price_quantum']
    
    def compute_greeks(
        self, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float,
        n_assets: int = 5
    ) -> Dict:
        """
        Compute all Greeks using finite differences.
        
        Returns:
            Dict with delta, gamma, vega, theta, rho
        """
        # Base price
        P = self._price(S, K, T, r, sigma, n_assets)
        
        # Delta: ∂P/∂S (central difference)
        P_up = self._price(S + self.dS, K, T, r, sigma, n_assets)
        P_down = self._price(S - self.dS, K, T, r, sigma, n_assets)
        delta = (P_up - P_down) / (2 * self.dS)
        
        # Gamma: ∂²P/∂S² (second derivative)
        gamma = (P_up - 2 * P + P_down) / (self.dS ** 2)
        
        # Vega: ∂P/∂σ (per 1% vol change)
        P_sigma_up = self._price(S, K, T, r, sigma + self.dsigma, n_assets)
        P_sigma_down = self._price(S, K, T, r, sigma - self.dsigma, n_assets)
        vega = (P_sigma_up - P_sigma_down) / (2 * self.dsigma) * 0.01
        
        # Theta: ∂P/∂T (daily decay)
        if T > self.dT:
            P_T_down = self._price(S, K, T - self.dT, r, sigma, n_assets)
            theta = -(P - P_T_down) / self.dT  # Negative because time decreases
        else:
            theta = 0
        
        # Rho: ∂P/∂r (per 1% rate change)
        P_r_up = self._price(S, K, T, r + self.dr, sigma, n_assets)
        P_r_down = self._price(S, K, T, r - self.dr, sigma, n_assets)
        rho = (P_r_up - P_r_down) / (2 * self.dr) * 0.01
        
        return {
            'price': P,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }


# ============================================================================
# VALIDATION
# ============================================================================

def run_greeks_validation():
    """
    Validate quantum Greeks against Black-Scholes.
    """
    print("=" * 80)
    print("QUANTUM GREEKS VALIDATION")
    print("=" * 80)
    print("Computing Greeks using QRC+QTC+FB-IQFT and comparing to Black-Scholes")
    print("=" * 80)
    
    calculator = QuantumGreeksCalculator()
    
    # Test cases
    test_cases = [
        {'S': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.20, 'label': 'ATM'},
        {'S': 100, 'K': 90, 'T': 1.0, 'r': 0.05, 'sigma': 0.20, 'label': 'ITM'},
        {'S': 100, 'K': 110, 'T': 1.0, 'r': 0.05, 'sigma': 0.20, 'label': 'OTM'},
        {'S': 100, 'K': 100, 'T': 0.25, 'r': 0.05, 'sigma': 0.20, 'label': 'Short T'},
        {'S': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.40, 'label': 'High σ'},
    ]
    
    results = []
    
    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {tc['label']} (S={tc['S']}, K={tc['K']}, T={tc['T']}, σ={tc['sigma']})")
        print(f"{'='*60}")
        
        # Black-Scholes reference
        bs = bs_greeks(tc['S'], tc['K'], tc['T'], tc['r'], tc['sigma'])
        bs_price = price_call_option_corrected(tc['S'], tc['K'], tc['T'], tc['r'], tc['sigma'])['price']
        
        # Quantum Greeks
        qg = calculator.compute_greeks(tc['S'], tc['K'], tc['T'], tc['r'], tc['sigma'])
        
        print(f"\n{'Greek':<10} {'BS':<12} {'Quantum':<12} {'Error':<10}")
        print("-" * 50)
        
        greek_names = ['price', 'delta', 'gamma', 'vega', 'theta', 'rho']
        bs_values = [bs_price, bs['delta'], bs['gamma'], bs['vega'], bs['theta'], bs['rho']]
        
        for name, bs_val in zip(greek_names, bs_values):
            q_val = qg[name]
            if abs(bs_val) > 0.0001:
                error = abs(q_val - bs_val) / abs(bs_val) * 100
            else:
                error = abs(q_val - bs_val) * 100
            print(f"{name:<10} {bs_val:<12.4f} {q_val:<12.4f} {error:<9.2f}%")
            
            results.append({
                'test': tc['label'],
                'greek': name,
                'bs': bs_val,
                'quantum': q_val,
                'error': error
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for greek in greek_names:
        greek_results = [r for r in results if r['greek'] == greek]
        errors = [r['error'] for r in greek_results]
        print(f"{greek:<10}: Mean Error = {np.mean(errors):.2f}%, Max = {np.max(errors):.2f}%")
    
    print("\n✅ GREEKS VALIDATION COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_greeks_validation()
