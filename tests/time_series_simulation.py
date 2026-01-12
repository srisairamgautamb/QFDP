"""
Time-Series Simulation Test
============================

Track QRC+QTC+FB-IQFT performance over simulated market history.

Simulates 2 years of market data with:
- Regime changes (calm → crisis → recovery)
- Correlation dynamics
- Volatility clustering

Evaluates:
- Cumulative pricing accuracy
- Adaptation speed
- Regime transition handling
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.fb_iqft_pricing import FBIQFTPricing
from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
from qfdp.integrated.corrected_qtc_integrated_pricer import CorrectedQTCIntegratedPricer

logging.basicConfig(level=logging.WARNING)


# ============================================================================
# MARKET SIMULATOR
# ============================================================================

class MarketSimulator:
    """
    Simulates realistic market dynamics with regime changes.
    """
    
    def __init__(self, n_assets: int = 5, seed: int = 42):
        np.random.seed(seed)
        self.n_assets = n_assets
        self.base_vol = 0.20
        
    def generate_regime_sequence(self, n_days: int) -> List[str]:
        """
        Generate a sequence of market regimes.
        
        Typical pattern:
        calm (60%) → medium (20%) → crisis (10%) → recovery (10%)
        """
        regimes = []
        current_regime = 'calm'
        
        transition_probs = {
            'calm': {'calm': 0.95, 'medium': 0.04, 'crisis': 0.01},
            'medium': {'calm': 0.30, 'medium': 0.60, 'crisis': 0.10},
            'crisis': {'calm': 0.05, 'medium': 0.25, 'crisis': 0.60, 'recovery': 0.10},
            'recovery': {'calm': 0.40, 'medium': 0.40, 'crisis': 0.05, 'recovery': 0.15}
        }
        
        for _ in range(n_days):
            regimes.append(current_regime)
            probs = transition_probs[current_regime]
            regimes_list = list(probs.keys())
            probs_list = list(probs.values())
            current_regime = np.random.choice(regimes_list, p=probs_list)
        
        return regimes
    
    def get_regime_params(self, regime: str) -> Dict:
        """Get parameters for each regime."""
        params = {
            'calm': {'rho': 0.30, 'vol_mult': 1.0},
            'medium': {'rho': 0.50, 'vol_mult': 1.3},
            'crisis': {'rho': 0.85, 'vol_mult': 2.5},
            'recovery': {'rho': 0.40, 'vol_mult': 1.5}
        }
        return params.get(regime, params['calm'])
    
    def simulate(self, n_days: int = 504) -> Dict:
        """
        Simulate n_days (default 2 years) of market data.
        
        Returns:
            Dict with prices, regimes, correlations, volatilities
        """
        regimes = self.generate_regime_sequence(n_days)
        
        # Initialize
        prices = np.zeros((n_days, self.n_assets))
        prices[0] = 100.0  # Start at 100
        
        correlations = []
        volatilities = []
        
        for t in range(1, n_days):
            regime = regimes[t]
            params = self.get_regime_params(regime)
            
            rho = params['rho']
            vol_mult = params['vol_mult']
            
            # Store current state
            corr_matrix = np.eye(self.n_assets) * (1 - rho) + rho
            correlations.append(corr_matrix)
            
            asset_vols = np.full(self.n_assets, self.base_vol * vol_mult)
            volatilities.append(asset_vols)
            
            # Generate correlated returns
            L = np.linalg.cholesky(corr_matrix)
            Z = np.random.standard_normal(self.n_assets)
            Z_corr = L @ Z
            
            dt = 1 / 252
            drift = 0.05 * dt  # 5% risk-free rate
            diffusion = asset_vols * np.sqrt(dt) * Z_corr
            
            log_returns = drift + diffusion
            prices[t] = prices[t-1] * np.exp(log_returns)
        
        return {
            'prices': prices,
            'regimes': regimes,
            'correlations': correlations,
            'volatilities': volatilities
        }


# ============================================================================
# ROLLING WINDOW EVALUATION
# ============================================================================

def run_time_series_simulation():
    """
    Run time-series simulation and track cumulative performance.
    """
    print("=" * 80)
    print("TIME-SERIES SIMULATION TEST")
    print("=" * 80)
    print("Simulating 2 years of market data with regime changes")
    print("=" * 80)
    
    # Initialize
    fb_iqft = FBIQFTPricing(M=16, alpha=1.0)
    qrc_qtc_pricer = CorrectedQTCIntegratedPricer(fb_iqft, qrc_beta=0.1, qtc_gamma=0.05)
    
    # Simulate market
    n_assets = 5
    n_days = 504  # 2 years
    simulator = MarketSimulator(n_assets=n_assets, seed=42)
    market_data = simulator.simulate(n_days)
    
    prices = market_data['prices']
    regimes = market_data['regimes']
    correlations = market_data['correlations']
    volatilities = market_data['volatilities']
    
    # Count regimes
    regime_counts = {r: regimes.count(r) for r in set(regimes)}
    print(f"\nRegime Distribution: {regime_counts}")
    
    # Rolling window evaluation
    window_size = 6
    eval_interval = 21  # Monthly evaluation
    
    results = []
    
    print(f"\nEvaluating every {eval_interval} days (monthly) over {n_days} days")
    print("-" * 80)
    print(f"{'Day':<8} {'Regime':<12} {'Portfolio $':<12} {'σ_p True':<12} {'σ_p QRC':<12} {'Error':<10}")
    print("-" * 80)
    
    for t in range(window_size, n_days, eval_interval):
        regime = regimes[t]
        current_prices = prices[t]
        corr = correlations[t-1] if t > 0 else np.eye(n_assets)
        vols = volatilities[t-1] if t > 0 else np.full(n_assets, 0.20)
        weights = np.ones(n_assets) / n_assets
        
        # Price history for QTC
        price_history = np.mean(prices[t-window_size:t], axis=1)  # Average portfolio price
        
        # True portfolio volatility
        vol_matrix = np.diag(vols)
        cov = vol_matrix @ corr @ vol_matrix
        sigma_p_true = float(np.sqrt(weights.T @ cov @ weights))
        
        # QRC+QTC pricing
        market_input = {
            'spot_prices': current_prices,
            'volatilities': vols,
            'correlation_matrix': corr,
            'weights': weights,
            'maturity': 1.0,
            'risk_free_rate': 0.05
        }
        
        portfolio_value = np.sum(weights * current_prices)
        
        qrc_result = qrc_qtc_pricer.price_with_full_quantum_pipeline(
            market_input, price_history, strike=portfolio_value, use_quantum_circuit=True
        )
        
        sigma_p_qrc = qrc_result['sigma_p_enhanced']
        error = abs(sigma_p_qrc - sigma_p_true) / sigma_p_true * 100
        
        results.append({
            'day': t,
            'regime': regime,
            'portfolio_value': portfolio_value,
            'sigma_p_true': sigma_p_true,
            'sigma_p_qrc': sigma_p_qrc,
            'error': error
        })
        
        # Print monthly summary
        print(f"{t:<8} {regime:<12} ${portfolio_value:<10.2f} {sigma_p_true:<12.4f} {sigma_p_qrc:<12.4f} {error:<9.2f}%")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    all_errors = [r['error'] for r in results]
    print(f"\nOverall σ_p Estimation Error:")
    print(f"  Mean: {np.mean(all_errors):.2f}%")
    print(f"  Std:  {np.std(all_errors):.2f}%")
    print(f"  Max:  {np.max(all_errors):.2f}%")
    print(f"  Min:  {np.min(all_errors):.2f}%")
    
    # By regime
    print("\nError by Regime:")
    for regime in set(regimes):
        regime_errors = [r['error'] for r in results if r['regime'] == regime]
        if regime_errors:
            print(f"  {regime:<12}: Mean = {np.mean(regime_errors):.2f}%, Count = {len(regime_errors)}")
    
    # Adaptation analysis
    print("\nRegime Transition Analysis:")
    transitions = 0
    transition_errors = []
    for i in range(1, len(results)):
        if results[i]['regime'] != results[i-1]['regime']:
            transitions += 1
            transition_errors.append(results[i]['error'])
    
    print(f"  Total transitions: {transitions}")
    if transition_errors:
        print(f"  Mean error at transition: {np.mean(transition_errors):.2f}%")
    
    # Pass/fail
    threshold = 10.0
    passed = sum(1 for e in all_errors if e < threshold)
    total = len(all_errors)
    
    print(f"\nPASS/FAIL (threshold = {threshold}% error):")
    print(f"  {passed}/{total} evaluations passed")
    
    if passed == total:
        print("\n🎉 ALL TIME-SERIES EVALUATIONS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} evaluation(s) exceeded threshold")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_time_series_simulation()
