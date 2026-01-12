"""
Historical Stress Scenarios Test
================================

Test QRC+QTC+FB-IQFT under historical market stress events:
1. 2008 Financial Crisis (ρ spike to 0.9, vol doubling)
2. COVID March 2020 (correlation/vol spike, sharp recovery)
3. Flash Crash 2010 (short-term extreme)
4. European Debt Crisis 2011-2012
5. China Devaluation 2015-2016

Uses realistic parameters derived from historical data.
"""

import numpy as np
from typing import Dict, List
import logging
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.fb_iqft_pricing import FBIQFTPricing
from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
from qfdp.integrated.corrected_qtc_integrated_pricer import CorrectedQTCIntegratedPricer

logging.basicConfig(level=logging.WARNING)


# ============================================================================
# HISTORICAL STRESS SCENARIOS
# ============================================================================

STRESS_SCENARIOS = {
    '2008_financial_crisis': {
        'description': '2008 Financial Crisis (Lehman collapse)',
        'date_range': 'Sep-Nov 2008',
        'rho_before': 0.35,
        'rho_during': 0.90,  # Correlations spiked to near 1
        'vol_multiplier': 2.5,  # VIX went from 20 to 80+
        'price_trend': 'down',  # Market dropped 40%+
        'price_drop_pct': 40,
    },
    'covid_march_2020': {
        'description': 'COVID-19 Market Crash',
        'date_range': 'Feb-Mar 2020',
        'rho_before': 0.40,
        'rho_during': 0.85,  # Correlation spike
        'vol_multiplier': 3.0,  # VIX hit 80+
        'price_trend': 'volatile',  # Extreme swings
        'price_drop_pct': 35,
    },
    'flash_crash_2010': {
        'description': 'Flash Crash (May 6, 2010)',
        'date_range': 'May 2010',
        'rho_before': 0.30,
        'rho_during': 0.95,  # Extreme correlation in minutes
        'vol_multiplier': 4.0,  # Intraday spike
        'price_trend': 'down',
        'price_drop_pct': 10,  # Quick recovery
    },
    'eu_debt_crisis_2011': {
        'description': 'European Debt Crisis',
        'date_range': 'Aug-Nov 2011',
        'rho_before': 0.35,
        'rho_during': 0.75,  # Elevated correlations
        'vol_multiplier': 1.8,
        'price_trend': 'volatile',
        'price_drop_pct': 20,
    },
    'china_devaluation_2015': {
        'description': 'China FX Devaluation / EM Crisis',
        'date_range': 'Aug 2015 - Feb 2016',
        'rho_before': 0.30,
        'rho_during': 0.65,
        'vol_multiplier': 1.6,
        'price_trend': 'down',
        'price_drop_pct': 15,
    },
    'vix_spike_2018': {
        'description': 'VIX Spike (Feb 2018, Volmageddon)',
        'date_range': 'Feb 2018',
        'rho_before': 0.25,
        'rho_during': 0.70,  # Quick correlation spike
        'vol_multiplier': 2.5,  # XIV blowup
        'price_trend': 'volatile',
        'price_drop_pct': 12,
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_correlation_matrix(n: int, rho: float) -> np.ndarray:
    """Generate N×N correlation matrix with uniform off-diagonal correlation."""
    return np.eye(n) * (1 - rho) + rho


def generate_stress_price_history(base_price: float, trend: str, drop_pct: float) -> np.ndarray:
    """
    Generate 6-point price history reflecting stress scenario.
    
    During stress:
    - 'down': Steady decline
    - 'volatile': Up and down swings
    """
    if trend == 'down':
        # Gradual decline then sharp drop
        return base_price * np.array([1.0, 0.97, 0.92, 0.85, 0.75, 1 - drop_pct/100])
    elif trend == 'volatile':
        # Whipsaw pattern
        return base_price * np.array([1.0, 0.95, 1.02, 0.88, 0.92, 1 - drop_pct/100])
    else:
        return np.full(6, base_price)


# ============================================================================
# STRESS TEST SUITE
# ============================================================================

def run_stress_scenario_tests():
    """
    Run comprehensive stress scenario tests.
    """
    print("=" * 100)
    print("HISTORICAL STRESS SCENARIO TESTS")
    print("=" * 100)
    print("Testing QRC+QTC+FB-IQFT under historical market stress events")
    print("=" * 100)
    
    # Initialize pricers
    fb_iqft = FBIQFTPricing(M=16, alpha=1.0)
    qrc_qtc_pricer = CorrectedQTCIntegratedPricer(fb_iqft, qrc_beta=0.1, qtc_gamma=0.05)
    
    # Portfolio sizes to test
    portfolio_sizes = [5, 10, 50]
    base_vol = 0.20
    
    results = []
    
    for scenario_name, scenario in STRESS_SCENARIOS.items():
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario['description']}")
        print(f"Date Range: {scenario['date_range']}")
        print(f"ρ: {scenario['rho_before']:.2f} → {scenario['rho_during']:.2f} | Vol: ×{scenario['vol_multiplier']}")
        print(f"{'='*80}")
        
        # Test at different portfolio sizes
        print(f"\n{'N Assets':<10} {'Morning σ_p':<15} {'Stress σ_p':<15} {'Price Before':<15} {'Price During':<15} {'Δ Price':<10}")
        print("-" * 80)
        
        for n in portfolio_sizes:
            asset_prices = np.full(n, 100.0)
            weights = np.ones(n) / n
            
            # BEFORE STRESS (morning calibration)
            corr_before = generate_correlation_matrix(n, scenario['rho_before'])
            asset_vols_before = np.full(n, base_vol)
            
            vol_matrix = np.diag(asset_vols_before)
            cov_before = vol_matrix @ corr_before @ vol_matrix
            sigma_p_before = float(np.sqrt(weights.T @ cov_before @ weights))
            
            # Price before stress
            price_before = price_call_option_corrected(100, 100, 1.0, 0.05, sigma_p_before)['price']
            
            # DURING STRESS (QRC+QTC should adapt)
            corr_during = generate_correlation_matrix(n, scenario['rho_during'])
            asset_vols_during = np.full(n, base_vol * scenario['vol_multiplier'])
            price_history = generate_stress_price_history(100.0, scenario['price_trend'], scenario['price_drop_pct'])
            
            # True stressed σ_p
            vol_matrix_stress = np.diag(asset_vols_during)
            cov_stress = vol_matrix_stress @ corr_during @ vol_matrix_stress
            sigma_p_stress = float(np.sqrt(weights.T @ cov_stress @ weights))
            
            # QRC+QTC pricing
            market_data = {
                'spot_prices': asset_prices,
                'volatilities': asset_vols_during,
                'correlation_matrix': corr_during,
                'weights': weights,
                'maturity': 1.0,
                'risk_free_rate': 0.05
            }
            
            qrc_result = qrc_qtc_pricer.price_with_full_quantum_pipeline(
                market_data, price_history, strike=100.0, use_quantum_circuit=True
            )
            
            price_during = qrc_result['price_quantum']
            price_change = (price_during / price_before - 1) * 100
            
            print(f"{n:<10} {sigma_p_before:<15.4f} {qrc_result['sigma_p_enhanced']:<15.4f} ${price_before:<14.2f} ${price_during:<14.2f} {price_change:+.1f}%")
            
            results.append({
                'scenario': scenario_name,
                'n': n,
                'sigma_p_before': sigma_p_before,
                'sigma_p_enhanced': qrc_result['sigma_p_enhanced'],
                'sigma_p_true_stress': sigma_p_stress,
                'price_before': price_before,
                'price_during': price_during,
                'price_change_pct': price_change
            })
    
    # Summary Analysis
    print("\n" + "=" * 100)
    print("STRESS TEST SUMMARY")
    print("=" * 100)
    
    # Price change by scenario severity
    print("\nPrice Change by Scenario (N=10 baseline):")
    for scenario_name, scenario in STRESS_SCENARIOS.items():
        scenario_results = [r for r in results if r['scenario'] == scenario_name and r['n'] == 10]
        if scenario_results:
            r = scenario_results[0]
            print(f"  {scenario['description'][:40]:<42} Price: +{r['price_change_pct']:.1f}%")
    
    # QRC+QTC adaptation quality
    print("\nQRC+QTC Volatility Adaptation (N=10):")
    for scenario_name, scenario in STRESS_SCENARIOS.items():
        scenario_results = [r for r in results if r['scenario'] == scenario_name and r['n'] == 10]
        if scenario_results:
            r = scenario_results[0]
            # How close is QRC+QTC enhanced sigma to true stress sigma?
            adaptation_error = abs(r['sigma_p_enhanced'] - r['sigma_p_true_stress']) / r['sigma_p_true_stress'] * 100
            print(f"  {scenario_name[:30]:<32} σ_enhanced: {r['sigma_p_enhanced']:.4f} vs True: {r['sigma_p_true_stress']:.4f} (Error: {adaptation_error:.1f}%)")
    
    # Worst case analysis
    print("\nWorst Case (2008 Financial Crisis, N=50):")
    worst_case = [r for r in results if r['scenario'] == '2008_financial_crisis' and r['n'] == 50]
    if worst_case:
        r = worst_case[0]
        print(f"  σ_p Before: {r['sigma_p_before']:.4f}")
        print(f"  σ_p Enhanced: {r['sigma_p_enhanced']:.4f}")
        print(f"  σ_p True Stress: {r['sigma_p_true_stress']:.4f}")
        print(f"  Price Change: {r['price_change_pct']:+.1f}%")
    
    print("\n" + "=" * 100)
    print("✅ STRESS SCENARIO TESTS COMPLETE")
    print("=" * 100)
    
    return results


if __name__ == "__main__":
    run_stress_scenario_tests()
