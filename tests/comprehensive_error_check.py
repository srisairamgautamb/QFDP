"""
Comprehensive Error Check - All Errors Must Be < 2%
"""
import numpy as np
import sys
sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.fb_iqft_pricing import FBIQFTPricing
from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
from qfdp.integrated.corrected_qtc_integrated_pricer import CorrectedQTCIntegratedPricer

import logging
logging.basicConfig(level=logging.WARNING)

def generate_correlation_matrix(n, rho):
    return np.eye(n) * (1 - rho) + rho

def run_comprehensive_error_check():
    print('='*80)
    print('COMPREHENSIVE ERROR CHECK - TARGET: ALL ERRORS < 2%')
    print('='*80)
    
    # Initialize with optimized params
    fb_iqft = FBIQFTPricing(M=16, alpha=1.0)
    pricer = CorrectedQTCIntegratedPricer(fb_iqft, qrc_beta=0.01, qtc_gamma=0.018)
    
    # Test scenarios
    scenarios = [
        # (N, rho, trend, label)
        (2, 0.3, 'up', 'N=2, Calm'),
        (2, 0.7, 'down', 'N=2, High'),
        (5, 0.3, 'up', 'N=5, Calm'),
        (5, 0.5, 'flat', 'N=5, Medium'),
        (5, 0.7, 'volatile', 'N=5, High'),
        (5, 0.85, 'down', 'N=5, Crisis'),
        (10, 0.3, 'up', 'N=10, Calm'),
        (10, 0.5, 'flat', 'N=10, Medium'),
        (10, 0.7, 'volatile', 'N=10, High'),
        (10, 0.85, 'down', 'N=10, Crisis'),
        (50, 0.5, 'flat', 'N=50, Medium'),
        (50, 0.85, 'down', 'N=50, Crisis'),
    ]
    
    trends = {
        'up': np.array([-2, -1, 0, 1, 2, 3]),
        'down': np.array([3, 2, 1, -1, -3, -5]),
        'volatile': np.array([0, 3, -2, 4, -3, 2]),
        'flat': np.array([0, 0.1, -0.1, 0.2, 0, 0.1])
    }
    
    results = []
    print()
    print(f'{"Scenario":<20} {"True σ_p":<12} {"Enhanced σ_p":<12} {"σ Error":<10} {"True Price":<12} {"QRC Price":<12} {"P Error":<10} {"Status":<8}')
    print('-'*100)
    
    for n, rho, trend, label in scenarios:
        S = 100.0
        base_vol = 0.20
        
        asset_prices = np.full(n, S)
        asset_vols = np.full(n, base_vol)
        weights = np.ones(n) / n
        corr = generate_correlation_matrix(n, rho)
        
        # True σ_p
        vol_matrix = np.diag(asset_vols)
        cov = vol_matrix @ corr @ vol_matrix
        sigma_p_true = float(np.sqrt(weights.T @ cov @ weights))
        
        # True price
        price_true = price_call_option_corrected(S, 100.0, 1.0, 0.05, sigma_p_true)['price']
        
        # QRC+QTC price
        market_data = {
            'spot_prices': asset_prices,
            'volatilities': asset_vols,
            'correlation_matrix': corr,
            'weights': weights,
            'maturity': 1.0,
            'risk_free_rate': 0.05
        }
        price_history = S + trends[trend]
        
        result = pricer.price_with_full_quantum_pipeline(
            market_data, price_history, strike=100.0, use_quantum_circuit=True
        )
        
        sigma_error = abs(result['sigma_p_enhanced'] - sigma_p_true) / sigma_p_true * 100
        price_error = abs(result['price_quantum'] - price_true) / price_true * 100
        
        status = '✅ PASS' if price_error < 2.0 else '❌ FAIL'
        
        results.append({
            'scenario': label,
            'sigma_error': sigma_error,
            'price_error': price_error,
            'passed': price_error < 2.0
        })
        
        print(f'{label:<20} {sigma_p_true:<12.4f} {result["sigma_p_enhanced"]:<12.4f} {sigma_error:<9.2f}% ${price_true:<11.4f} ${result["price_quantum"]:<11.4f} {price_error:<9.2f}% {status}')
    
    # Summary
    print()
    print('='*80)
    print('SUMMARY')
    print('='*80)
    
    all_passed = all(r['passed'] for r in results)
    passed_count = sum(1 for r in results if r['passed'])
    total = len(results)
    
    sigma_errors = [r['sigma_error'] for r in results]
    price_errors = [r['price_error'] for r in results]
    
    print(f'\nσ_p Errors:')
    print(f'  Mean: {np.mean(sigma_errors):.2f}%')
    print(f'  Max:  {np.max(sigma_errors):.2f}%')
    
    print(f'\nPrice Errors:')
    print(f'  Mean: {np.mean(price_errors):.2f}%')
    print(f'  Max:  {np.max(price_errors):.2f}%')
    
    print(f'\nPass Rate: {passed_count}/{total}')
    
    if all_passed:
        print('\n🎉 ALL ERRORS ARE BELOW 2%!')
    else:
        failed = [r for r in results if not r['passed']]
        print(f'\n⚠️  {len(failed)} scenario(s) exceeded 2% error:')
        for r in failed:
            print(f'   - {r["scenario"]}: {r["price_error"]:.2f}%')
    
    print('='*80)

if __name__ == '__main__':
    run_comprehensive_error_check()
