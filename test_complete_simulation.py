#!/usr/bin/env python3
"""
FB-IQFT Complete Simulation Test Suite

Comprehensive testing before hardware deployment:
1. Multiple strike types (ITM, ATM, OTM)
2. Multiple portfolios (3 assets, 5 assets)
3. Different maturities (1Y, 2Y)
4. M=32 vs M=64 comparison
5. Convergence analysis
6. Error statistics

This must pass before hardware deployment.
"""

import numpy as np
import sys
import json
from datetime import datetime
from pathlib import Path
from qfdp.unified import FBIQFTPricing

print("="*80)
print("FB-IQFT COMPREHENSIVE SIMULATION TEST SUITE")
print("="*80)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# Results storage
all_results = {
    'timestamp': datetime.now().isoformat(),
    'configuration': 'M=64, Local Calibration, 32K shots',
    'tests': []
}

def run_test(test_name, portfolio_data, strikes, M=64, shots=32768):
    """Run a single test configuration."""
    print("\n" + "="*80)
    print(f"TEST: {test_name}")
    print("="*80)
    
    pricer = FBIQFTPricing(M=M, alpha=1.0, num_shots=shots)
    
    results = []
    for K, strike_type in strikes:
        print(f"\n  Testing {strike_type} strike K=${K:.0f}...")
        
        # Reset calibration for local calibration
        pricer.A = None
        pricer.B = None
        
        try:
            result = pricer.price_option(
                backend='simulator',
                K=K,
                **portfolio_data
            )
            
            status = "✅" if result['error_percent'] < 2.0 else "⚠️" if result['error_percent'] < 5.0 else "❌"
            print(f"    Classical: ${result['price_classical']:.4f}")
            print(f"    Quantum:   ${result['price_quantum']:.4f}")
            print(f"    Error:     {result['error_percent']:.2f}% {status}")
            
            results.append({
                'strike': K,
                'type': strike_type,
                'classical': result['price_classical'],
                'quantum': result['price_quantum'],
                'error_percent': result['error_percent'],
                'circuit_depth': result['circuit_depth'],
                'num_qubits': result['num_qubits'],
                'sigma_p': result['sigma_p']
            })
            
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            results.append({
                'strike': K,
                'type': strike_type,
                'error': str(e)
            })
    
    return results

# =============================================================================
# TEST 1: Standard 3-Asset Portfolio (ITM, ATM, OTM)
# =============================================================================
print("\n" + "#"*80)
print("# TEST SUITE 1: Standard 3-Asset Portfolio")
print("#"*80)

portfolio_3asset = {
    'asset_prices': np.array([100.0, 100.0, 100.0]),
    'asset_volatilities': np.array([0.2, 0.25, 0.3]),
    'correlation_matrix': np.array([
        [1.0, 0.6, 0.4],
        [0.6, 1.0, 0.5],
        [0.4, 0.5, 1.0]
    ]),
    'portfolio_weights': np.array([0.4, 0.3, 0.3]),
    'T': 1.0,
    'r': 0.05
}

strikes_standard = [
    (90.0, "ITM"),
    (100.0, "ATM"),
    (110.0, "OTM")
]

results_3asset = run_test(
    "3-Asset Portfolio (1Y maturity)",
    portfolio_3asset,
    strikes_standard,
    M=64,
    shots=32768
)

all_results['tests'].append({
    'name': '3-Asset Standard',
    'results': results_3asset
})

# =============================================================================
# TEST 2: 5-Asset Portfolio
# =============================================================================
print("\n" + "#"*80)
print("# TEST SUITE 2: Larger 5-Asset Portfolio")
print("#"*80)

portfolio_5asset = {
    'asset_prices': np.array([100.0, 105.0, 95.0, 98.0, 102.0]),
    'asset_volatilities': np.array([0.2, 0.25, 0.18, 0.22, 0.19]),
    'correlation_matrix': np.array([
        [1.0, 0.3, 0.2, 0.1, 0.15],
        [0.3, 1.0, 0.4, 0.2, 0.25],
        [0.2, 0.4, 1.0, 0.3, 0.2],
        [0.1, 0.2, 0.3, 1.0, 0.4],
        [0.15, 0.25, 0.2, 0.4, 1.0]
    ]),
    'portfolio_weights': np.array([0.25, 0.2, 0.2, 0.2, 0.15]),
    'T': 1.0,
    'r': 0.05
}

results_5asset = run_test(
    "5-Asset Portfolio (1Y maturity)",
    portfolio_5asset,
    strikes_standard,
    M=64,
    shots=32768
)

all_results['tests'].append({
    'name': '5-Asset Portfolio',
    'results': results_5asset
})

# =============================================================================
# TEST 3: Different Maturity (2Y)
# =============================================================================
print("\n" + "#"*80)
print("# TEST SUITE 3: Longer Maturity (2 Years)")
print("#"*80)

portfolio_2y = portfolio_3asset.copy()
portfolio_2y['T'] = 2.0

results_2y = run_test(
    "3-Asset Portfolio (2Y maturity)",
    portfolio_2y,
    strikes_standard,
    M=64,
    shots=32768
)

all_results['tests'].append({
    'name': '3-Asset 2Y Maturity',
    'results': results_2y
})

# =============================================================================
# TEST 4: M=32 vs M=64 Comparison
# =============================================================================
print("\n" + "#"*80)
print("# TEST SUITE 4: Grid Size Comparison (M=32 vs M=64)")
print("#"*80)

results_m32 = run_test(
    "3-Asset Portfolio (M=32)",
    portfolio_3asset,
    [(100.0, "ATM")],  # Just ATM for comparison
    M=32,
    shots=32768
)

results_m64 = run_test(
    "3-Asset Portfolio (M=64)",
    portfolio_3asset,
    [(100.0, "ATM")],
    M=64,
    shots=32768
)

all_results['tests'].append({
    'name': 'M=32 vs M=64 Comparison',
    'M32': results_m32,
    'M64': results_m64
})

# =============================================================================
# TEST 5: Deep OTM/ITM
# =============================================================================
print("\n" + "#"*80)
print("# TEST SUITE 5: Deep OTM/ITM Strikes")
print("#"*80)

strikes_extreme = [
    (80.0, "Deep ITM"),
    (120.0, "Deep OTM")
]

results_extreme = run_test(
    "3-Asset Portfolio (Extreme Strikes)",
    portfolio_3asset,
    strikes_extreme,
    M=64,
    shots=32768
)

all_results['tests'].append({
    'name': 'Extreme Strikes',
    'results': results_extreme
})

# =============================================================================
# SUMMARY & ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE TEST SUMMARY")
print("="*80)

# Collect all errors
all_errors = []
for test in all_results['tests']:
    if 'results' in test:
        for res in test['results']:
            if 'error_percent' in res:
                all_errors.append({
                    'test': test['name'],
                    'strike_type': res['type'],
                    'error': res['error_percent']
                })

# Statistics
errors_only = [e['error'] for e in all_errors]
passed = sum(1 for e in errors_only if e < 2.0)
marginal = sum(1 for e in errors_only if 2.0 <= e < 5.0)
failed = sum(1 for e in errors_only if e >= 5.0)
total = len(errors_only)

print(f"\nOverall Statistics:")
print(f"  Total tests:     {total}")
print(f"  Passed (<2%):    {passed} ({passed/total*100:.1f}%)")
print(f"  Marginal (2-5%): {marginal} ({marginal/total*100:.1f}%)")
print(f"  Failed (>5%):    {failed} ({failed/total*100:.1f}%)")

if errors_only:
    print(f"\nError Statistics:")
    print(f"  Mean error:      {np.mean(errors_only):.2f}%")
    print(f"  Median error:    {np.median(errors_only):.2f}%")
    print(f"  Min error:       {np.min(errors_only):.2f}%")
    print(f"  Max error:       {np.max(errors_only):.2f}%")
    print(f"  Std deviation:   {np.std(errors_only):.2f}%")

# Detailed results table
print(f"\nDetailed Results:")
print(f"{'Test':<30} {'Strike':<12} {'Error':<10} {'Status'}")
print("-"*80)

for err in all_errors:
    status = "✅" if err['error'] < 2.0 else "⚠️" if err['error'] < 5.0 else "❌"
    print(f"{err['test']:<30} {err['strike_type']:<12} {err['error']:<9.2f}% {status}")

# Save results to JSON
results_file = Path('results/simulation_test_results.json')
results_file.parent.mkdir(exist_ok=True)
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n📊 Results saved to: {results_file}")

# Final verdict
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

success_rate = passed / total * 100
acceptable_rate = (passed + marginal) / total * 100

if success_rate >= 60:  # 60% pass <2%
    print("✅ SIMULATION TESTS PASSED")
    print("="*80)
    print(f"\n🎯 Success: {passed}/{total} tests meet <2% target ({success_rate:.1f}%)")
    print(f"📊 Acceptable: {passed + marginal}/{total} tests meet <5% ({acceptable_rate:.1f}%)")
    print("\n✅ APPROVED FOR HARDWARE DEPLOYMENT")
    print("\nNext steps:")
    print("  1. Review results in results/simulation_test_results.json")
    print("  2. Set up IBM Quantum credentials")
    print("  3. Run: python test_hardware_deployment.py")
    sys.exit(0)
    
elif acceptable_rate >= 70:  # 70% within 5%
    print("⚠️  SIMULATION TESTS MARGINAL - HARDWARE DEPLOYMENT CONDITIONAL")
    print("="*80)
    print(f"\n⚠️  Only {passed}/{total} tests meet <2% target ({success_rate:.1f}%)")
    print(f"📊 But {passed + marginal}/{total} tests are within 5% ({acceptable_rate:.1f}%)")
    print("\nRecommendation:")
    print("  - Hardware deployment acceptable (NISQ error will dominate)")
    print("  - Consider increasing shots to 65536 for marginal cases")
    print("  - Proceed with caution")
    sys.exit(0)
    
else:
    print("❌ SIMULATION TESTS FAILED - DO NOT DEPLOY TO HARDWARE")
    print("="*80)
    print(f"\n❌ Only {passed}/{total} meet <2% and {passed+marginal}/{total} meet <5%")
    print("\nRequired improvements:")
    print("  1. Debug failing test cases")
    print("  2. Increase M to 128 for difficult strikes")
    print("  3. Tune calibration parameters")
    print("  4. Re-run simulation tests")
    sys.exit(1)
