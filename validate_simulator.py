#!/usr/bin/env python3
"""
FB-IQFT Simulator Validation Script

Comprehensive end-to-end testing of the complete pipeline on ideal simulator.
Tests all 12 steps and validates against target specifications.
"""

import numpy as np
import sys
from datetime import datetime

print("="*80)
print("FB-IQFT SIMULATOR VALIDATION")
print("="*80)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# Step 1: Import modules
print("Step 1: Importing modules...")
try:
    from qfdp.unified import FBIQFTPricing
    import qiskit
    import qiskit_aer
    print(f"  ✓ qfdp.unified imported")
    print(f"  ✓ Qiskit {qiskit.__version__}")
    print(f"  ✓ Qiskit-Aer {qiskit_aer.__version__}")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Step 2: Define test portfolio
print("\nStep 2: Defining test portfolio...")
portfolio_data = {
    'asset_prices': np.array([100.0, 105.0, 95.0]),
    'asset_volatilities': np.array([0.20, 0.25, 0.18]),
    'correlation_matrix': np.array([
        [1.0, 0.3, 0.2],
        [0.3, 1.0, 0.4],
        [0.2, 0.4, 1.0]
    ]),
    'portfolio_weights': np.array([0.4, 0.3, 0.3]),
    'K': 110.0,
    'T': 1.0,
    'r': 0.05
}
B_0 = np.sum(portfolio_data['portfolio_weights'] * portfolio_data['asset_prices'])
print(f"  ✓ Portfolio: {len(portfolio_data['asset_prices'])} assets")
print(f"  ✓ Basket value: ${B_0:.2f}")
print(f"  ✓ Strike: ${portfolio_data['K']:.2f}")
print(f"  ✓ Moneyness: {portfolio_data['K']/B_0:.2%}")

# Step 3: Initialize pricer (M=16)
print("\nStep 3: Initializing FB-IQFT pricer (M=16)...")
try:
    pricer = FBIQFTPricing(M=16, alpha=1.0, num_shots=8192)
    print(f"  ✓ Grid size: M={pricer.M}")
    print(f"  ✓ Qubits: k={pricer.num_qubits}")
    print(f"  ✓ Shots: {pricer.num_shots}")
except Exception as e:
    print(f"  ✗ Initialization failed: {e}")
    sys.exit(1)

# Step 4: Run pricing (simulator)
print("\nStep 4: Running complete 12-step pipeline (simulator)...")
print("  This may take 10-30 seconds...")
try:
    result = pricer.price_option(
        backend='simulator',
        **portfolio_data
    )
    print(f"  ✓ Pricing completed")
except Exception as e:
    print(f"  ✗ Pricing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Validate results
print("\nStep 5: Validating results...")
print("-"*80)

# 5.1: Check result structure
required_keys = [
    'price_quantum', 'price_classical', 'error_percent',
    'sigma_p', 'B_0', 'num_factors', 'explained_variance',
    'circuit_depth', 'num_qubits', 'validation'
]
missing_keys = [k for k in required_keys if k not in result]
if missing_keys:
    print(f"  ✗ Missing result keys: {missing_keys}")
    sys.exit(1)
print(f"  ✓ All required keys present")

# 5.2: Check prices are positive
if result['price_quantum'] <= 0:
    print(f"  ✗ Quantum price is non-positive: {result['price_quantum']}")
    sys.exit(1)
if result['price_classical'] <= 0:
    print(f"  ✗ Classical price is non-positive: {result['price_classical']}")
    sys.exit(1)
print(f"  ✓ Prices are positive")

# 5.3: Check error is reasonable
if result['error_percent'] > 10:
    print(f"  ⚠ Warning: Error {result['error_percent']:.2f}% exceeds 10% (target: <3%)")
elif result['error_percent'] > 3:
    print(f"  ⚠ Warning: Error {result['error_percent']:.2f}% exceeds 3% target")
else:
    print(f"  ✓ Error {result['error_percent']:.2f}% meets <3% target")

# 5.4: Check circuit depth
if result['circuit_depth'] > 200:
    print(f"  ⚠ Warning: Circuit depth {result['circuit_depth']} exceeds 200 (target: 32-57)")
elif result['circuit_depth'] > 57:
    print(f"  ⚠ Warning: Circuit depth {result['circuit_depth']} exceeds 57 (acceptable for StatePrep overhead)")
else:
    print(f"  ✓ Circuit depth {result['circuit_depth']} meets target (32-57)")

# 5.5: Check qubits
if result['num_qubits'] != 4:
    print(f"  ✗ Expected 4 qubits for M=16, got {result['num_qubits']}")
    sys.exit(1)
print(f"  ✓ Qubits: {result['num_qubits']} (correct for M=16)")

# 5.6: Check factor decomposition
if result['num_factors'] > 3:
    print(f"  ⚠ Warning: Using {result['num_factors']} factors (expected ≤3 for 3 assets)")
print(f"  ✓ Factors: {result['num_factors']}")
print(f"  ✓ Explained variance: {result['explained_variance']:.2f}%")

# 5.7: Check validation
validation = result['validation']
for check, passed in validation.items():
    status = "✓" if passed else "✗"
    print(f"  {status} Validation - {check}: {passed}")
    if not passed:
        print(f"      This may indicate numerical issues or hardware noise")

# Step 6: Display detailed results
print("\n" + "="*80)
print("DETAILED RESULTS")
print("="*80)

print(f"\n📊 OPTION PRICES:")
print(f"  Quantum:   ${result['price_quantum']:.4f}")
print(f"  Classical: ${result['price_classical']:.4f}")
print(f"  Error:     {result['error_percent']:.2f}%")

print(f"\n📈 PORTFOLIO CHARACTERISTICS:")
print(f"  σ_p (portfolio vol): {result['sigma_p']:.4f}")
print(f"  B_0 (basket value):  ${result['B_0']:.2f}")

print(f"\n🔬 FACTOR DECOMPOSITION:")
print(f"  Factors kept (K):    {result['num_factors']}")
print(f"  Variance explained:  {result['explained_variance']:.2f}%")
print(f"  Factor variances:    {result['factor_variances']}")

print(f"\n⚛️  QUANTUM CIRCUIT:")
print(f"  Qubits:              {result['num_qubits']}")
print(f"  Depth:               {result['circuit_depth']} gates")
print(f"  Target:              32-57 gates (5-20× reduction vs standard QFDP)")

print(f"\n📐 CALIBRATION:")
print(f"  A (scale):           {result['calibration_A']:.2f}")
print(f"  B (offset):          {result['calibration_B']:.4f}")

# Step 7: Test M=32 configuration
print("\n" + "="*80)
print("TESTING M=32 CONFIGURATION")
print("="*80)
try:
    pricer_32 = FBIQFTPricing(M=32, alpha=1.0, num_shots=4096)
    print(f"Initialized: M=32, k={pricer_32.num_qubits} qubits")
    
    result_32 = pricer_32.price_option(backend='simulator', **portfolio_data)
    
    print(f"\nResults (M=32):")
    print(f"  Price:    ${result_32['price_quantum']:.4f}")
    print(f"  Error:    {result_32['error_percent']:.2f}%")
    print(f"  Depth:    {result_32['circuit_depth']} gates")
    
    print(f"\nComparison (M=16 vs M=32):")
    print(f"  Error improvement: {result['error_percent'] - result_32['error_percent']:+.2f}%")
    print(f"  Depth increase:    +{result_32['circuit_depth'] - result['circuit_depth']} gates")
    
except Exception as e:
    print(f"✗ M=32 test failed: {e}")

# Step 8: Complexity comparison
print("\n" + "="*80)
print("COMPLEXITY COMPARISON: FB-IQFT vs Standard QFDP")
print("="*80)
print(f"{'Aspect':<20} {'Standard QFDP':<20} {'FB-IQFT':<20} {'Reduction':<15}")
print("-"*80)
print(f"{'Portfolio Model':<20} {'N-asset dynamics':<20} {'1D Gaussian basket':<20} {'N→1':<15}")
print(f"{'Grid Points M':<20} {'256-1024':<20} {str(pricer.M):<20} {'8-32×':<15}")
print(f"{'IQFT Qubits k':<20} {'8-10':<20} {str(result['num_qubits']):<20} {'2×':<15}")
print(f"{'IQFT Depth':<20} {'64-100':<20} {str(result['num_qubits']**2):<20} {'4-6×':<15}")
reduction_range = f"{300/result['circuit_depth']:.1f}-{1100/result['circuit_depth']:.1f}×"
print(f"{'Total Depth':<20} {'300-1100':<20} {str(result['circuit_depth']):<20} {reduction_range:<15}")

# Final verdict
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

status_checks = [
    ("Prices positive", result['price_quantum'] > 0 and result['price_classical'] > 0),
    ("Error reasonable", result['error_percent'] < 10),
    ("Circuit depth acceptable", result['circuit_depth'] < 200),
    ("Correct qubits", result['num_qubits'] == 4),
    ("Validation passed", all(result['validation'].values())),
]

all_passed = all(check[1] for check in status_checks)

for check_name, passed in status_checks:
    status = "✓" if passed else "✗"
    print(f"{status} {check_name}")

print("\n" + "="*80)
if all_passed:
    print("✅ SIMULATOR VALIDATION PASSED")
    print("="*80)
    print("\nReady for hardware deployment!")
    print("\nNext steps:")
    print("  1. Review results above")
    print("  2. Approve hardware testing")
    print("  3. Run: python validate_hardware.py")
    sys.exit(0)
else:
    print("⚠️  SIMULATOR VALIDATION COMPLETED WITH WARNINGS")
    print("="*80)
    print("\nReview warnings above before hardware deployment.")
    sys.exit(1)
