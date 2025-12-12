#!/usr/bin/env python3
"""
FB-IQFT Improved Validation Script

Tests with enhanced settings:
- M=64 (instead of 16)
- Robust calibration
- 32K shots (instead of 8K)
- ATM, ITM, OTM strikes

Target: <2% error
"""

import numpy as np
import sys
from datetime import datetime

# Import improved modules
from qfdp.unified.carr_madan_gaussian import compute_characteristic_function, apply_carr_madan_transform
from qfdp.unified.carr_madan_improved import (
    setup_fourier_grid_adaptive,
    classical_fft_baseline_improved,
    calibrate_robust
)
from qfdp.unified.frequency_encoding import encode_frequency_state
from qfdp.unified.iqft_application import apply_iqft, extract_strike_amplitudes
from qfdp.unified.calibration import reconstruct_option_prices, validate_prices

print("="*80)
print("FB-IQFT IMPROVED VALIDATION")
print("="*80)
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Configuration: M=64, Shots=32768, Robust Calibration")
print()

# Test portfolio
asset_prices = np.array([100.0, 100.0, 100.0])
asset_volatilities = np.array([0.2, 0.25, 0.3])
correlation_matrix = np.array([
    [1.0, 0.6, 0.4],
    [0.6, 1.0, 0.5],
    [0.4, 0.5, 1.0]
])
portfolio_weights = np.array([0.4, 0.3, 0.3])

# Option parameters
T = 1.0
r = 0.05

# Compute portfolio characteristics
cov = np.outer(asset_volatilities, asset_volatilities) * correlation_matrix
sigma_p = np.sqrt(portfolio_weights @ cov @ portfolio_weights)
B_0 = np.sum(portfolio_weights * asset_prices)

print(f"Portfolio:")
print(f"  Basket value B_0: ${B_0:.2f}")
print(f"  Portfolio vol σ_p: {sigma_p:.4f}")
print()

# Configuration
M = 64
num_qubits = int(np.log2(M))
alpha = 1.0
num_shots = 32768

print(f"Quantum Parameters:")
print(f"  Grid size M: {M}")
print(f"  Qubits k: {num_qubits}")
print(f"  Shots: {num_shots}")
print()

def price_option_improved(K_target):
    """Price option using improved pipeline."""
    
    # PHASE 1: CLASSICAL PREPROCESSING
    print(f"\nPricing K=${K_target:.2f} (moneyness: {K_target/B_0:.2%})")
    print("-"*70)
    
    # PHASE 2: CARR-MADAN SETUP (IMPROVED)
    print("  [1] Setup adaptive Fourier grid...")
    u_grid, k_grid, delta_u, delta_k = setup_fourier_grid_adaptive(
        M, sigma_p, T, B_0, r, alpha, K_target=K_target
    )
    
    print("  [2] Compute characteristic function...")
    phi_values = compute_characteristic_function(u_grid, r, sigma_p, T)
    
    print("  [3] Apply Carr-Madan transform...")
    psi_values = apply_carr_madan_transform(u_grid, r, sigma_p, T, alpha)
    
    print("  [4] Classical FFT baseline (improved)...")
    C_classical = classical_fft_baseline_improved(psi_values, alpha, delta_u, k_grid)
    
    # Validate classical prices
    forward_price = B_0 * np.exp(r * T)
    if not (np.all(C_classical >= -1e-8) and np.all(C_classical <= forward_price * 1.01)):
        print("  ⚠️  Warning: Classical prices violate bounds")
    
    # PHASE 3: QUANTUM COMPUTATION
    print("  [5] Encode quantum state...")
    circuit, norm_factor = encode_frequency_state(psi_values, num_qubits)
    
    print("  [6] Apply IQFT...")
    circuit = apply_iqft(circuit, num_qubits)
    
    print(f"  [7] Measure ({num_shots} shots)...")
    quantum_probs = extract_strike_amplitudes(circuit, num_shots, backend='simulator')
    
    # PHASE 4: POST-PROCESSING (IMPROVED)
    print("  [8] Robust calibration...")
    A, B = calibrate_robust(quantum_probs, C_classical, k_grid, method='robust')
    print(f"      A={A:.4f}, B={B:.4f}")
    
    print("  [9] Reconstruct prices...")
    option_prices_quantum = reconstruct_option_prices(quantum_probs, A, B, k_grid, B_0)
    
    # Find price at target strike
    target_idx = np.argmin(np.abs(k_grid - np.log(K_target / B_0)))
    price_quantum = option_prices_quantum[target_idx]
    price_classical = C_classical[target_idx]
    
    error_percent = abs(price_quantum - price_classical) / price_classical * 100 if price_classical > 1e-10 else np.inf
    
    # Validation
    validation = validate_prices(option_prices_quantum, k_grid, B_0, r, T, tol=0.05)
    
    print(f"\n  Results:")
    print(f"    Classical: ${price_classical:.4f}")
    print(f"    Quantum:   ${price_quantum:.4f}")
    print(f"    Error:     {error_percent:.2f}%")
    print(f"    Circuit:   depth={circuit.depth()}, qubits={num_qubits}")
    
    return {
        'K': K_target,
        'price_classical': price_classical,
        'price_quantum': price_quantum,
        'error_percent': error_percent,
        'calibration_A': A,
        'calibration_B': B,
        'validation': validation,
        'circuit_depth': circuit.depth(),
        'num_qubits': num_qubits
    }

# Test 1: ITM option
print("\n" + "="*80)
print("TEST 1: 10% ITM OPTION")
print("="*80)
result_itm = price_option_improved(K_target=90.0)

# Test 2: ATM option
print("\n" + "="*80)
print("TEST 2: ATM OPTION")
print("="*80)
result_atm = price_option_improved(K_target=100.0)

# Test 3: OTM option
print("\n" + "="*80)
print("TEST 3: 10% OTM OPTION")
print("="*80)
result_otm = price_option_improved(K_target=110.0)

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n{'Strike':<15} {'Type':<8} {'Classical':<12} {'Quantum':<12} {'Error':<10} {'Status'}")
print("-"*80)

def status(error):
    if error < 2.0:
        return "✅ PASS"
    elif error < 5.0:
        return "⚠️  MARGINAL"
    else:
        return "❌ FAIL"

results = [result_itm, result_atm, result_otm]
types = ['ITM', 'ATM', 'OTM']

for res, typ in zip(results, types):
    print(f"K={res['K']:<11.0f} {typ:<8} ${res['price_classical']:<10.4f} ${res['price_quantum']:<10.4f} {res['error_percent']:<9.2f}% {status(res['error_percent'])}")

# Overall assessment
print("\n" + "="*80)
print("VALIDATION ASSESSMENT")
print("="*80)

all_pass = all(r['error_percent'] < 2.0 for r in results)
any_pass = any(r['error_percent'] < 2.0 for r in results)

print(f"\nCircuit Properties:")
print(f"  Grid size M: {M}")
print(f"  Qubits k: {num_qubits}")
print(f"  Circuit depth: {result_atm['circuit_depth']} (composite)")
print(f"  Portfolio σ_p: {sigma_p:.4f}")

print(f"\nCalibration:")
print(f"  ITM: A={result_itm['calibration_A']:.2f}, B={result_itm['calibration_B']:.4f}")
print(f"  ATM: A={result_atm['calibration_A']:.2f}, B={result_atm['calibration_B']:.4f}")
print(f"  OTM: A={result_otm['calibration_A']:.2f}, B={result_otm['calibration_B']:.4f}")

print("\n" + "="*80)
if all_pass:
    print("✅ ALL TESTS PASSED (<2% error)")
    print("="*80)
    print("\nReady for hardware deployment!")
    sys.exit(0)
elif any_pass:
    print("⚠️  PARTIAL SUCCESS")
    print("="*80)
    print("\nSome strikes meet <2% target. Consider:")
    print("  - Increase M to 128 for OTM options")
    print("  - Adjust damping parameter α")
    sys.exit(0)
else:
    print("❌ TESTS DID NOT MEET <2% TARGET")
    print("="*80)
    print("\nNext steps:")
    print("  1. Try M=128 (7 qubits)")
    print("  2. Check classical FFT accuracy vs Black-Scholes")
    print("  3. Implement Richardson extrapolation")
    sys.exit(1)
