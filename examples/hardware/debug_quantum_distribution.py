"""
Debug Quantum State Distribution
=================================

Check if the quantum state is actually sampling from a Gaussian distribution.
"""

import numpy as np
import sys
import os
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qfdp.fb_iqft.pricing_v2 import prepare_factor_distribution_state
from qiskit.primitives import StatevectorSampler

# Test parameters
n_qubits = 4
shots = 10000

print("="*70)
print("QUANTUM STATE DISTRIBUTION DIAGNOSTIC")
print("="*70)
print()

# Prepare the quantum state
qc, factor_grid = prepare_factor_distribution_state(n_qubits)

print(f"Factor grid: {len(factor_grid)} points")
print(f"Range: [{factor_grid[0]:.2f}, {factor_grid[-1]:.2f}]")
print()

# Add measurements
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
qreg = QuantumRegister(n_qubits, 'q')
creg = ClassicalRegister(n_qubits, 'c')
qc_test = QuantumCircuit(qreg, creg)
qc_test.compose(qc, inplace=True)
qc_test.measure(qreg, creg)

# Sample
sampler = StatevectorSampler()
job = sampler.run([qc_test], shots=shots)
result = job.result()
counts = result[0].data.c.get_counts()

print(f"Sampled {shots} times, got {len(counts)} unique outcomes")
print()

# Analyze distribution
print("MEASUREMENT DISTRIBUTION:")
print("-" * 70)
print(f"{'Index':<6} {'Factor':<8} {'Counts':<8} {'Prob':<8} {'Expected':<8} {'Error'}")
print("-" * 70)

# Expected Gaussian probabilities
expected_probs = norm.pdf(factor_grid, 0, 1)
expected_probs /= expected_probs.sum()

total_counts = sum(counts.values())
measured_probs = np.zeros(len(factor_grid))

for bitstring, count in counts.items():
    idx = int(bitstring, 2)  # Qiskit bitstrings are already in correct order
    measured_probs[idx] = count / total_counts

# Show top outcomes
for idx in range(len(factor_grid)):
    factor_val = factor_grid[idx]
    measured = measured_probs[idx]
    expected = expected_probs[idx]
    error_pct = abs(measured - expected) / expected * 100 if expected > 0 else 0
    
    count = int(measured * total_counts)
    if count > 0:  # Only show non-zero outcomes
        print(f"{idx:<6} {factor_val:>7.2f} {count:<8} {measured*100:>6.2f}% {expected*100:>6.2f}% {error_pct:>6.1f}%")

print("-" * 70)

# Compute statistics
measured_mean = np.sum(factor_grid * measured_probs)
measured_std = np.sqrt(np.sum((factor_grid - measured_mean)**2 * measured_probs))

expected_mean = 0.0
expected_std = 1.0

print()
print("STATISTICS:")
print(f"  Measured mean: {measured_mean:.4f} (expected: {expected_mean:.4f})")
print(f"  Measured std:  {measured_std:.4f} (expected: {expected_std:.4f})")
print()

# Portfolio pricing test
print("="*70)
print("PORTFOLIO PRICING TEST")
print("="*70)

# Simple test case
spot = 100.0
strike = 105.0
rate = 0.05
maturity = 1.0
portfolio_vol = 0.141  # From actual test

print(f"Spot: ${spot}, Strike: ${strike}")
print(f"Portfolio vol: {portfolio_vol:.4f}")
print()

# Compute expected payoff using measured distribution
expected_payoff_quantum = 0.0
for idx, prob in enumerate(measured_probs):
    if prob > 0:
        factor_value = factor_grid[idx]
        drift = (rate - 0.5 * portfolio_vol**2) * maturity
        diffusion = portfolio_vol * np.sqrt(maturity) * factor_value
        portfolio_value = spot * np.exp(drift + diffusion)
        payoff = max(portfolio_value - strike, 0)
        expected_payoff_quantum += prob * payoff

# Compute using correct Gaussian
expected_payoff_correct = 0.0
for idx, prob in enumerate(expected_probs):
    factor_value = factor_grid[idx]
    drift = (rate - 0.5 * portfolio_vol**2) * maturity
    diffusion = portfolio_vol * np.sqrt(maturity) * factor_value
    portfolio_value = spot * np.exp(drift + diffusion)
    payoff = max(portfolio_value - strike, 0)
    expected_payoff_correct += prob * payoff

discount = np.exp(-rate * maturity)
price_quantum = discount * expected_payoff_quantum
price_correct = discount * expected_payoff_correct

print(f"Quantum state price:  ${price_quantum:.4f}")
print(f"True Gaussian price:  ${price_correct:.4f}")
print(f"Difference: ${abs(price_quantum - price_correct):.4f} ({abs(price_quantum - price_correct)/price_correct*100:.1f}%)")
print()

# Classical MC for reference
from qfdp.fb_iqft.pricing_v2 import _classical_mc_reference
weights = np.ones(5) / 5
vols = np.array([0.20, 0.25, 0.18, 0.22, 0.19])
corr = np.array([
    [1.0, 0.5, 0.3, 0.2, 0.1],
    [0.5, 1.0, 0.4, 0.3, 0.2],
    [0.3, 0.4, 1.0, 0.5, 0.3],
    [0.2, 0.3, 0.5, 1.0, 0.4],
    [0.1, 0.2, 0.3, 0.4, 1.0]
])
classical_price = _classical_mc_reference(weights, vols, corr, spot, strike, rate, maturity)

print(f"Classical MC price:   ${classical_price:.4f}")
print()

print("="*70)
print("CONCLUSION:")
print("="*70)
if abs(price_quantum - price_correct) / price_correct < 0.05:
    print("✓ Quantum state is correctly encoding Gaussian distribution")
    print("  Problem must be elsewhere in the pricing logic")
else:
    print("✗ Quantum state is NOT correctly sampling from Gaussian")
    print("  StatePreparation may not be working as expected")
print("="*70)
