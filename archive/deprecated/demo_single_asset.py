#!/usr/bin/env python3
"""
QFDP Single-Asset Option Pricing Demo
======================================

Demonstrates the complete workflow:
1. State preparation (log-normal distribution)
2. Payoff oracle (call option)
3. MLQAE pricing (amplitude estimation)

Run: python demo_single_asset.py
"""

import numpy as np
from qiskit import QuantumRegister

from qfdp_multiasset.state_prep import prepare_lognormal_asset
from qfdp_multiasset.oracles import apply_call_payoff_rotation
from qfdp_multiasset.mlqae import run_mlqae


def main():
    print("=" * 60)
    print("QFDP Single-Asset Call Option Pricing")
    print("=" * 60)
    
    # Option parameters (AAPL-like stock)
    S0 = 150.0      # Current price: $150
    r = 0.03        # Risk-free rate: 3%
    sigma = 0.25    # Volatility: 25%
    T = 1.0         # Maturity: 1 year
    K = 155.0       # Strike: $155 (3.3% OTM)
    
    n_qubits = 8    # 256 price points
    shots = 2000    # MLQAE measurements
    
    print(f"\nOption Parameters:")
    print(f"  Spot price (S0): ${S0:.2f}")
    print(f"  Strike (K): ${K:.2f}")
    print(f"  Volatility (σ): {sigma*100:.1f}%")
    print(f"  Risk-free rate (r): {r*100:.1f}%")
    print(f"  Time to maturity (T): {T:.1f} year")
    print(f"\nQuantum Parameters:")
    print(f"  Qubits: {n_qubits} (→ {2**n_qubits} price points)")
    print(f"  MLQAE shots: {shots}")
    
    # Step 1: Prepare log-normal asset distribution
    print(f"\n[1/3] Preparing quantum state...")
    circuit, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits=n_qubits)
    print(f"  ✓ Circuit depth: {circuit.depth()}")
    print(f"  ✓ Price range: [${prices.min():.2f}, ${prices.max():.2f}]")
    
    # Step 2: Encode call payoff on ancilla
    print(f"\n[2/3] Encoding call payoff oracle...")
    anc = QuantumRegister(1, 'ancilla')
    circuit.add_register(anc)
    scale = apply_call_payoff_rotation(circuit, circuit.qregs[0], anc[0], prices, K)
    print(f"  ✓ Max payoff (scale): ${scale:.2f}")
    print(f"  ✓ Total qubits: {circuit.num_qubits}")
    
    # Step 3: Price with MLQAE
    print(f"\n[3/3] Running MLQAE amplitude estimation...")
    result = run_mlqae(
        circuit, anc[0], scale,
        grover_powers=[0],  # k=0 (no Grover for now)
        shots_per_power=shots,
        seed=42
    )
    
    # Results
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  Call option price: ${result.price_estimate:.2f}")
    print(f"  Amplitude estimate: {result.amplitude_estimate:.4f}")
    print(f"  95% Confidence interval: [${result.confidence_interval[0]:.2f}, ${result.confidence_interval[1]:.2f}]")
    print(f"  Log-likelihood: {result.log_likelihood:.2f}")
    print(f"  Total measurements: {result.total_shots}")
    print(f"  Oracle queries: {result.oracle_queries}")
    
    # Classical benchmark (for comparison)
    print(f"\n{'=' * 60}")
    print(f"CLASSICAL BENCHMARK")
    print(f"{'=' * 60}")
    
    from qiskit.quantum_info import Statevector
    sv = Statevector(circuit)
    ancilla_idx = circuit.qubits.index(anc[0])
    prob_1 = sum(
        float((amp.conjugate() * amp).real)
        for i, amp in enumerate(sv.data)
        if (i >> ancilla_idx) & 1
    )
    classical_price = prob_1 * scale
    
    error = abs(result.price_estimate - classical_price)
    rel_error = error / classical_price
    
    print(f"  Exact price (statevector): ${classical_price:.2f}")
    print(f"  MLQAE error: ${error:.2f} ({rel_error*100:.1f}%)")
    
    if rel_error < 0.05:
        print(f"  ✓ Excellent accuracy!")
    elif rel_error < 0.10:
        print(f"  ✓ Good accuracy")
    else:
        print(f"  ⚠ Higher variance (try more shots)")
    
    print(f"\n{'=' * 60}")
    print(f"Demo complete! Foundation validated.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
