#!/usr/bin/env python3
"""
Test Enhanced Unified IQFT
==========================

Demonstrates improvements from combining FB-QDP + qfdp_multiasset IQFT.

Tests:
1. Compare implementations (explicit vs library vs approximate)
2. Resource scaling analysis
3. Tensor IQFT for multi-asset
4. Fidelity validation

Author: QFDP Unified Team
Date: November 30, 2025
"""

import sys
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

print("="*70)
print("ENHANCED UNIFIED IQFT TEST")
print("="*70)
print()

# Import unified IQFT
sys.path.insert(0, '/Volumes/Hippocampus/QFDP')
from unified_qfdp.enhanced_iqft import (
    build_iqft_explicit,
    build_iqft_library,
    build_iqft_auto,
    apply_tensor_iqft,
    estimate_iqft_resources,
    estimate_tensor_iqft_resources,
    compare_iqft_implementations,
    global_phase_invariant_fidelity,
    IQFTConfig
)

# Test 1: Compare implementations
print("="*70)
print("TEST 1: Implementation Comparison")
print("="*70)
print()

for n in [4, 6, 8]:
    print(f"n = {n} qubits:")
    results = compare_iqft_implementations(n)
    
    for impl_name, impl_data in results.items():
        if impl_name == 'resources':
            print(f"\n  Resource estimates:")
            for k, v in impl_data.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {impl_data['name']}:")
            print(f"    Gates: {impl_data['gates']}, Depth: {impl_data['depth']}")
    print()

# Test 2: Resource scaling
print("="*70)
print("TEST 2: Resource Scaling (Exact vs Approximate)")
print("="*70)
print()

print("| n | H gates | Phase (exact) | Phase (approx=1) | T-count (exact) |")
print("|---|---------|---------------|------------------|-----------------|")

for n in [4, 8, 12, 16]:
    res_exact = estimate_iqft_resources(n, approximation_degree=0)
    res_approx = estimate_iqft_resources(n, approximation_degree=1)
    
    print(f"| {n:2d} | {res_exact.h_gates:7d} | "
          f"{res_exact.phase_gates:13d} | "
          f"{res_approx.phase_gates:16d} | "
          f"{res_exact.t_count_estimate:15d} |")

print()

# Test 3: Tensor IQFT for multi-asset
print("="*70)
print("TEST 3: Tensor IQFT (Multi-Asset Portfolio)")
print("="*70)
print()

N_assets = 5
n_qubits_per_asset = 6

print(f"Portfolio: {N_assets} assets, {n_qubits_per_asset} qubits each")
print()

tensor_resources = estimate_tensor_iqft_resources(
    N_assets, 
    n_qubits_per_asset,
    approximation_degree=0
)

print("Resources:")
print(f"  Total qubits: {N_assets * n_qubits_per_asset}")
print(f"  Total H gates: {tensor_resources['total_h_gates']}")
print(f"  Total phase gates: {tensor_resources['total_phase_gates']}")
print(f"  Total gates: {tensor_resources['total_gates']}")
print(f"  Sequential depth: {tensor_resources['sequential_depth']}")
print(f"  Parallel depth: {tensor_resources['parallel_depth']}")
print(f"  T-count (fault-tolerant): {tensor_resources['total_t_count']:,}")
print()

print("Advantage of parallel execution:")
print(f"  Depth reduction: {tensor_resources['sequential_depth'] / tensor_resources['parallel_depth']:.1f}×")
print()

# Test 4: Fidelity validation
print("="*70)
print("TEST 4: IQFT Correctness (Forward QFT → IQFT)")
print("="*70)
print()

n = 4
print(f"Testing with {n} qubits...")
print()

# Create a test state
test_circuit = QuantumCircuit(n)
test_circuit.h(0)
test_circuit.cx(0, 1)
test_circuit.ry(0.5, 2)
initial_state = Statevector(test_circuit)

# Apply QFT then IQFT (should return to initial state)
from qiskit.circuit.library import QFT

qft_circuit = test_circuit.copy()
qft = QFT(n, inverse=False).to_gate()
qft_circuit.append(qft, range(n))

# Now apply IQFT
iqft = build_iqft_library(n)
final_circuit = qft_circuit.copy()
final_circuit.compose(iqft, inplace=True)

final_state = Statevector(final_circuit)

# Compute fidelity
fidelity = global_phase_invariant_fidelity(
    initial_state.data,
    final_state.data
)

print(f"Initial state: {initial_state.data[:4]}...")
print(f"After QFT→IQFT: {final_state.data[:4]}...")
print(f"\nFidelity: {fidelity:.10f}")

if fidelity > 0.9999:
    print("✅ IQFT correctness validated (F > 0.9999)")
else:
    print(f"⚠️  Low fidelity: {fidelity}")

print()

# Test 5: Approximation impact
print("="*70)
print("TEST 5: Approximation Degree Impact")
print("="*70)
print()

n = 8
print(f"Testing with {n} qubits")
print()

# Test different approximation degrees
for approx_deg in [0, 1, 2, 3]:
    config = IQFTConfig(approximation_degree=approx_deg)
    iqft = build_iqft_library(n, config)
    
    # Apply to test state
    test_qc = QuantumCircuit(n)
    for i in range(n):
        test_qc.h(i)
    test_qc.compose(iqft, inplace=True)
    
    state = Statevector(test_qc)
    
    # Measure "quality" by state concentration
    probs = np.abs(state.data) ** 2
    concentration = np.max(probs)
    
    resources = estimate_iqft_resources(n, approx_deg)
    
    print(f"Approximation degree {approx_deg}:")
    print(f"  Phase gates: {resources.phase_gates}")
    print(f"  Depth: {resources.depth_approx}")
    print(f"  Max probability: {concentration:.6f}")
    print()

# Summary
print("="*70)
print("SUMMARY: Enhanced IQFT Capabilities")
print("="*70)
print()
print("✅ Multiple implementations (explicit, library, auto)")
print("✅ Approximation support (reduce gates/depth)")
print("✅ Tensor IQFT for multi-asset (parallel execution)")
print("✅ Resource estimation (gates, depth, T-count)")
print("✅ Fidelity validation utilities")
print()
print("Improvements over individual codebases:")
print("  - FB-QDP: Added approximation, tensor support, better resources")
print("  - qfdp_multiasset: Added explicit construction, comparison tools")
print("  - Both: Unified interface, automatic selection")
print()
print("="*70)
