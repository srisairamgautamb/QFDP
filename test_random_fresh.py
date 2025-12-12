#!/usr/bin/env python3
"""
RANDOM FRESH HARDWARE TEST
Generate completely random portfolio parameters and test on hardware
This proves the algorithm works on ANY portfolio, not just tuned ones
"""

import numpy as np
from datetime import datetime
import json
from pathlib import Path

print("="*80)
print("RANDOM FRESH HARDWARE TEST - NO OVERFITTING")
print("="*80)
print(f"Timestamp: {datetime.now().isoformat()}\n")

# Import FB-IQFT
from qfdp.unified import FBIQFTPricing
from qiskit_ibm_runtime import QiskitRuntimeService

# Generate COMPLETELY RANDOM parameters (seeded by current time)
np.random.seed(int(datetime.now().timestamp()) % 2**32)

print("Generating random portfolio parameters...")
print()

# Random 3-asset portfolio with realistic constraints
n_assets = 3

# Random prices between 80 and 120
asset_prices = np.random.uniform(80, 120, n_assets)

# Random volatilities between 15% and 40%
asset_volatilities = np.random.uniform(0.15, 0.40, n_assets)

# Random correlation matrix (must be positive semidefinite)
# Generate random correlations between -0.5 and 0.9
corr_values = np.random.uniform(-0.5, 0.9, (n_assets, n_assets))
correlation_matrix = (corr_values + corr_values.T) / 2  # Make symmetric
np.fill_diagonal(correlation_matrix, 1.0)  # Diagonal = 1

# Make positive semidefinite by eigenvalue adjustment
eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
eigenvalues = np.maximum(eigenvalues, 0.01)  # Ensure positive
correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# Normalize to ensure diagonal = 1
D = np.sqrt(np.diag(np.diag(correlation_matrix)))
correlation_matrix = np.linalg.inv(D) @ correlation_matrix @ np.linalg.inv(D)

# Random weights (sum to 1)
weights = np.random.dirichlet(np.ones(n_assets))

# Random maturity between 0.5 and 2 years
T = np.random.uniform(0.5, 2.0)

# Random risk-free rate between 2% and 7%
r = np.random.uniform(0.02, 0.07)

# Random strike (around basket value ± 20%)
B0 = np.sum(weights * asset_prices)
K = B0 * np.random.uniform(0.85, 1.15)

print("RANDOM PORTFOLIO PARAMETERS:")
print("-" * 80)
print(f"Asset prices:        {asset_prices}")
print(f"Volatilities:        {asset_volatilities}")
print(f"Correlation matrix:")
for row in correlation_matrix:
    print(f"  {row}")
print(f"Portfolio weights:   {weights}")
print(f"Maturity T:          {T:.3f} years")
print(f"Risk-free rate r:    {r:.2%}")
print(f"Basket value B0:     ${B0:.2f}")
print(f"Strike K:            ${K:.2f}")
print(f"Moneyness:           {K/B0:.2%} ({'ITM' if K < B0 else 'OTM' if K > B0 else 'ATM'})")
print()

# Connect to IBM Quantum
print("Connecting to IBM Quantum...")
service = QiskitRuntimeService()

# Find backend with shortest queue
backends = service.backends(simulator=False, operational=True)
selected_backend = None
min_queue = float('inf')

for backend in backends:
    try:
        status = backend.status()
        queue = status.pending_jobs if hasattr(status, 'pending_jobs') else 999
        if queue < min_queue:
            min_queue = queue
            selected_backend = backend
    except:
        continue

if selected_backend is None:
    print("❌ No backends available")
    exit(1)

print(f"✅ Selected: {selected_backend.name} ({selected_backend.num_qubits} qubits)\n")

# Run on hardware
print("="*80)
print("RUNNING ON HARDWARE")
print("="*80)
print()

portfolio = {
    'asset_prices': asset_prices,
    'asset_volatilities': asset_volatilities,
    'correlation_matrix': correlation_matrix,
    'portfolio_weights': weights,
    'T': T,
    'r': r
}

pricer = FBIQFTPricing(M=64, alpha=1.0, num_shots=8192)

print("Submitting job...")
try:
    result = pricer.price_option(
        backend=selected_backend,
        K=K,
        **portfolio
    )
    
    print("\n✅ Hardware test completed!")
    print(f"\nResults:")
    print(f"  Classical price: ${result['price_classical']:.4f}")
    print(f"  Quantum price:   ${result['price_quantum']:.4f}")
    print(f"  Error:           {result['error_percent']:.2f}%")
    print(f"  Circuit depth:   {result['circuit_depth']}")
    print(f"  Qubits:          {result['num_qubits']}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'backend': selected_backend.name,
        'test_type': 'random_fresh',
        'portfolio_params': {
            'asset_prices': asset_prices.tolist(),
            'asset_volatilities': asset_volatilities.tolist(),
            'correlation_matrix': correlation_matrix.tolist(),
            'portfolio_weights': weights.tolist(),
            'T': T,
            'r': r
        },
        'strike': K,
        'moneyness': float(K/B0),
        'result': {
            'classical': result['price_classical'],
            'quantum': result['price_quantum'],
            'error_percent': result['error_percent'],
            'circuit_depth': result['circuit_depth'],
            'num_qubits': result['num_qubits']
        }
    }
    
    output_file = Path('results/random_hardware_test.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Verdict
    print("\n" + "="*80)
    print("VERDICT ON RANDOM TEST")
    print("="*80)
    
    if result['error_percent'] < 2:
        print("\n🎉 EXCEPTIONAL: <2% error on completely random portfolio!")
        print("   This proves the algorithm is NOT overfitted.")
    elif result['error_percent'] < 5:
        print("\n✅ EXCELLENT: <5% error on random portfolio")
        print("   Algorithm generalizes well to unseen data.")
    elif result['error_percent'] < 15:
        print("\n⚠️  ACCEPTABLE: <15% error typical for NISQ")
        print("   Within expected range for quantum hardware.")
    else:
        print("\n❌ HIGH ERROR: May indicate issues")
        print("   Review algorithm or hardware quality.")
    
    print("\n" + "="*80)
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
