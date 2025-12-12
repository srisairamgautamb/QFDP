#!/usr/bin/env python3
"""
Fresh Hardware Validation Test
Run FB-IQFT on hardware from scratch to validate all results
"""

import numpy as np
from datetime import datetime
import json
from pathlib import Path

print("="*80)
print("FRESH FB-IQFT HARDWARE VALIDATION")
print("="*80)
print(f"Timestamp: {datetime.now().isoformat()}\n")

# Import FB-IQFT
from qfdp.unified import FBIQFTPricing

# Connect to IBM Quantum
from qiskit_ibm_runtime import QiskitRuntimeService

print("Connecting to IBM Quantum...")
service = QiskitRuntimeService()

# Find backend with shortest queue
print("\nChecking backend availability...")
backends = service.backends(simulator=False, operational=True)

selected_backend = None
min_queue = float('inf')

for backend in backends:
    try:
        status = backend.status()
        queue = status.pending_jobs if hasattr(status, 'pending_jobs') else 999
        print(f"  {backend.name}: {queue} jobs in queue")
        if queue < min_queue:
            min_queue = queue
            selected_backend = backend
    except:
        continue

if selected_backend is None:
    print("❌ No backends available")
    exit(1)

print(f"\n✅ Selected: {selected_backend.name} ({selected_backend.num_qubits} qubits, {min_queue} in queue)\n")

# Test portfolio
portfolio = {
    'asset_prices': np.array([100.0, 100.0, 100.0]),
    'asset_volatilities': np.array([0.20, 0.25, 0.30]),
    'correlation_matrix': np.array([
        [1.0, 0.6, 0.4],
        [0.6, 1.0, 0.5],
        [0.4, 0.5, 1.0]
    ]),
    'portfolio_weights': np.array([0.4, 0.3, 0.3]),
    'T': 1.0,
    'r': 0.05
}

# Test strikes
strikes_to_test = [
    (90.0, 'ITM'),
    (100.0, 'ATM'),
    (110.0, 'OTM')
]

print("="*80)
print("RUNNING FRESH HARDWARE TESTS")
print("="*80)
print()

# Initialize pricer
pricer = FBIQFTPricing(M=64, alpha=1.0, num_shots=8192)

results = {
    'timestamp': datetime.now().isoformat(),
    'backend': selected_backend.name,
    'num_qubits': selected_backend.num_qubits,
    'strikes': {}
}

for K, strike_type in strikes_to_test:
    print(f"\n{'-'*80}")
    print(f"Testing {strike_type} (K=${K:.0f})")
    print(f"{'-'*80}")
    
    # Reset calibration
    pricer.A = None
    pricer.B = None
    
    print(f"Submitting to {selected_backend.name}...")
    
    try:
        result = pricer.price_option(
            backend=selected_backend,
            K=K,
            **portfolio
        )
        
        print(f"\n✅ {strike_type} completed!")
        print(f"  Classical price: ${result['price_classical']:.4f}")
        print(f"  Quantum price:   ${result['price_quantum']:.4f}")
        print(f"  Error:           {result['error_percent']:.2f}%")
        print(f"  Circuit depth:   {result['circuit_depth']}")
        print(f"  Qubits:          {result['num_qubits']}")
        
        results['strikes'][strike_type] = {
            'K': K,
            'classical': result['price_classical'],
            'quantum': result['price_quantum'],
            'error_percent': result['error_percent'],
            'circuit_depth': result['circuit_depth'],
            'num_qubits': result['num_qubits'],
            'sigma_p': result['sigma_p']
        }
        
    except Exception as e:
        print(f"\n❌ {strike_type} failed: {e}")
        import traceback
        traceback.print_exc()
        results['strikes'][strike_type] = {
            'K': K,
            'status': 'failed',
            'error': str(e)
        }

# Save results
output_file = Path('results/fresh_hardware_validation.json')
output_file.parent.mkdir(exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "="*80)
print("FRESH VALIDATION SUMMARY")
print("="*80)
print()

errors = []
for strike_type in ['ITM', 'ATM', 'OTM']:
    if strike_type in results['strikes'] and 'error_percent' in results['strikes'][strike_type]:
        error = results['strikes'][strike_type]['error_percent']
        errors.append(error)
        status = "🌟" if error < 1 else ("✅" if error < 3 else "⚠️")
        K = results['strikes'][strike_type]['K']
        print(f"{strike_type} (K=${K:.0f}): {error:.2f}% {status}")

if errors:
    print(f"\nMean error: {np.mean(errors):.2f}%")
    print(f"Min error:  {np.min(errors):.2f}%")
    print(f"Max error:  {np.max(errors):.2f}%")
    
    if np.mean(errors) < 1:
        print("\n🎉 EXCEPTIONAL: Sub-1% mean hardware error validated!")
    elif np.mean(errors) < 5:
        print("\n✅ EXCELLENT: All strikes <5% error")

print(f"\n✅ Results saved to: {output_file}")
print("="*80)
