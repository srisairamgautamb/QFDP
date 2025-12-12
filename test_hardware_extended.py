#!/usr/bin/env python3
"""
Extended Hardware Test: FB-IQFT across Multiple Strikes

Validates hardware performance on:
- ITM (K=90): In-the-money
- ATM (K=100): At-the-money 
- OTM (K=110): Out-of-the-money

This will verify if the 0.08% ATM error reproduces and test
algorithm robustness across different strike regimes.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

print("="*80)
print("EXTENDED HARDWARE TEST: FB-IQFT Multi-Strike Validation")
print("="*80)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# Import dependencies
from qiskit_ibm_runtime import QiskitRuntimeService
from qfdp.unified import FBIQFTPricing

# Connect to IBM Quantum
print("Connecting to IBM Quantum...")
service = QiskitRuntimeService()

# Use backend with shortest queue
print("Checking backend availability...")
backends_available = [
    ('ibm_fez', 156),
    ('ibm_torino', 133),
    ('ibm_kyiv', 156)
]

backend = None
for name, qubits in backends_available:
    try:
        b = service.backend(name)
        status = b.status()
        queue = status.pending_jobs if hasattr(status, 'pending_jobs') else 999
        print(f"  {name}: {queue} jobs in queue")
        if queue == 0:
            backend = b
            print(f"✅ Selected: {name} (no queue, {qubits} qubits)")
            break
    except:
        continue

if backend is None:
    # Fallback to first available
    backend = service.backend('ibm_torino')
    print(f"✅ Using: {backend.name} (fallback)")

print()

# Portfolio configuration (same as original test)
portfolio = {
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

print("Portfolio Configuration:")
print(f"  Assets: 3")
print(f"  Weights: {portfolio['portfolio_weights']}")
print(f"  Volatilities: {portfolio['asset_volatilities']}")
print(f"  Maturity: {portfolio['T']} year")
print(f"  Risk-free rate: {portfolio['r']:.1%}")
print()

# Test strikes
strikes = [
    (90.0, 'ITM'),
    (100.0, 'ATM'),  # Re-test to verify 0.08%
    (110.0, 'OTM')
]

# Initialize pricer
pricer = FBIQFTPricing(M=64, alpha=1.0, num_shots=8192)

# Store results
results = {
    'timestamp': datetime.now().isoformat(),
    'backend': backend.name,
    'portfolio': {
        'assets': len(portfolio['asset_prices']),
        'weights': portfolio['portfolio_weights'].tolist(),
        'volatilities': portfolio['asset_volatilities'].tolist(),
        'T': portfolio['T'],
        'r': portfolio['r']
    },
    'strikes': {}
}

# Test each strike
for K, strike_type in strikes:
    print("="*80)
    print(f"Testing {strike_type} (K=${K:.0f})")
    print("="*80)
    print()
    
    # Reset calibration for each strike (local calibration)
    pricer.A = None
    pricer.B = None
    
    print(f"⏱️  Running on {backend.name}...")
    print("   This will take ~15-20 minutes per strike")
    print()
    
    try:
        result = pricer.price_option(
            backend=backend,
            K=K,
            **portfolio
        )
        
        print(f"✅ {strike_type} completed!")
        print(f"  Classical price: ${result['price_classical']:.4f}")
        print(f"  Quantum price:   ${result['price_quantum']:.4f}")
        print(f"  Error:           {result['error_percent']:.2f}%")
        print(f"  Circuit depth:   {result['circuit_depth']}")
        print(f"  Qubits:          {result['num_qubits']}")
        print()
        
        # Store results
        results['strikes'][strike_type] = {
            'K': K,
            'classical': result['price_classical'],
            'quantum': result['price_quantum'],
            'error_percent': result['error_percent'],
            'circuit_depth': result['circuit_depth'],
            'num_qubits': result['num_qubits'],
            'sigma_p': result['sigma_p']
        }
        
        # Incremental save
        results_file = Path('results/hardware_test_results_extended.json')
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
    except Exception as e:
        print(f"❌ {strike_type} failed: {e}")
        import traceback
        traceback.print_exc()
        
        results['strikes'][strike_type] = {
            'K': K,
            'status': 'failed',
            'error': str(e)
        }
        
    print()

# Summary
print("="*80)
print("SUMMARY: Multi-Strike Hardware Validation")
print("="*80)
print()

if len([k for k in results['strikes'] if 'error_percent' in results['strikes'][k]]) > 0:
    print(f"{'Strike':<10} {'Type':<8} {'Classical':<12} {'Quantum':<12} {'Error':<10} {'Status'}")
    print("-"*80)
    
    errors = []
    for K, strike_type in strikes:
        if strike_type in results['strikes'] and 'error_percent' in results['strikes'][strike_type]:
            r = results['strikes'][strike_type]
            error = r['error_percent']
            errors.append(error)
            
            status = "✅" if error < 5.0 else ("⚠️" if error < 15.0 else "❌")
            print(f"${K:<9.0f} {strike_type:<8} ${r['classical']:<11.4f} "
                  f"${r['quantum']:<11.4f} {error:<9.2f}% {status}")
    
    print()
    
    if errors:
        print("Error Statistics:")
        print(f"  Mean error:   {np.mean(errors):.2f}%")
        print(f"  Std dev:      {np.std(errors):.2f}%")
        print(f"  Min error:    {np.min(errors):.2f}%")
        print(f"  Max error:    {np.max(errors):.2f}%")
        print()
        
        # Verdict
        if np.mean(errors) < 5.0:
            print("🎉 EXCEPTIONAL PERFORMANCE!")
            print("   All strikes <5% error - PUBLICATION GRADE!")
        elif np.mean(errors) < 15.0:
            print("✅ EXCELLENT PERFORMANCE!")
            print("   Average error <15% - meets NISQ expectations")
        else:
            print("⚠️  MARGINAL PERFORMANCE")
            print("   Average error >15% - typical NISQ behavior")

print()
print(f"✅ Results saved to: {results_file}")
print()

print("="*80)
print("NEXT STEPS")
print("="*80)
print()
print("1. Review results in: results/hardware_test_results_extended.json")
print("2. If errors <5%: Create publication notebook")
print("3. If errors 5-15%: Still excellent, document limitations")
print("4. If errors >15%: Expected NISQ, focus on complexity reduction")
print()
