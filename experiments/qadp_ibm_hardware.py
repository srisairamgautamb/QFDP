#!/usr/bin/env python3
"""
QADP - IBM Quantum Hardware Execution
=====================================

This script runs the QADP framework on REAL IBM Quantum hardware.
API Key provided by user.
"""

import numpy as np
import sys
import time
from datetime import datetime

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

print("=" * 80)
print("🚀 QADP - IBM QUANTUM HARDWARE EXECUTION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# =============================================================================
# STEP 1: Connect to IBM Quantum
# =============================================================================

print("🔌 STEP 1: Connecting to IBM Quantum...")
print("-" * 50)

API_TOKEN = "71ZGWcl3-sDX9RlhN9NCvhcGxg0FMRNF6eVhotgnxobr"

from qiskit_ibm_runtime import QiskitRuntimeService

try:
    # Try existing credentials first
    try:
        service = QiskitRuntimeService()
        print("✅ Using existing saved credentials")
    except:
        # Save new credentials - Note: newer versions use ibm_quantum_platform
        print("📝 Saving new IBM Quantum credentials...")
        try:
            # Try new channel name first
            QiskitRuntimeService.save_account(
                channel="ibm_quantum_platform",
                token=API_TOKEN,
                overwrite=True
            )
        except:
            # Fallback to older channel name
            QiskitRuntimeService.save_account(
                token=API_TOKEN,
                overwrite=True
            )
        service = QiskitRuntimeService()
        print("✅ Credentials saved and connected!")
    
    # List available backends
    print("\n📡 Available quantum backends:")
    backends = service.backends(simulator=False, operational=True)
    
    for b in backends[:5]:
        print(f"   • {b.name}: {b.num_qubits} qubits")
    
    # Select best backend
    preferred = ['ibm_torino', 'ibm_kyiv', 'ibm_osaka', 'ibm_sherbrooke', 'ibm_brisbane']
    backend = None
    backend_name = None
    
    for name in preferred:
        try:
            backend = service.backend(name)
            backend_name = name
            print(f"\n✅ Selected: {name}")
            break
        except:
            continue
    
    if backend is None and backends:
        backend = backends[0]
        backend_name = backend.name
        print(f"\n✅ Selected: {backend_name} (first available)")
    
    if backend is None:
        print("\n❌ No backends available!")
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ Connection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# STEP 2: Import QADP Components
# =============================================================================

print("\n" + "=" * 80)
print("📦 STEP 2: Loading QADP Framework...")
print("-" * 50)

from qrc import QuantumRecurrentCircuit
from qtc import QuantumTemporalConvolution
from qfdp.unified import FBIQFTPricing

print("✅ All modules loaded")

# =============================================================================
# STEP 3: Prepare Test Data
# =============================================================================

print("\n" + "=" * 80)
print("📊 STEP 3: Preparing Test Data...")
print("-" * 50)

# Simple synthetic portfolio (for hardware test)
n_assets = 4
asset_prices = np.array([100.0, 105.0, 98.0, 102.0])
asset_vols = np.array([0.20, 0.25, 0.22, 0.23])
weights = np.array([0.30, 0.25, 0.25, 0.20])
rho = 0.6  # Elevated regime
correlation = np.eye(n_assets) + rho * (1 - np.eye(n_assets))
K, T, r = 100.0, 1.0, 0.05

print(f"  Assets: 4")
print(f"  Correlation: ρ = {rho}")
print(f"  Strike: K = ${K}")
print(f"  Maturity: T = {T} year")

# =============================================================================
# STEP 4: Run on SIMULATOR first (baseline)
# =============================================================================

print("\n" + "=" * 80)
print("🖥️  STEP 4: Simulator Baseline...")
print("-" * 50)

pricer = FBIQFTPricing(M=64, alpha=1.0, num_shots=8192)

sim_start = time.time()
sim_result = pricer.price_option(
    asset_prices=asset_prices,
    asset_volatilities=asset_vols,
    correlation_matrix=correlation,
    portfolio_weights=weights,
    K=K, T=T, r=r,
    backend='simulator'
)
sim_time = time.time() - sim_start

print(f"  Classical price: ${sim_result['price_classical']:.4f}")
print(f"  Quantum price:   ${sim_result['price_quantum']:.4f}")
print(f"  Error:           {sim_result['error_percent']:.2f}%")
print(f"  Time:            {sim_time:.2f}s")

# =============================================================================
# STEP 5: Run on IBM QUANTUM HARDWARE
# =============================================================================

print("\n" + "=" * 80)
print(f"⚛️  STEP 5: Running on IBM Quantum Hardware ({backend_name})...")
print("-" * 50)
print()
print("⏱️  This may take 5-20 minutes (queue + execution)...")
print("   Please wait...")
print()

# Create fresh pricer for hardware
pricer_hw = FBIQFTPricing(M=64, alpha=1.0, num_shots=8192)
pricer_hw.A = None  # Force recalibration
pricer_hw.B = None

hw_start = time.time()

try:
    hw_result = pricer_hw.price_option(
        asset_prices=asset_prices,
        asset_volatilities=asset_vols,
        correlation_matrix=correlation,
        portfolio_weights=weights,
        K=K, T=T, r=r,
        backend=backend
    )
    hw_time = time.time() - hw_start
    
    print(f"\n✅ HARDWARE EXECUTION COMPLETE!")
    print(f"  Classical price: ${hw_result['price_classical']:.4f}")
    print(f"  Quantum price:   ${hw_result['price_quantum']:.4f}")
    print(f"  Error:           {hw_result['error_percent']:.2f}%")
    print(f"  Circuit depth:   {hw_result['circuit_depth']}")
    print(f"  Qubits:          {hw_result['num_qubits']}")
    print(f"  Time:            {hw_time:.2f}s")
    
    hardware_success = True
    
except Exception as e:
    print(f"\n❌ Hardware execution failed: {e}")
    import traceback
    traceback.print_exc()
    hw_result = None
    hw_time = 0
    hardware_success = False

# =============================================================================
# STEP 6: Comparison Table
# =============================================================================

print("\n" + "=" * 80)
print("📊 STEP 6: Results Comparison")
print("=" * 80)

print(f"\n{'Backend':<20} {'Price':<12} {'Error':<10} {'Time':<10}")
print("-" * 55)
print(f"{'Simulator':<20} ${sim_result['price_quantum']:<10.4f} {sim_result['error_percent']:<9.2f}% {sim_time:<9.2f}s")

if hardware_success:
    print(f"{backend_name:<20} ${hw_result['price_quantum']:<10.4f} {hw_result['error_percent']:<9.2f}% {hw_time:<9.2f}s")
    
    # Hardware noise
    noise_contribution = hw_result['error_percent'] - sim_result['error_percent']
    print(f"\n  Hardware noise contribution: {noise_contribution:+.2f}%")
    
    if hw_result['error_percent'] < 30:
        print("\n  ✅ HARDWARE EXECUTION SUCCESSFUL (error < 30% NISQ threshold)")
    else:
        print("\n  ⚠️  High error - may need error mitigation")
else:
    print(f"{'Hardware':<20} {'FAILED':<12} {'-':<10} {'-':<10}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("🏁 QADP IBM QUANTUM HARDWARE DEMO - COMPLETE")
print("=" * 80)
print()
print(f"  ✅ Framework: Quantum Adaptive Derivative Pricing (QADP)")
print(f"  ✅ Backend: {backend_name}")
print(f"  ✅ Simulator error: {sim_result['error_percent']:.2f}%")
if hardware_success:
    print(f"  ✅ Hardware error: {hw_result['error_percent']:.2f}%")
print()
print("=" * 80)
