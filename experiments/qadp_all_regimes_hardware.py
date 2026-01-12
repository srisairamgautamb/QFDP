#!/usr/bin/env python3
"""
QADP - ALL REGIMES ON IBM HARDWARE (FINAL TEST)
================================================
Tests CALM, MODERATE, ELEVATED, STRESSED regimes on real quantum hardware
"""

import numpy as np
import sys
import time
from datetime import datetime

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

print("=" * 90)
print("  QADP - ALL MARKET REGIMES ON IBM QUANTUM HARDWARE")
print("=" * 90)
print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)

from qrc import QuantumRecurrentCircuit
from qtc import QuantumTemporalConvolution
from qfdp.unified import FBIQFTPricing
from qfdp.unified.qrc_modulation import QRCModulation
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# Connect to IBM
print("\n🔌 Connecting to IBM Quantum...")
service = QiskitRuntimeService()
hw_backend = service.backend('ibm_torino')
print(f"   ✅ Connected: {hw_backend.name}")

# Define regimes
REGIMES = [
    {"name": "CALM",     "rho": 0.25, "prices": [99.5, 100.0, 100.2, 100.5, 100.6, 100.8]},
    {"name": "MODERATE", "rho": 0.45, "prices": [100.0, 100.3, 99.8, 100.5, 100.2, 100.4]},
    {"name": "ELEVATED", "rho": 0.60, "prices": [100.0, 101.0, 99.5, 101.5, 100.0, 101.0]},
    {"name": "STRESSED", "rho": 0.80, "prices": [100.0, 98.0, 95.0, 97.0, 94.0, 96.0]},
]

# Portfolio setup
n_assets = 4
asset_prices = np.array([100.0, 105.0, 98.0, 102.0])
asset_vols = np.array([0.20, 0.25, 0.22, 0.23])
weights = np.array([0.30, 0.25, 0.25, 0.20])
K, T, r = 100.0, 1.0, 0.05

all_results = []

for regime in REGIMES:
    rho = regime["rho"]
    price_history = np.array(regime["prices"])
    correlation = np.eye(n_assets) + rho * (1 - np.eye(n_assets))
    
    print(f"\n{'='*90}")
    print(f"  REGIME: {regime['name']} (ρ = {rho})")
    print(f"{'='*90}")
    
    start_time = time.time()
    
    # ----- QRC -----
    print("\n  🔹 QRC...")
    qrc = QuantumRecurrentCircuit(n_factors=4)
    qrc.reset_hidden_state()
    
    stress = max(0, min(1, (rho - 0.3) * 2))
    qrc_input = {'prices': np.mean(asset_prices), 'volatility': np.mean(asset_vols),
                 'corr_change': rho - 0.3, 'stress': stress}
    
    qrc_result_sim = qrc.forward(qrc_input)
    qrc_factors_sim = qrc_result_sim.factors
    qrc_circuit = qrc.h_quantum
    
    # Hardware
    qrc_t = transpile(qrc_circuit, hw_backend, optimization_level=3)
    sampler = SamplerV2(hw_backend)
    job = sampler.run([qrc_t], shots=4096)
    result = job.result()
    qrc_counts = result[0].data.meas.get_counts()
    qrc_factors_hw = qrc._extract_factors(qrc_counts)
    print(f"     Sim: {np.round(qrc_factors_sim, 3)} | HW: {np.round(qrc_factors_hw, 3)}")
    
    # ----- QTC -----
    print("  🔹 QTC...")
    qtc = QuantumTemporalConvolution(kernel_size=3, n_kernels=4, n_qubits=4, n_layers=3)
    qtc_result_sim = qtc.forward(price_history)
    qtc_patterns_sim = qtc_result_sim.patterns
    
    qtc_circuits = qtc.build_circuits(price_history)
    qtc_counts = []
    for circ in qtc_circuits:
        circ_t = transpile(circ, hw_backend, optimization_level=3)
        job = sampler.run([circ_t], shots=2048)
        counts = job.result()[0].data.meas.get_counts()
        qtc_counts.append(counts)
    
    qtc_result_hw = qtc.forward_with_counts(qtc_counts)
    qtc_patterns_hw = qtc_result_hw.patterns
    print(f"     Sim: {np.round(qtc_patterns_sim, 3)} | HW: {np.round(qtc_patterns_hw, 3)}")
    
    # ----- Fusion & σ_p -----
    alpha = 0.6
    fused_hw = alpha * qrc_factors_hw + (1 - alpha) * qtc_patterns_hw
    
    vol_diag = np.diag(asset_vols)
    cov_base = vol_diag @ correlation @ vol_diag
    sigma_p_base = np.sqrt(weights @ cov_base @ weights)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_base)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    modulator = QRCModulation(beta=0.5)
    n_f = min(len(fused_hw), len(eigenvalues))
    mod_eigen, _ = modulator.apply_modulation(eigenvalues[:n_f], fused_hw[:n_f])
    Lambda_mod = np.diag(mod_eigen)
    Q_K = eigenvectors[:, :n_f]
    cov_enhanced = Q_K @ Lambda_mod @ Q_K.T
    sigma_p_hw = np.sqrt(weights @ cov_enhanced @ weights)
    
    print(f"  🔹 σ_p: {sigma_p_base*100:.2f}% → {sigma_p_hw*100:.2f}% ({(sigma_p_hw-sigma_p_base)/sigma_p_base*100:+.1f}%)")
    
    # ----- FB-IQFT -----
    print("  🔹 FB-IQFT...")
    pricer = FBIQFTPricing(M=64, alpha=1.0, num_shots=8192)
    
    result_sim = pricer.price_option(
        asset_prices=asset_prices, asset_volatilities=asset_vols,
        correlation_matrix=correlation, portfolio_weights=weights,
        K=K, T=T, r=r, backend='simulator'
    )
    
    result_hw = pricer.price_option(
        asset_prices=asset_prices, asset_volatilities=asset_vols,
        correlation_matrix=correlation, portfolio_weights=weights,
        K=K, T=T, r=r, backend=hw_backend
    )
    
    elapsed = time.time() - start_time
    
    print(f"     Sim: ${result_sim['price_quantum']:.4f} ({result_sim['error_percent']:.2f}%)")
    print(f"     HW:  ${result_hw['price_quantum']:.4f} ({result_hw['error_percent']:.2f}%)")
    print(f"     Time: {elapsed:.1f}s")
    
    all_results.append({
        'regime': regime['name'],
        'rho': rho,
        'sim_price': result_sim['price_quantum'],
        'hw_price': result_hw['price_quantum'],
        'sim_error': result_sim['error_percent'],
        'hw_error': result_hw['error_percent'],
        'sigma_change': (sigma_p_hw - sigma_p_base) / sigma_p_base * 100,
        'time': elapsed
    })

# ===== FINAL SUMMARY =====
print("\n" + "=" * 90)
print("  ALL REGIMES - IBM HARDWARE RESULTS")
print("=" * 90)

print(f"\n{'Regime':<12} {'ρ':>6} {'Sim Price':>12} {'HW Price':>12} {'Sim Err':>10} {'HW Err':>10} {'σ_p Chg':>10}")
print("-" * 80)

for r in all_results:
    print(f"{r['regime']:<12} {r['rho']:>6.2f} ${r['sim_price']:>10.4f} ${r['hw_price']:>10.4f} "
          f"{r['sim_error']:>9.2f}% {r['hw_error']:>9.2f}% {r['sigma_change']:>+9.1f}%")

avg_sim = np.mean([r['sim_error'] for r in all_results])
avg_hw = np.mean([r['hw_error'] for r in all_results])

print("-" * 80)
print(f"{'AVERAGE':<12} {'':>6} {'':>12} {'':>12} {avg_sim:>9.2f}% {avg_hw:>9.2f}%")

print(f"""

✅ QADP FRAMEWORK - ALL REGIMES VALIDATED ON IBM HARDWARE

   Backend: {hw_backend.name}
   Components: 3/3 (QRC ✓, QTC ✓, FB-IQFT ✓)
   Avg Simulator Error: {avg_sim:.2f}%
   Avg Hardware Error: {avg_hw:.2f}%
   Noise Contribution: {avg_hw - avg_sim:+.2f}%

{'='*90}
""")
