#!/usr/bin/env python3
"""
============================================================================
QADP - COMPLETE FRAMEWORK (ALL COMPONENTS HARDWARE-READY)
============================================================================

This script runs the FULL QADP framework with ALL 3 quantum components:
1. QRC → IBM Hardware → Regime Factors
2. QTC → IBM Hardware → Temporal Patterns (using new build_circuits())
3. FB-IQFT → IBM Hardware → Option Price

Workflow:
1. SIMULATOR test first
2. If good → Synthetic data on HARDWARE
3. If good → Real market data on HARDWARE

Author: QADP Research Team
============================================================================
"""

import numpy as np
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

print("=" * 90)
print("  QADP - COMPLETE FRAMEWORK (ALL COMPONENTS)")
print("=" * 90)
print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)

# =============================================================================
# IMPORT COMPONENTS
# =============================================================================

print("\n📦 Loading QADP Components...")

from qrc import QuantumRecurrentCircuit
from qtc import QuantumTemporalConvolution
from qfdp.unified import FBIQFTPricing
from qfdp.unified.qrc_modulation import QRCModulation
from qiskit import transpile

print("  ✅ All modules loaded")

# =============================================================================
# SYNTHETIC TEST DATA
# =============================================================================

print("\n📊 Preparing Synthetic Test Data...")

n_assets = 4
asset_prices = np.array([100.0, 105.0, 98.0, 102.0])
asset_vols = np.array([0.20, 0.25, 0.22, 0.23])
weights = np.array([0.30, 0.25, 0.25, 0.20])
rho = 0.6
correlation = np.eye(n_assets) + rho * (1 - np.eye(n_assets))
price_history = np.array([99.0, 100.5, 99.8, 101.2, 100.0, 102.0])
K, T, r = 100.0, 1.0, 0.05

print(f"  Portfolio: {n_assets} assets, ρ={rho}")
print(f"  Price history: {price_history}")
print(f"  Strike: ${K}, Maturity: {T}yr")

# =============================================================================
# RUN ON SIMULATOR FIRST
# =============================================================================

def run_on_simulator():
    """Run complete QADP pipeline on simulator."""
    print("\n" + "=" * 90)
    print("  PHASE 1: SIMULATOR VALIDATION")
    print("=" * 90)
    
    results = {}
    
    # ----- QRC -----
    print("\n🔹 QRC (Regime Detection)...")
    qrc = QuantumRecurrentCircuit(n_factors=4)
    qrc.reset_hidden_state()
    
    avg_corr = np.mean(correlation[np.triu_indices(n_assets, 1)])
    stress = max(0, min(1, (avg_corr - 0.3) * 2))
    qrc_input = {'prices': np.mean(asset_prices), 'volatility': np.mean(asset_vols),
                 'corr_change': avg_corr - 0.3, 'stress': stress}
    
    qrc_result = qrc.forward(qrc_input)
    qrc_factors = qrc_result.factors
    qrc_circuit = qrc.h_quantum  # Store circuit for hardware
    
    print(f"    Factors: {np.round(qrc_factors, 4)}")
    print(f"    Circuit: {qrc_circuit.num_qubits} qubits, depth {qrc_circuit.depth()}")
    results['qrc'] = {'factors': qrc_factors, 'circuit': qrc_circuit, 'depth': qrc_result.circuit_depth}
    
    # ----- QTC -----
    print("\n🔹 QTC (Temporal Patterns)...")
    qtc = QuantumTemporalConvolution(kernel_size=3, n_kernels=4, n_qubits=4, n_layers=3)
    
    # Build circuits (for hardware capability verification)
    qtc_circuits = qtc.build_circuits(price_history)
    print(f"    Built {len(qtc_circuits)} kernel circuits")
    
    # Run normal forward for simulator baseline
    qtc_result = qtc.forward(price_history)
    qtc_patterns = qtc_result.patterns
    
    print(f"    Patterns: {np.round(qtc_patterns, 4)}")
    print(f"    Circuits: {qtc_circuits[0].num_qubits} qubits, depth {qtc_circuits[0].depth()}")
    results['qtc'] = {'patterns': qtc_patterns, 'circuits': qtc_circuits, 'depth': qtc_result.circuit_depth}
    
    # ----- Feature Fusion -----
    print("\n🔹 Feature Fusion...")
    alpha = 0.6
    fused = alpha * qrc_factors + (1 - alpha) * qtc_patterns
    print(f"    Fused: {np.round(fused, 4)}")
    results['fused'] = fused
    
    # ----- Enhanced σ_p -----
    print("\n🔹 Enhanced σ_p...")
    vol_diag = np.diag(asset_vols)
    cov_base = vol_diag @ correlation @ vol_diag
    sigma_p_base = np.sqrt(weights @ cov_base @ weights)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_base)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    modulator = QRCModulation(beta=0.5)
    n_f = min(len(fused), len(eigenvalues))
    mod_eigen, h_factors = modulator.apply_modulation(eigenvalues[:n_f], fused[:n_f])
    Lambda_mod = np.diag(mod_eigen)
    Q_K = eigenvectors[:, :n_f]
    cov_enhanced = Q_K @ Lambda_mod @ Q_K.T
    sigma_p_enhanced = np.sqrt(weights @ cov_enhanced @ weights)
    
    print(f"    σ_p (base): {sigma_p_base:.4f}")
    print(f"    σ_p (enhanced): {sigma_p_enhanced:.4f}")
    print(f"    Change: {(sigma_p_enhanced - sigma_p_base) / sigma_p_base * 100:+.2f}%")
    results['sigma_p'] = {'base': sigma_p_base, 'enhanced': sigma_p_enhanced, 'h_factors': h_factors}
    
    # ----- FB-IQFT -----
    print("\n🔹 FB-IQFT Pricing...")
    pricer = FBIQFTPricing(M=64, alpha=1.0, num_shots=8192)
    
    fb_result = pricer.price_option(
        asset_prices=asset_prices,
        asset_volatilities=asset_vols,
        correlation_matrix=correlation,
        portfolio_weights=weights,
        K=K, T=T, r=r,
        backend='simulator'
    )
    
    print(f"    Classical (BS): ${fb_result['price_classical']:.4f}")
    print(f"    Quantum: ${fb_result['price_quantum']:.4f}")
    print(f"    Error: {fb_result['error_percent']:.2f}%")
    print(f"    Circuit: {fb_result['num_qubits']} qubits, depth {fb_result['circuit_depth']}")
    results['fb_iqft'] = fb_result
    results['pricer'] = pricer
    
    return results


def run_on_hardware(sim_results, hw_backend):
    """Run complete QADP pipeline on IBM hardware."""
    print("\n" + "=" * 90)
    print(f"  PHASE 2: IBM HARDWARE ({hw_backend.name})")
    print("=" * 90)
    
    from qiskit_ibm_runtime import SamplerV2
    
    results = {}
    
    # ----- QRC on Hardware -----
    print("\n🔹 QRC on IBM Hardware...")
    hw_start = time.time()
    
    qrc_circuit = sim_results['qrc']['circuit']
    qrc_transpiled = transpile(qrc_circuit, hw_backend, optimization_level=3)
    print(f"    Transpiled depth: {qrc_transpiled.depth()}")
    
    try:
        sampler = SamplerV2(hw_backend)
        job = sampler.run([qrc_transpiled], shots=4096)
        print(f"    Job: {job.job_id()}")
        result = job.result()
        qrc_counts = result[0].data.meas.get_counts()
        
        # Extract factors using QRC's method
        qrc = QuantumRecurrentCircuit(n_factors=4)
        qrc_factors_hw = qrc._extract_factors(qrc_counts)
        
        print(f"    Hardware factors: {np.round(qrc_factors_hw, 4)}")
        print(f"    Time: {time.time() - hw_start:.1f}s")
        results['qrc'] = {'factors': qrc_factors_hw, 'success': True}
    except Exception as e:
        print(f"    ❌ Error: {e}")
        results['qrc'] = {'factors': sim_results['qrc']['factors'], 'success': False}
    
    # ----- QTC on Hardware -----
    print("\n🔹 QTC on IBM Hardware...")
    hw_start = time.time()
    
    try:
        qtc_circuits = sim_results['qtc']['circuits']
        qtc_kernel_counts = []
        
        for i, circ in enumerate(qtc_circuits):
            circ_transpiled = transpile(circ, hw_backend, optimization_level=3)
            sampler = SamplerV2(hw_backend)
            job = sampler.run([circ_transpiled], shots=2048)
            result = job.result()
            counts = result[0].data.meas.get_counts()
            qtc_kernel_counts.append(counts)
            print(f"    Kernel {i}: job {job.job_id()}")
        
        # Process using QTC's hardware method
        qtc = QuantumTemporalConvolution(kernel_size=3, n_kernels=4, n_qubits=4, n_layers=3)
        qtc_result_hw = qtc.forward_with_counts(qtc_kernel_counts)
        qtc_patterns_hw = qtc_result_hw.patterns
        
        print(f"    Hardware patterns: {np.round(qtc_patterns_hw, 4)}")
        print(f"    Time: {time.time() - hw_start:.1f}s")
        results['qtc'] = {'patterns': qtc_patterns_hw, 'success': True}
    except Exception as e:
        print(f"    ❌ Error: {e}")
        results['qtc'] = {'patterns': sim_results['qtc']['patterns'], 'success': False}
    
    # ----- Feature Fusion with Hardware Results -----
    print("\n🔹 Feature Fusion (Hardware Results)...")
    qrc_factors = results['qrc']['factors']
    qtc_patterns = results['qtc']['patterns']
    alpha = 0.6
    fused_hw = alpha * qrc_factors + (1 - alpha) * qtc_patterns
    print(f"    Fused (HW): {np.round(fused_hw, 4)}")
    results['fused'] = fused_hw
    
    # ----- Enhanced σ_p with Hardware Results -----
    print("\n🔹 Enhanced σ_p (Hardware Results)...")
    vol_diag = np.diag(asset_vols)
    cov_base = vol_diag @ correlation @ vol_diag
    sigma_p_base = np.sqrt(weights @ cov_base @ weights)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_base)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    modulator = QRCModulation(beta=0.5)
    n_f = min(len(fused_hw), len(eigenvalues))
    mod_eigen, h_factors = modulator.apply_modulation(eigenvalues[:n_f], fused_hw[:n_f])
    Lambda_mod = np.diag(mod_eigen)
    Q_K = eigenvectors[:, :n_f]
    cov_enhanced = Q_K @ Lambda_mod @ Q_K.T
    sigma_p_hw = np.sqrt(weights @ cov_enhanced @ weights)
    
    print(f"    σ_p (HW enhanced): {sigma_p_hw:.4f}")
    results['sigma_p'] = sigma_p_hw
    
    # ----- FB-IQFT on Hardware -----
    print("\n🔹 FB-IQFT on IBM Hardware...")
    hw_start = time.time()
    
    try:
        pricer = FBIQFTPricing(M=64, alpha=1.0, num_shots=8192)
        fb_result_hw = pricer.price_option(
            asset_prices=asset_prices,
            asset_volatilities=asset_vols,
            correlation_matrix=correlation,
            portfolio_weights=weights,
            K=K, T=T, r=r,
            backend=hw_backend
        )
        
        print(f"    Classical: ${fb_result_hw['price_classical']:.4f}")
        print(f"    Quantum (HW): ${fb_result_hw['price_quantum']:.4f}")
        print(f"    Error: {fb_result_hw['error_percent']:.2f}%")
        print(f"    Time: {time.time() - hw_start:.1f}s")
        results['fb_iqft'] = {'result': fb_result_hw, 'success': True}
    except Exception as e:
        print(f"    ❌ Error: {e}")
        results['fb_iqft'] = {'result': sim_results['fb_iqft'], 'success': False}
    
    return results


def print_comparison(sim_results, hw_results=None):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("  RESULTS COMPARISON")
    print("=" * 90)
    
    print(f"\n{'Component':<20} {'SIMULATOR':<25} {'HARDWARE':<25} {'Status':<10}")
    print("-" * 80)
    
    qrc_sim = np.round(sim_results['qrc']['factors'], 3)
    qrc_hw = np.round(hw_results['qrc']['factors'], 3) if hw_results else qrc_sim
    qrc_status = '✅ HW' if hw_results and hw_results['qrc']['success'] else '⚠️ SIM'
    print(f"{'QRC Factors':<20} {str(qrc_sim):<25} {str(qrc_hw):<25} {qrc_status:<10}")
    
    qtc_sim = np.round(sim_results['qtc']['patterns'], 3)
    qtc_hw = np.round(hw_results['qtc']['patterns'], 3) if hw_results else qtc_sim
    qtc_status = '✅ HW' if hw_results and hw_results['qtc']['success'] else '⚠️ SIM'
    print(f"{'QTC Patterns':<20} {str(qtc_sim):<25} {str(qtc_hw):<25} {qtc_status:<10}")
    
    fused_sim = np.round(sim_results['fused'], 3)
    fused_hw = np.round(hw_results['fused'], 3) if hw_results else fused_sim
    print(f"{'Fused Features':<20} {str(fused_sim):<25} {str(fused_hw):<25} {'✅':<10}")
    
    sigma_sim = f"{sim_results['sigma_p']['enhanced']:.4f}"
    sigma_hw = f"{hw_results['sigma_p']:.4f}" if hw_results else sigma_sim
    print(f"{'σ_p Enhanced':<20} {sigma_sim:<25} {sigma_hw:<25} {'✅':<10}")
    
    price_sim = f"${sim_results['fb_iqft']['price_quantum']:.4f}"
    price_hw = f"${hw_results['fb_iqft']['result']['price_quantum']:.4f}" if hw_results else price_sim
    fb_status = '✅ HW' if hw_results and hw_results['fb_iqft']['success'] else '⚠️ SIM'
    print(f"{'FB-IQFT Price':<20} {price_sim:<25} {price_hw:<25} {fb_status:<10}")
    
    err_sim = f"{sim_results['fb_iqft']['error_percent']:.2f}%"
    err_hw = f"{hw_results['fb_iqft']['result']['error_percent']:.2f}%" if hw_results else err_sim
    print(f"{'Error vs BS':<20} {err_sim:<25} {err_hw:<25}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='simulator',
                       choices=['simulator', 'hardware'],
                       help='Run mode')
    args = parser.parse_args()
    
    # STEP 1: Always run simulator first
    sim_results = run_on_simulator()
    
    # STEP 2: Check if simulator is good
    sim_error = sim_results['fb_iqft']['error_percent']
    
    print("\n" + "=" * 90)
    print("  SIMULATOR VALIDATION RESULTS")
    print("=" * 90)
    print(f"\n  Simulator error: {sim_error:.2f}%")
    
    if sim_error < 1.0:
        print("  ✅ SIMULATOR VALIDATION PASSED - Ready for hardware")
    else:
        print("  ⚠️  High simulator error - investigate before hardware")
    
    if args.mode == 'hardware':
        # STEP 3: Run on hardware
        print("\n🔌 Connecting to IBM Quantum...")
        
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        try:
            service = QiskitRuntimeService()
        except:
            API_TOKEN = "71ZGWcl3-sDX9RlhN9NCvhcGxg0FMRNF6eVhotgnxobr"
            QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=API_TOKEN, overwrite=True)
            service = QiskitRuntimeService()
        
        hw_backend = None
        for name in ['ibm_torino', 'ibm_kyiv', 'ibm_osaka']:
            try:
                hw_backend = service.backend(name)
                print(f"  ✅ Selected: {name}")
                break
            except:
                continue
        
        if hw_backend:
            hw_results = run_on_hardware(sim_results, hw_backend)
            print_comparison(sim_results, hw_results)
            
            # Final summary
            hw_components = sum([
                hw_results['qrc']['success'],
                hw_results['qtc']['success'],
                hw_results['fb_iqft']['success']
            ])
            
            print(f"\n  Hardware components: {hw_components}/3")
            print(f"  Simulator error: {sim_error:.2f}%")
            print(f"  Hardware error: {hw_results['fb_iqft']['result']['error_percent']:.2f}%")
        else:
            print("  ❌ No backend available")
    else:
        print_comparison(sim_results)
    
    print("\n" + "=" * 90)
    print("  ✅ COMPLETE")
    print("=" * 90)
