"""
Test Corrected Factor-Based Quantum Monte Carlo
================================================

This uses the proper, mathematically sound implementation.
Target: <2% error on real hardware.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qfdp.fb_iqft.pricing_v2 import factor_based_quantum_monte_carlo

def test_fb_qmc():
    """Test corrected FB-QMC."""
    
    # 5-asset basket
    N = 5
    weights = np.ones(N) / N
    vols = np.array([0.20, 0.25, 0.18, 0.22, 0.19])
    
    corr = np.array([
        [1.0, 0.5, 0.3, 0.2, 0.1],
        [0.5, 1.0, 0.4, 0.3, 0.2],
        [0.3, 0.4, 1.0, 0.5, 0.3],
        [0.2, 0.3, 0.5, 1.0, 0.4],
        [0.1, 0.2, 0.3, 0.4, 1.0]
    ])
    
    spot = 100.0
    strike = 105.0
    rate = 0.05
    maturity = 1.0
    
    # Test on simulator with high resolution
    print("Testing on SIMULATOR (validation)...")
    print()
    
    result_sim = factor_based_quantum_monte_carlo(
        weights, vols, corr,
        spot_value=spot, strike=strike,
        risk_free_rate=rate, maturity=maturity,
        K=4,
        n_qubits_per_factor=4,  # 16 grid points
        shots=8192,  # High shots for accuracy
        run_on_hardware=False,
        validate_vs_classical=True
    )
    
    print()
    print("="*70)
    print("SIMULATOR TEST RESULTS")
    print("="*70)
    print(f"✓ Price: ${result_sim.price:.4f}")
    print(f"✓ Classical MC: ${result_sim.classical_price_baseline:.4f}")
    print(f"✓ Error: {result_sim.error_pct:.2f}%")
    print(f"✓ Target: <2%")
    
    if result_sim.error_pct < 2.0:
        print("\n✅ SUCCESS: Error < 2%!")
    elif result_sim.error_pct < 5.0:
        print("\n⚠️  Close but not quite <2%")
    else:
        print("\n❌ Error too high")
    
    print()
    
    # Now test on hardware
    print("="*70)
    print("DEPLOYING TO IBM QUANTUM HARDWARE")
    print("="*70)
    print()
    
    result_hw = factor_based_quantum_monte_carlo(
        weights, vols, corr,
        spot_value=spot, strike=strike,
        risk_free_rate=rate, maturity=maturity,
        K=4,
        n_qubits_per_factor=4,
        shots=8192,
        run_on_hardware=True,
        backend_name=None,  # Auto-select (excludes ibm_fez)
        validate_vs_classical=True
    )
    
    print()
    print("="*70)
    print("HARDWARE TEST RESULTS")
    print("="*70)
    print(f"✓ Price: ${result_hw.price:.4f}")
    print(f"✓ Classical MC: ${result_hw.classical_price_baseline:.4f}")
    print(f"✓ Error: {result_hw.error_pct:.2f}%")
    
    if result_hw.error_pct < 2.0:
        print("\n🎉 SUCCESS ON HARDWARE: Error < 2%!")
    else:
        print(f"\n⚠️  Hardware error {result_hw.error_pct:.2f}% (target: <2%)")
    
    # Save results
    with open('fb_qmc_final_results.txt', 'w') as f:
        f.write("Factor-Based Quantum Monte Carlo - Final Results\n")
        f.write("="*70 + "\n\n")
        f.write("SIMULATOR:\n")
        f.write(f"  Price: ${result_sim.price:.4f}\n")
        f.write(f"  Classical: ${result_sim.classical_price_baseline:.4f}\n")
        f.write(f"  Error: {result_sim.error_pct:.2f}%\n\n")
        f.write("HARDWARE:\n")
        f.write(f"  Price: ${result_hw.price:.4f}\n")
        f.write(f"  Classical: ${result_hw.classical_price_baseline:.4f}\n")
        f.write(f"  Error: {result_hw.error_pct:.2f}%\n\n")
        f.write(f"Target achieved: {result_hw.error_pct < 2.0}\n")
    
    print("\n✓ Results saved to fb_qmc_final_results.txt")
    print("="*70)
    
    return result_sim, result_hw


if __name__ == "__main__":
    result_sim, result_hw = test_fb_qmc()
