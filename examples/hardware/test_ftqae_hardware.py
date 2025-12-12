"""
FT-QAE Hardware Test
====================

Test Factor-Tensorized Quantum Amplitude Estimation on IBM Quantum hardware.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qfdp.ft_qae.pricing import ft_qae_price_option

def test_ftqae_hardware():
    """Test FT-QAE on simulator then hardware."""
    
    # 3-asset basket (small for testing)
    N = 3
    weights = np.array([0.4, 0.3, 0.3])
    vols = np.array([0.20, 0.25, 0.18])
    
    corr = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    
    spot = 100.0
    strike = 105.0
    rate = 0.05
    maturity = 1.0
    
    print("="*70)
    print("FT-QAE SIMULATOR TEST")
    print("="*70)
    print()
    
    # Test on simulator first
    result_sim = ft_qae_price_option(
        weights,
        vols,
        corr,
        spot_value=spot,
        strike=strike,
        risk_free_rate=rate,
        maturity=maturity,
        K_factors=2,
        n_qubits_per_factor=3,  # 8 points per factor
        qae_iterations=[1, 2, 4],
        shots_per_iteration=1000,
        validate_vs_classical=True,
        run_on_hardware=False
    )
    
    print()
    print("="*70)
    print("SIMULATOR RESULTS")
    print("="*70)
    print(f"✓ FT-QAE Price: ${result_sim.price:.4f}")
    print(f"✓ Classical MC: ${result_sim.classical_mc_price:.4f}")
    if result_sim.classical_mc_price:
        error_sim = abs(result_sim.price - result_sim.classical_mc_price) / result_sim.classical_mc_price * 100
        print(f"✓ Error: {error_sim:.2f}%")
    print(f"✓ Qubits: {result_sim.total_qubits}")
    print(f"✓ Depth: {result_sim.circuit_depth}")
    print()
    
    # Test on hardware
    print("="*70)
    print("FT-QAE HARDWARE TEST")
    print("="*70)
    print()
    
    result_hw = ft_qae_price_option(
        weights,
        vols,
        corr,
        spot_value=spot,
        strike=strike,
        risk_free_rate=rate,
        maturity=maturity,
        K_factors=2,
        n_qubits_per_factor=3,
        qae_iterations=[1, 2, 4],
        shots_per_iteration=1000,
        validate_vs_classical=True,
        run_on_hardware=True
    )
    
    print()
    print("="*70)
    print("HARDWARE RESULTS")
    print("="*70)
    print(f"✓ FT-QAE Price: ${result_hw.price:.4f}")
    print(f"✓ Classical MC: ${result_hw.classical_mc_price:.4f}")
    if result_hw.classical_mc_price:
        error_hw = abs(result_hw.price - result_hw.classical_mc_price) / result_hw.classical_mc_price * 100
        print(f"✓ Error: {error_hw:.2f}%")
    print(f"✓ Qubits: {result_hw.total_qubits}")
    print(f"✓ Depth: {result_hw.circuit_depth}")
    print()
    
    print("="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Simulator: ${result_sim.price:.4f} (error: {error_sim:.2f}%)")
    print(f"Hardware:  ${result_hw.price:.4f} (error: {error_hw:.2f}%)")
    print(f"Classical: ${result_sim.classical_mc_price:.4f}")
    print("="*70)

if __name__ == "__main__":
    test_ftqae_hardware()
