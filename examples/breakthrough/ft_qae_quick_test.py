"""
Quick Validation Test for FT-QAE
=================================

Tests that the FT-QAE implementation runs correctly.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qfdp.ft_qae.pricing import ft_qae_price_option

def test_ft_qae_small():
    """Test FT-QAE on a small 3-asset basket."""
    
    print("="*70)
    print("FT-QAE QUICK VALIDATION TEST")
    print("="*70)
    print()
    
    # 3-asset basket
    N = 3
    weights = np.array([0.4, 0.3, 0.3])
    vols = np.array([0.20, 0.25, 0.18])
    
    # Correlation matrix
    corr = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    
    # Option parameters
    spot = 100.0
    strike = 100.0
    rate = 0.05
    maturity = 1.0
    
    # Run FT-QAE with small parameters for quick test
    result = ft_qae_price_option(
        weights,
        vols,
        corr,
        spot_value=spot,
        strike=strike,
        risk_free_rate=rate,
        maturity=maturity,
        K_factors=2,  # Small for quick test
        n_qubits_per_factor=4,  # 16 points per factor
        qae_iterations=[1, 2, 4],  # Few iterations
        shots_per_iteration=100,  # Low shots for speed
        validate_vs_classical=True
    )
    
    print()
    print("TEST RESULTS:")
    print(f"  ✓ FT-QAE Price: ${result.price:.2f}")
    print(f"  ✓ Classical MC: ${result.classical_mc_price:.2f}")
    print(f"  ✓ Qubits: {result.total_qubits}")
    print(f"  ✓ Depth: {result.circuit_depth}")
    print(f"  ✓ Variance explained: {result.variance_explained*100:.1f}%")
    
    # Basic sanity checks
    assert result.price > 0, "Price should be positive"
    assert result.total_qubits == 2 * 4 + 1, "Should have K*n + 1 qubits"
    assert result.variance_explained > 0.5, "Should explain >50% variance"
    
    print()
    print("✓ ALL TESTS PASSED!")
    print()
    
    return result


if __name__ == "__main__":
    result = test_ft_qae_small()
