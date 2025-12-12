"""
Test Copula Reconstruction Error Fix
====================================

Validates that adaptive K selection resolves the high reconstruction error
issue (0.8751 → <0.3) while maintaining variance explained ≥95%.

Research paper requirement: Frobenius error < 0.3 for all test cases.
"""

import numpy as np
from qfdp_multiasset.sparse_copula import FactorDecomposer


def test_fixed_K_high_error():
    """Show that fixed K=3 produces high error for N=5."""
    corr = np.array([
        [1.0, 0.8, 0.6, 0.4, 0.2],
        [0.8, 1.0, 0.7, 0.5, 0.3],
        [0.6, 0.7, 1.0, 0.6, 0.4],
        [0.4, 0.5, 0.6, 1.0, 0.5],
        [0.2, 0.3, 0.4, 0.5, 1.0]
    ])
    
    decomposer = FactorDecomposer()
    L, D, metrics = decomposer.fit(corr, K=3)
    
    print(f"Fixed K=3:")
    print(f"  Variance explained: {metrics.variance_explained:.1%}")
    print(f"  Frobenius error: {metrics.frobenius_error:.4f}")
    print(f"  Status: {'❌ FAIL' if metrics.frobenius_error > 0.3 else '✅ PASS'} (error > 0.3)")
    print()
    
    assert metrics.frobenius_error > 0.3, "Expected high error with K=3"
    return metrics.frobenius_error


def test_adaptive_K_low_error():
    """Show that adaptive K selection achieves error < 0.3."""
    corr = np.array([
        [1.0, 0.8, 0.6, 0.4, 0.2],
        [0.8, 1.0, 0.7, 0.5, 0.3],
        [0.6, 0.7, 1.0, 0.6, 0.4],
        [0.4, 0.5, 0.6, 1.0, 0.5],
        [0.2, 0.3, 0.4, 0.5, 1.0]
    ])
    
    decomposer = FactorDecomposer()
    L, D, metrics = decomposer.fit(corr, K=None)  # Auto-select
    
    print(f"Adaptive K selection:")
    print(f"  Selected K: {L.shape[1]}/{corr.shape[0]}")
    print(f"  Variance explained: {metrics.variance_explained:.1%}")
    print(f"  Frobenius error: {metrics.frobenius_error:.4f}")
    print(f"  Status: {'✅ PASS' if metrics.frobenius_error < 0.3 else '❌ FAIL'} (error < 0.3)")
    print()
    
    assert metrics.frobenius_error < 0.3, f"Expected error < 0.3, got {metrics.frobenius_error}"
    assert metrics.variance_explained >= 0.95, f"Expected variance ≥95%, got {metrics.variance_explained:.1%}"
    return L.shape[1], metrics.frobenius_error


def test_n10_adaptive_K():
    """Test adaptive K for N=10 assets."""
    # Generate synthetic correlation with strong factor structure
    np.random.seed(42)
    K_true = 4
    N = 10
    
    # Generate factor model
    L_true = np.random.randn(N, K_true) / np.sqrt(K_true)
    corr = L_true @ L_true.T + 0.1 * np.eye(N)
    
    # Normalize to correlation
    std_devs = np.sqrt(np.diag(corr))
    corr = corr / np.outer(std_devs, std_devs)
    np.fill_diagonal(corr, 1.0)
    
    decomposer = FactorDecomposer()
    L, D, metrics = decomposer.fit(corr, K=None)
    
    print(f"N=10 adaptive K:")
    print(f"  Selected K: {L.shape[1]}/{N}")
    print(f"  Variance explained: {metrics.variance_explained:.1%}")
    print(f"  Frobenius error: {metrics.frobenius_error:.4f}")
    print(f"  Status: {'✅ PASS' if metrics.frobenius_error < 0.3 else '❌ FAIL'}")
    print()
    
    assert metrics.frobenius_error < 0.3
    assert metrics.variance_explained >= 0.95
    return L.shape[1], metrics.frobenius_error


def test_n20_adaptive_K():
    """Test adaptive K for N=20 assets."""
    np.random.seed(123)
    K_true = 5
    N = 20
    
    # Generate factor model
    L_true = np.random.randn(N, K_true) / np.sqrt(K_true)
    corr = L_true @ L_true.T + 0.15 * np.eye(N)
    
    # Normalize to correlation
    std_devs = np.sqrt(np.diag(corr))
    corr = corr / np.outer(std_devs, std_devs)
    np.fill_diagonal(corr, 1.0)
    
    decomposer = FactorDecomposer()
    L, D, metrics = decomposer.fit(corr, K=None)
    
    print(f"N=20 adaptive K:")
    print(f"  Selected K: {L.shape[1]}/{N}")
    print(f"  Variance explained: {metrics.variance_explained:.1%}")
    print(f"  Frobenius error: {metrics.frobenius_error:.4f}")
    print(f"  Status: {'✅ PASS' if metrics.frobenius_error < 0.3 else '❌ FAIL'}")
    print()
    
    assert metrics.frobenius_error < 0.3
    assert metrics.variance_explained >= 0.95
    return L.shape[1], metrics.frobenius_error


def test_backward_compatibility():
    """Ensure explicit K still works (backward compatibility)."""
    corr = np.eye(5)
    corr[0, 1] = corr[1, 0] = 0.5
    
    decomposer = FactorDecomposer()
    L, D, metrics = decomposer.fit(corr, K=2)  # Explicit K
    
    print(f"Backward compatibility (explicit K=2):")
    print(f"  Shape: {L.shape}")
    print(f"  Status: ✅ PASS")
    print()
    
    assert L.shape == (5, 2), "Explicit K should override auto-selection"


if __name__ == '__main__':
    print("=" * 70)
    print("COPULA RECONSTRUCTION ERROR FIX VALIDATION")
    print("=" * 70)
    print()
    
    # Test 1: Show the problem
    print("Test 1: Fixed K=3 produces high error (demonstrates problem)")
    print("-" * 70)
    error_before = test_fixed_K_high_error()
    
    # Test 2: Show the fix
    print("Test 2: Adaptive K solves the problem")
    print("-" * 70)
    K_selected, error_after = test_adaptive_K_low_error()
    
    # Test 3: N=10 case
    print("Test 3: N=10 portfolio validation")
    print("-" * 70)
    K_10, error_10 = test_n10_adaptive_K()
    
    # Test 4: N=20 case
    print("Test 4: N=20 portfolio validation")
    print("-" * 70)
    K_20, error_20 = test_n20_adaptive_K()
    
    # Test 5: Backward compatibility
    print("Test 5: Backward compatibility")
    print("-" * 70)
    test_backward_compatibility()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"N=5:  Error reduced {error_before:.4f} → {error_after:.4f} (K={K_selected})")
    print(f"N=10: Error = {error_10:.4f} (K={K_10})")
    print(f"N=20: Error = {error_20:.4f} (K={K_20})")
    print()
    print("✅ All tests passed! Copula reconstruction error < 0.3")
    print("✅ Research paper quality achieved")
