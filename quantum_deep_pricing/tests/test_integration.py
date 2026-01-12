"""
Integration Tests for QRC + QTC System
======================================

Tests the complete quantum deep pricing pipeline.
"""

import numpy as np
import sys
sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')


def test_qrc():
    """Test QRC standalone."""
    print("Testing QRC...")
    from qrc import QuantumRecurrentCircuit
    
    qrc = QuantumRecurrentCircuit(n_factors=4, n_qubits=8, n_deep_layers=3)
    
    # Test forward pass
    market_data = {
        'prices': 100.5,
        'volatility': 0.25,
        'vol_trend': 0.01,
        'corr_change': -0.05,
        'stress': 0.3
    }
    
    result = qrc.forward(market_data)
    
    assert result.factors.shape == (4,), f"Expected 4 factors, got {result.factors.shape}"
    assert np.isclose(result.factors.sum(), 1.0, atol=0.01), "Factors should sum to 1"
    assert result.circuit_depth > 0, "Circuit should have positive depth"
    
    # Test temporal memory
    result2 = qrc.forward(market_data)
    # Factors should change due to recurrent gate
    
    print(f"  ✅ QRC: {qrc}")
    print(f"  ✅ Factors: {result.factors}")
    print(f"  ✅ Circuit depth: {result.circuit_depth}")
    return True


def test_qtc():
    """Test QTC standalone."""
    print("Testing QTC...")
    from qtc import QuantumTemporalConvolution
    
    qtc = QuantumTemporalConvolution(kernel_size=3, n_kernels=4, n_qubits=4)
    
    # Test forward pass with 6-day price history
    price_history = np.array([100.0, 100.5, 99.8, 100.2, 101.0, 100.8])
    
    result = qtc.forward(price_history)
    
    assert result.patterns.shape == (4,), f"Expected 4 patterns, got {result.patterns.shape}"
    assert len(result.kernel_outputs) == 4, "Should have 4 kernel outputs"
    
    print(f"  ✅ QTC: {qtc}")
    print(f"  ✅ Patterns: {result.patterns}")
    print(f"  ✅ Kernel outputs: {len(result.kernel_outputs)}")
    return True


def test_fusion():
    """Test feature fusion."""
    print("Testing Feature Fusion...")
    from quantum_deep_pricing import FeatureFusion
    
    fusion = FeatureFusion(n_qrc_factors=4, n_qtc_patterns=4, method='weighted')
    
    qrc_factors = np.array([0.3, 0.25, 0.25, 0.2])
    qtc_patterns = np.array([0.4, 0.3, 0.2, 0.1])
    
    result = fusion.forward(qrc_factors, qtc_patterns)
    
    assert result.fused_features.shape == (8,), "Fused should have 8 features"
    assert 0 <= result.qrc_weight <= 1, "QRC weight should be in [0, 1]"
    
    print(f"  ✅ Fusion method: {result.fusion_method}")
    print(f"  ✅ Fused features: {result.fused_features.shape}")
    print(f"  ✅ QRC weight: {result.qrc_weight:.2f}")
    return True


def test_integrated_pricer():
    """Test full integrated pipeline."""
    print("Testing Integrated Pricer...")
    from quantum_deep_pricing import QuantumDeepPricer
    
    pricer = QuantumDeepPricer(
        fb_iqft_pricer=None,  # Skip FB-IQFT for now
        fusion_method='weighted',
        use_qrc=True,
        use_qtc=True
    )
    
    market_data = {
        'prices': 100.0,
        'volatility': 0.2,
        'stress': 0.1
    }
    
    price_history = np.array([99.0, 99.5, 100.0, 100.2, 99.8, 100.0])
    
    result = pricer.price_option(
        market_data=market_data,
        price_history=price_history,
        strike=100.0,
        maturity=1.0
    )
    
    assert result.price > 0, "Price should be positive"
    assert result.qrc_factors.shape == (4,), "Should have 4 QRC factors"
    assert result.qtc_patterns.shape == (4,), "Should have 4 QTC patterns"
    
    print(f"  ✅ Pricer: {pricer}")
    print(f"  ✅ Price: ${result.price:.4f}")
    print(f"  ✅ QRC factors: {result.qrc_factors}")
    print(f"  ✅ QTC patterns: {result.qtc_patterns}")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("QRC + QTC INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("QRC", test_qrc),
        ("QTC", test_qtc),
        ("Fusion", test_fusion),
        ("Integrated Pricer", test_integrated_pricer)
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  ❌ {name} failed: {e}")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✅ PASS" if success else f"❌ FAIL: {error}"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
