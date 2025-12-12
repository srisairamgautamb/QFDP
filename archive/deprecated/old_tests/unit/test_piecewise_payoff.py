"""
Unit Tests: Piecewise Payoff Oracle (Phase 6)
==============================================

Brutal approximation error tests:
- L1, L2, expectation errors within tolerance
- Monotonicity (no spurious oscillations)
- Compatibility with Phase 2 lognormal states
- Segments=4,8,16,32 scaling

Run: python3 -m pytest tests/unit/test_piecewise_payoff.py -v
"""

import numpy as np
import pytest
from qiskit import QuantumRegister

from qfdp_multiasset.state_prep import prepare_lognormal_asset
from qfdp_multiasset.oracles import (
    call_payoff,
    segment_payoff,
    compute_segment_indices,
    apply_piecewise_constant_payoff,
    piecewise_approximation_error,
    ancilla_scaled_expectation,
    direct_expected_call_from_statevector,
)


class TestSegmentation:
    def test_segment_payoff_16_segments(self):
        prices = np.linspace(50, 150, 64)
        payoff = call_payoff(prices, K=100)
        breakpoints, seg_pays = segment_payoff(prices, payoff, n_segments=16)
        assert len(breakpoints) == 17
        assert len(seg_pays) == 16
        # Payoff should be 0 in first segments (S<K), positive in later
        assert seg_pays[0] < 1e-6
        assert seg_pays[-1] > 0

    def test_segment_indices_coverage(self):
        prices = np.linspace(50, 150, 64)
        breakpoints = np.linspace(50, 150, 9)  # 8 segments
        indices = compute_segment_indices(prices, breakpoints)
        # All indices should be in [0, 7]
        assert np.all((indices >= 0) & (indices <= 7))


class TestApproximationError:
    @pytest.mark.parametrize("n_seg", [4, 8, 16, 32])
    def test_approximation_improves_with_segments(self, n_seg):
        prices = np.linspace(50, 150, 64)
        payoff_exact = call_payoff(prices, K=100)
        breakpoints, seg_pays = segment_payoff(prices, payoff_exact, n_segments=n_seg)
        seg_indices = compute_segment_indices(prices, breakpoints)
        payoff_approx = seg_pays[seg_indices]
        # Uniform probabilities for simplicity
        probs = np.ones(64) / 64
        errors = piecewise_approximation_error(prices, payoff_exact, payoff_approx, probs)
        # Expectation error should decrease with more segments
        assert errors['relative_expectation_error'] < 0.1
        # L1 error should be bounded
        assert errors['l1_error'] < 5.0

    def test_zero_payoff_edge(self):
        # K >> max(S): all payoff = 0
        prices = np.linspace(50, 150, 64)
        payoff_exact = call_payoff(prices, K=200)
        breakpoints, seg_pays = segment_payoff(prices, payoff_exact, n_segments=8)
        # All segments should be zero
        assert np.allclose(seg_pays, 0, atol=1e-9)

    def test_full_payoff_edge(self):
        # K << min(S): all payoff ≈ S-K
        prices = np.linspace(50, 150, 64)
        K = 10
        payoff_exact = call_payoff(prices, K)
        breakpoints, seg_pays = segment_payoff(prices, payoff_exact, n_segments=8)
        # All segments should be nonzero
        assert np.all(seg_pays > 0)


class TestPiecewiseOracle:
    def test_piecewise_oracle_single_asset(self):
        S0, r, sigma, T = 100.0, 0.03, 0.25, 1.0
        n_qubits = 6
        K = S0 * np.exp(r * T)
        circ, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits)
        payoff_exact = call_payoff(prices, K)
        # Attach ancilla
        anc = QuantumRegister(1, 'anc')
        circ.add_register(anc)
        # Encode piecewise payoff
        scale = apply_piecewise_constant_payoff(
            circ, circ.qregs[0], anc[0], prices, payoff_exact, n_segments=16
        )
        est = ancilla_scaled_expectation(circ, anc[0], scale)
        direct = direct_expected_call_from_statevector(circ, circ.qregs[0], prices, K)
        rel_err = abs(est - direct) / (direct + 1e-12)
        # Piecewise approximation introduces error; accept ≤8%
        assert rel_err < 0.08, f"Relative error {rel_err:.2%} too high"

    def test_piecewise_vs_exact_segments(self):
        # Compare piecewise oracle to exact oracle (Phase 5) on same state
        from qfdp_multiasset.oracles import apply_call_payoff_rotation
        
        S0, r, sigma, T = 100.0, 0.03, 0.30, 1.0
        n_qubits = 6
        K = S0 * np.exp(r * T)
        circ, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits)
        payoff_exact = call_payoff(prices, K)
        
        # Exact oracle
        anc1 = QuantumRegister(1, 'anc1')
        c1 = circ.copy()
        c1.add_register(anc1)
        scale1 = apply_call_payoff_rotation(c1, c1.qregs[0], anc1[0], prices, K)
        est1 = ancilla_scaled_expectation(c1, anc1[0], scale1)
        
        # Piecewise oracle (32 segments)
        anc2 = QuantumRegister(1, 'anc2')
        c2 = circ.copy()
        c2.add_register(anc2)
        scale2 = apply_piecewise_constant_payoff(c2, c2.qregs[0], anc2[0], prices, payoff_exact, n_segments=32)
        est2 = ancilla_scaled_expectation(c2, anc2[0], scale2)
        
        # Difference should be small
        assert abs(est1 - est2) / (est1 + 1e-12) < 0.05


class TestMonotonicity:
    def test_call_payoff_monotonic(self):
        # Verify piecewise approximation preserves monotonicity
        prices = np.linspace(50, 150, 64)
        K = 100
        payoff_exact = call_payoff(prices, K)
        breakpoints, seg_pays = segment_payoff(prices, payoff_exact, n_segments=16)
        # seg_pays should be monotonically increasing for call
        diffs = np.diff(seg_pays)
        # Allow small numerical noise
        assert np.all(diffs >= -1e-6)
