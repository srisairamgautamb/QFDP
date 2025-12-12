"""
Unit Tests: Call Payoff Ancilla Rotation (Phase 5)
==================================================

- Single-asset expectation matches direct marginal sum
- Multi-asset (Phase 3) compatibility: expectation from ancilla equals direct from statevector
- Random K stress

Run: python3 -m pytest tests/unit/test_call_payoff_oracle.py -v
"""

import numpy as np
import pytest
from qiskit import QuantumRegister

from qfdp_multiasset.state_prep import prepare_lognormal_asset
from qfdp_multiasset.sparse_copula import (
    generate_synthetic_correlation_matrix,
    encode_sparse_copula_with_decomposition,
)
from qfdp_multiasset.oracles import (
    apply_call_payoff_rotation,
    ancilla_scaled_expectation,
    direct_expected_call_from_statevector,
)


class TestSingleAssetCall:
    def test_single_asset_call_matches_direct(self):
        S0, r, sigma, T = 100.0, 0.03, 0.25, 1.0
        n_qubits = 6
        K = S0 * np.exp(r * T)
        circ, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits)
        # Attach ancilla
        anc = QuantumRegister(1, 'anc')
        circ.add_register(anc)
        # Encode payoff
        scale = apply_call_payoff_rotation(circ, circ.qregs[0], anc[0], prices, K)
        est = ancilla_scaled_expectation(circ, anc[0], scale)
        direct = direct_expected_call_from_statevector(circ, circ.qregs[0], prices, K)
        rel_err = abs(est - direct) / (direct + 1e-12)
        assert rel_err < 0.02, f"Relative error {rel_err:.2%} too high"

    def test_random_strikes(self):
        S0, r, sigma, T = 100.0, 0.03, 0.30, 1.0
        n_qubits = 6
        circ, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits)
        anc = QuantumRegister(1, 'anc')
        circ.add_register(anc)
        rng = np.random.default_rng(2026)
        for _ in range(5):
            K = float(rng.uniform(prices[0], prices[-1]))
            c2 = circ.copy()
            c2.add_register(QuantumRegister(1, 'a2'))
            anc2 = c2.qubits[-1]
            scale = apply_call_payoff_rotation(c2, c2.qregs[0], anc2, prices, K)
            est = ancilla_scaled_expectation(c2, anc2, scale)
            direct = direct_expected_call_from_statevector(c2, c2.qregs[0], prices, K)
            assert abs(est - direct) / (direct + 1e-12) < 0.03


class TestPhase3Compatibility:
    def test_3_asset_copula_call_on_asset0(self):
        # Build correlated 3-asset state
        asset_params = [
            (100.0, 0.03, 0.25, 1.0),
            (150.0, 0.03, 0.20, 1.0),
            (200.0, 0.03, 0.30, 1.0),
        ]
        corr = generate_synthetic_correlation_matrix(N=3, K=2, seed=5)
        circ, metrics = encode_sparse_copula_with_decomposition(
            asset_params, corr, n_factors=2,
            n_qubits_asset=4, n_qubits_factor=3
        )
        # Choose asset 0
        K = 100.0 * np.exp(0.03 * 1.0)
        prices0 = np.linspace(  # reconstruct price grid for n=4 from state_prep logic (approximate)
            100*np.exp(-3*0.25), 100*np.exp(+3*0.25), 16
        )
        # Attach ancilla
        anc = QuantumRegister(1, 'anc')
        circ.add_register(anc)
        # Encode payoff on asset_0 register
        asset0_reg = [qreg for qreg in circ.qregs if qreg.name.startswith('asset_')][0]
        scale = apply_call_payoff_rotation(circ, asset0_reg, anc[0], prices0, K)
        est = ancilla_scaled_expectation(circ, anc[0], scale)
        # Direct expectation from statevector marginal on asset0
        direct = direct_expected_call_from_statevector(circ, asset0_reg, prices0, K)
        # The ancilla-encoded expectation must match the direct marginal expectation
        assert abs(est - direct) / (direct + 1e-12) < 0.05
