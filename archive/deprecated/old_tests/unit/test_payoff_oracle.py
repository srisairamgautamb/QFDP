"""
Unit Tests: Payoff Oracle (Digital Threshold) - Phase 5 Initial
===============================================================

Brutal tests for ancilla-marking digital payoff oracle:
- Single-asset: P(S>=K) via ancilla probability matches classical
- Edge cases: K << min(prices), K >> max(prices)
- Randomized thresholds

Run: python3 -m pytest tests/unit/test_payoff_oracle.py -v
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from qfdp_multiasset.state_prep import prepare_lognormal_asset
from qfdp_multiasset.oracles import (
    mark_threshold_states,
    ancilla_probability,
    classical_threshold_probability,
)


class TestDigitalPayoffOracle:
    def build_asset_state(self, S0=100.0, r=0.03, sigma=0.2, T=1.0, n_qubits=6):
        circ, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits=n_qubits)
        return circ, prices

    def attach_ancilla(self, circ: QuantumCircuit) -> Qubit:
        anc = QuantumRegister(1, 'anc')
        circ.add_register(anc)
        return anc[0]

    @pytest.mark.parametrize("n_qubits", [4, 6])
    def test_probability_matches_classical(self, n_qubits):
        S0, r, sigma, T = 100.0, 0.03, 0.25, 1.0
        circ, prices = self.build_asset_state(S0, r, sigma, T, n_qubits)

        # Build classical probabilities used to construct state
        # Reconstruct from prepare_lognormal_asset logic
        mu = (r - 0.5 * sigma**2) * T
        sigma_r = sigma * np.sqrt(T)
        log_returns = np.log(prices / S0)
        pdf_log = (1.0 / (np.sqrt(2*np.pi) * sigma_r)) * np.exp(-0.5 * ((log_returns - mu) / sigma_r)**2)
        pdf_price = pdf_log / prices
        classical_probs = pdf_price / pdf_price.sum()

        # Choose K at-the-money
        K = S0 * np.exp(r * T)

        # Attach ancilla and mark threshold
        anc = self.attach_ancilla(circ)
        count = mark_threshold_states(circ, circ.qregs[0], anc, prices, K)
        assert count > 0

        # Compute ancilla probability
        p_quantum = ancilla_probability(circ, anc)
        p_classical = classical_threshold_probability(prices, classical_probs, K)

        assert abs(p_quantum - p_classical) < 0.05, f"Quantum {p_quantum:.4f} vs Classical {p_classical:.4f}"

    def test_deep_in_the_money(self):
        # High K -> probability ~0
        S0, r, sigma, T = 100.0, 0.03, 0.2, 1.0
        circ, prices = self.build_asset_state(S0, r, sigma, T, n_qubits=6)
        anc = self.attach_ancilla(circ)

        K = prices[-1] * 10  # Far above max
        count = mark_threshold_states(circ, circ.qregs[0], anc, prices, K)
        assert count == 0
        p = ancilla_probability(circ, anc)
        assert p < 1e-9

    def test_deep_out_of_the_money(self):
        # Low K -> probability ~1
        S0, r, sigma, T = 100.0, 0.03, 0.2, 1.0
        circ, prices = self.build_asset_state(S0, r, sigma, T, n_qubits=6)
        anc = self.attach_ancilla(circ)

        K = prices[0] / 10  # Far below min
        count = mark_threshold_states(circ, circ.qregs[0], anc, prices, K)
        assert count == len(prices)
        p = ancilla_probability(circ, anc)
        assert 1.0 - p < 1e-9

    def test_randomized_thresholds(self):
        S0, r, sigma, T = 100.0, 0.03, 0.3, 1.0
        circ, prices = self.build_asset_state(S0, r, sigma, T, n_qubits=6)

        mu = (r - 0.5 * sigma**2) * T
        sigma_r = sigma * np.sqrt(T)
        log_returns = np.log(prices / S0)
        pdf_log = (1.0 / (np.sqrt(2*np.pi) * sigma_r)) * np.exp(-0.5 * ((log_returns - mu) / sigma_r)**2)
        pdf_price = pdf_log / prices
        classical_probs = pdf_price / pdf_price.sum()

        anc = self.attach_ancilla(circ)

        rng = np.random.default_rng(2025)
        for _ in range(5):
            # Random K within price range
            K = float(rng.uniform(low=prices[0], high=prices[-1]))
            # Rebuild oracle on a fresh copy to avoid compounding gates
            circ_k = circ.copy()
            anc_k = Qubit()  # placeholder, we need proper ancilla per copy
            # Instead of copying, re-attach new ancilla per iteration
            circ_k.add_register(QuantumRegister(1, 'anc_k'))
            anc_k = circ_k.qubits[-1]

            mark_threshold_states(circ_k, circ_k.qregs[0], anc_k, prices, K)
            p_q = ancilla_probability(circ_k, anc_k)
            p_c = classical_threshold_probability(prices, classical_probs, K)
            assert abs(p_q - p_c) < 0.08
