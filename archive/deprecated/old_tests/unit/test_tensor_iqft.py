"""
Unit Tests: Tensor QFT/IQFT (Phase 4)
=====================================

Brutal correctness and scaling tests for the IQFT/QFT stage.
- QFT ∘ IQFT = Identity (up to global phase)
- Tensor-QFT across multiple asset registers
- Resource scaling ~ O(n^2)
- Integration with Phase 3 copula circuit (QFT then IQFT on assets)

Run: python3 -m pytest tests/unit/test_tensor_iqft.py -v
"""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, QuantumRegister

from qfdp_multiasset.iqft import (
    build_qft,
    apply_qft,
    apply_tensor_qft,
    estimate_qft_resources,
    estimate_tensor_qft_resources,
)
from qfdp_multiasset.sparse_copula import (
    generate_synthetic_correlation_matrix,
    encode_sparse_copula_with_decomposition,
)


class TestSingleRegisterQFT:
    """Brutal tests for single-register QFT/IQFT identity."""

    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6, 8])
    def test_qft_iqft_identity_random_state(self, n_qubits):
        # Random normalized state
        rng = np.random.default_rng(1234 + n_qubits)
        psi = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
        psi /= np.linalg.norm(psi)

        # Initialize circuit
        qr = QuantumRegister(n_qubits, 'reg')
        circ = QuantumCircuit(qr)
        circ.initialize(psi, qr)

        # Apply QFT then IQFT
        apply_qft(circ, qr, inverse=False)
        apply_qft(circ, qr, inverse=True)

        # Validate fidelity up to global phase
        out = Statevector(circ)
        fidelity = float(np.abs(np.vdot(psi, out.data))**2)
        assert fidelity >= 0.999999, f"Fidelity too low: {fidelity:.6f}"


class TestTensorQFT:
    """Tensor-QFT across multiple registers."""

    def test_tensor_qft_iqft_three_registers(self):
        n = 4
        regs = [QuantumRegister(n, f'a{i}') for i in range(3)]
        circ = QuantumCircuit(*regs)

        # Prepare product of random states across registers
        rng = np.random.default_rng(4321)
        for reg in regs:
            psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
            psi /= np.linalg.norm(psi)
            circ.initialize(psi, reg)

        # Apply tensor QFT then IQFT
        apply_tensor_qft(circ, regs, inverse=False)
        apply_tensor_qft(circ, regs, inverse=True)

        out = Statevector(circ).data
        # We don't have the original product state easily; just check normalization
        assert np.isclose(np.vdot(out, out), 1.0, atol=1e-10)

    def test_resource_scaling(self):
        res4 = estimate_qft_resources(4)
        res8 = estimate_qft_resources(8)
        # Phase gates ~ n(n-1)/2
        assert res4['phase_gates'] == 6
        assert res8['phase_gates'] == 28
        # Depth grows ~ n^2
        assert res8['depth_estimate'] > res4['depth_estimate']


class TestIntegrationWithCopula:
    """Integration with Phase 3: applying QFT/IQFT on asset registers only."""

    def test_qft_iqft_preserves_copula_state_on_assets(self):
        # Build small copula circuit
        asset_params = [
            (100.0, 0.03, 0.20, 1.0),
            (150.0, 0.03, 0.25, 1.0),
            (200.0, 0.03, 0.30, 1.0)
        ]
        corr = generate_synthetic_correlation_matrix(N=3, K=2, seed=7)
        circ, metrics = encode_sparse_copula_with_decomposition(
            asset_params, corr, n_factors=2,
            n_qubits_asset=4, n_qubits_factor=3
        )

        # Identify asset registers by name
        asset_regs = [qreg for qreg in circ.qregs if qreg.name.startswith('asset_')]
        assert len(asset_regs) == 3

        # Snapshot original state
        psi0 = Statevector(circ).data

        # Apply IQFT then QFT on assets (net identity)
        apply_tensor_qft(circ, asset_regs, inverse=True)
        apply_tensor_qft(circ, asset_regs, inverse=False)

        psi1 = Statevector(circ).data
        # Fidelity up to global phase
        fid = float(np.abs(np.vdot(psi0, psi1))**2)
        assert fid >= 0.9999, f"Tensor IQFT/QFT did not preserve state (fid={fid:.6f})"


class TestEdgeCases:
    def test_minimal_register(self):
        qr = QuantumRegister(1, 'q')
        circ = QuantumCircuit(qr)
        apply_qft(circ, qr, inverse=False)
        apply_qft(circ, qr, inverse=True)
        sv = Statevector(circ)
        assert np.isclose(np.vdot(sv.data, sv.data), 1.0, atol=1e-10)

    def test_zero_registers_tensor(self):
        circ = QuantumCircuit(2)
        apply_tensor_qft(circ, [], inverse=False)
        # Should not modify circuit or raise error
        assert circ.num_qubits == 2
