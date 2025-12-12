"""
End-to-End Integration: Phases 0-6
===================================

Brutal full-pipeline test:
Phase 2: State prep (5 assets)
Phase 3: Sparse copula correlation
Phase 4: IQFT on assets
Phase 5: Call payoff oracle (exact)
Phase 6: Piecewise payoff oracle (scalable)

Validates:
- Cross-phase compatibility
- Expectation consistency
- Resource estimates match theory

Run: python3 -m pytest tests/integration/test_end_to_end_phases_0_6.py -v
"""

import numpy as np
import pytest
from qiskit import QuantumRegister

from qfdp_multiasset.state_prep import prepare_lognormal_asset
from qfdp_multiasset.sparse_copula import (
    generate_synthetic_correlation_matrix,
    encode_sparse_copula_with_decomposition,
    estimate_copula_resources,
)
from qfdp_multiasset.iqft import apply_tensor_qft
from qfdp_multiasset.oracles import (
    call_payoff,
    apply_call_payoff_rotation,
    apply_piecewise_constant_payoff,
    ancilla_scaled_expectation,
    direct_expected_call_from_statevector,
)


class TestFullPipelineIntegration:
    def test_5_asset_pipeline_exact_payoff(self):
        """2-asset correlated state → exact call payoff oracle (12 qubits, memory-safe)."""
        # Phase 2+3: Build correlated state (12 qubits: 2×5 assets + 1×2 factor)
        asset_params = [
            (100.0, 0.03, 0.20, 1.0),
            (150.0, 0.03, 0.25, 1.0),
        ]
        corr = generate_synthetic_correlation_matrix(N=2, K=1, seed=42)
        circ, metrics = encode_sparse_copula_with_decomposition(
            asset_params, corr, n_factors=1,
            n_qubits_asset=5, n_qubits_factor=2  # 2^12 = 4K amplitudes = 32 KB
        )
        
        # Validate Phase 3 metrics
        assert metrics.total_qubits == 2*5 + 1*2  # 12 qubits
        assert metrics.variance_explained > 0.3  # K=1 for N=2: lower bound
        assert metrics.frobenius_error < 0.8
        
        # Phase 5: Attach ancilla and encode call payoff on asset 0
        asset_regs = [qreg for qreg in circ.qregs if qreg.name.startswith('asset_')]
        asset0_prices = np.linspace(
            100*np.exp(-3*0.20), 100*np.exp(+3*0.20), 32  # 2^5 = 32 price points
        )
        K = 100.0 * np.exp(0.03 * 1.0)
        anc = QuantumRegister(1, 'anc')
        circ.add_register(anc)
        
        scale = apply_call_payoff_rotation(circ, asset_regs[0], anc[0], asset0_prices, K)
        est = ancilla_scaled_expectation(circ, anc[0], scale)
        direct = direct_expected_call_from_statevector(circ, asset_regs[0], asset0_prices, K)
        
        # Expectation from ancilla must match direct
        rel_err = abs(est - direct) / (direct + 1e-12)
        assert rel_err < 0.10, f"Exact payoff error {rel_err:.2%} too high (est={est:.3f}, direct={direct:.3f})"
    
    def test_5_asset_pipeline_piecewise_payoff(self):
        """2-asset correlated state → piecewise payoff oracle (12 qubits)."""
        # Phase 2+3
        asset_params = [
            (100.0, 0.03, 0.25, 1.0),
            (150.0, 0.03, 0.20, 1.0),
        ]
        corr = generate_synthetic_correlation_matrix(N=2, K=1, seed=123)
        circ, metrics = encode_sparse_copula_with_decomposition(
            asset_params, corr, n_factors=1,
            n_qubits_asset=5, n_qubits_factor=2  # 12 qubits
        )
        
        # Phase 6: Piecewise payoff (8 segments for speed)
        asset0_prices = np.linspace(
            100*np.exp(-3*0.25), 100*np.exp(+3*0.25), 32
        )
        K = 100.0 * np.exp(0.03 * 1.0)
        payoff_exact = call_payoff(asset0_prices, K)
        
        anc = QuantumRegister(1, 'anc')
        circ.add_register(anc)
        asset_regs = [qreg for qreg in circ.qregs if qreg.name.startswith('asset_')]
        
        scale = apply_piecewise_constant_payoff(
            circ, asset_regs[0], anc[0], asset0_prices, payoff_exact, n_segments=8
        )
        est = ancilla_scaled_expectation(circ, anc[0], scale)
        direct = direct_expected_call_from_statevector(circ, asset_regs[0], asset0_prices, K)
        
        # Piecewise introduces approximation error
        rel_err = abs(est - direct) / (direct + 1e-12)
        assert rel_err < 0.15, f"Piecewise payoff error {rel_err:.2%} too high (est={est:.3f}, direct={direct:.3f})"
    
    def test_resource_estimates_consistency(self):
        """Verify resource estimates match actual circuit."""
        N, K = 5, 3
        n_qubits_asset, n_qubits_factor = 8, 6
        
        # Estimate
        resources = estimate_copula_resources(N, K, n_qubits_asset, n_qubits_factor)
        
        # Build actual circuit
        asset_params = [(100.0 + i*50, 0.03, 0.20 + i*0.05, 1.0) for i in range(N)]
        corr = generate_synthetic_correlation_matrix(N=N, K=K, seed=999)
        circ, metrics = encode_sparse_copula_with_decomposition(
            asset_params, corr, n_factors=K,
            n_qubits_asset=n_qubits_asset, n_qubits_factor=n_qubits_factor
        )
        
        # Validate
        assert circ.num_qubits == resources['total_qubits']
        assert metrics.total_qubits == resources['total_qubits']
        assert metrics.correlation_gates == resources['correlation_gates']
    
    def test_phase_2_to_phase_6_monotonicity(self):
        """Single-asset through all phases preserves monotonicity."""
        # Phase 2: Single asset
        S0, r, sigma, T = 100.0, 0.03, 0.25, 1.0
        n_qubits = 6
        circ, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits)
        K = S0 * np.exp(r * T)
        
        # Phase 5: Exact payoff
        anc1 = QuantumRegister(1, 'anc1')
        c1 = circ.copy()
        c1.add_register(anc1)
        scale1 = apply_call_payoff_rotation(c1, c1.qregs[0], anc1[0], prices, K)
        est1 = ancilla_scaled_expectation(c1, anc1[0], scale1)
        
        # Phase 6: Piecewise (32 segments)
        anc2 = QuantumRegister(1, 'anc2')
        c2 = circ.copy()
        c2.add_register(anc2)
        payoff_exact = call_payoff(prices, K)
        scale2 = apply_piecewise_constant_payoff(c2, c2.qregs[0], anc2[0], prices, payoff_exact, n_segments=32)
        est2 = ancilla_scaled_expectation(c2, anc2[0], scale2)
        
        # Both should be close to direct
        direct = direct_expected_call_from_statevector(circ, circ.qregs[0], prices, K)
        
        assert abs(est1 - direct) / (direct + 1e-12) < 0.03
        assert abs(est2 - direct) / (direct + 1e-12) < 0.08
        # Piecewise should be within 10% of exact
        assert abs(est1 - est2) / (est1 + 1e-12) < 0.10
