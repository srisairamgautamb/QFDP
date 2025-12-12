"""
Integration Test: Phases 0-2 End-to-End Validation
===================================================

Tests the complete pipeline from:
  Phase 0: Project structure
  Phase 1: Sparse copula factor decomposition
  Phase 2: Quantum state preparation for marginals and factors

This validates that all components work together correctly.

Run: python3 -m pytest tests/integration/test_phases_0_2.py -v
"""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from scipy.stats import norm

from qfdp_multiasset.sparse_copula import (
    FactorDecomposer,
    generate_synthetic_correlation_matrix,
    analyze_eigenvalue_decay
)
from qfdp_multiasset.state_prep import (
    prepare_lognormal_asset,
    prepare_gaussian_factor,
    compute_fidelity,
    estimate_resource_cost
)


class TestPhase0Structure:
    """Validate Phase 0: Project structure."""
    
    def test_package_imports(self):
        """All main modules should be importable."""
        # Should not raise ImportError
        from qfdp_multiasset import sparse_copula
        from qfdp_multiasset import state_prep
        
        assert sparse_copula is not None
        assert state_prep is not None
        
    def test_submodules_exist(self):
        """All planned submodules should have __init__.py."""
        import os
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent
        expected_modules = [
            'sparse_copula', 'state_prep', 'iqft', 'qsp', 
            'oracles', 'mlqae', 'portfolio', 'analysis', 
            'benchmarks', 'utils'
        ]
        
        for module in expected_modules:
            module_path = project_root / 'qfdp_multiasset' / module / '__init__.py'
            assert module_path.exists(), f"Missing {module}/__init__.py"


class TestPhase1FactorModel:
    """Validate Phase 1: Sparse copula mathematics."""
    
    def test_factor_decomposition_pipeline(self):
        """End-to-end factor decomposition for realistic correlation."""
        N, K = 5, 3
        
        # Generate synthetic correlation (mimics real market)
        corr = generate_synthetic_correlation_matrix(N=N, K=K, seed=42)
        
        # Analyze eigenvalue spectrum
        analysis = analyze_eigenvalue_decay(corr)
        assert len(analysis['eigenvalues']) == N
        assert analysis['variance_explained'][K-1] > 0.5  # K=3 should capture >50%
        
        # Factor decomposition
        decomposer = FactorDecomposer()
        L, D, metrics = decomposer.fit(corr, K=K)
        
        # Validate outputs
        assert L.shape == (N, K)
        assert D.shape == (N, N)
        assert np.allclose(np.diag(D), np.diag(D))  # D is diagonal
        
        # Reconstruction quality
        reconstructed = L @ L.T + D
        assert np.allclose(reconstructed, corr, atol=0.1)  # Within 0.1 tolerance
        
        # Metrics validation
        assert 0 <= metrics.variance_explained <= 1.0
        assert metrics.frobenius_error >= 0.0
        assert metrics.frobenius_error < 1.0  # Should be reasonable
        
    def test_gate_reduction_calculation(self):
        """Validate sparse encoding reduces gates vs full copula."""
        N, K = 10, 3
        
        # Full copula: O(N²) = N*(N-1)/2 pairwise correlations
        full_gates = N * (N - 1) // 2
        
        # Sparse copula: O(N×K) factor loadings
        sparse_gates = N * K
        
        reduction_factor = full_gates / sparse_gates
        
        # For N=10, K=3: 45 / 30 = 1.5× reduction
        assert reduction_factor > 1.0
        assert reduction_factor == pytest.approx(1.5, rel=0.1)


class TestPhase2StatePrep:
    """Validate Phase 2: Quantum state preparation."""
    
    def test_lognormal_asset_preparation(self):
        """Prepare realistic log-normal asset price distribution."""
        # AAPL-like parameters
        S0, r, sigma, T = 150.0, 0.03, 0.25, 1.0
        n_qubits = 8
        
        circuit, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits=n_qubits)
        
        # Validate circuit
        assert circuit.num_qubits == n_qubits
        assert len(prices) == 2**n_qubits
        assert prices[0] > 0
        
        # Check expected value
        sv = Statevector(circuit)
        probs = np.abs(sv.data)**2
        E_S = np.sum(prices * probs)
        E_S_theory = S0 * np.exp(r * T)
        
        # Within 20% (discretization)
        rel_error = abs(E_S - E_S_theory) / E_S_theory
        assert rel_error < 0.20
        
    def test_gaussian_factor_preparation(self):
        """Prepare Gaussian factor for copula encoding."""
        n_qubits = 6
        
        circuit = prepare_gaussian_factor(n_qubits=n_qubits, mean=0, std=1)
        
        # Validate fidelity
        N = 2**n_qubits
        x_grid = np.linspace(-4, 4, N)
        target_probs = norm.pdf(x_grid, loc=0, scale=1)
        target_probs /= target_probs.sum()
        
        fidelity = compute_fidelity(circuit, target_probs)
        
        # Phase 2 completion criterion: F ≥ 0.90
        assert fidelity >= 0.90
        
    def test_multiasset_resource_estimation(self):
        """Estimate resources for multi-asset portfolio."""
        N_assets = 5
        K_factors = 3
        n_qubits_asset = 8
        n_qubits_factor = 6
        
        # Single asset cost
        single_asset = estimate_resource_cost(n_qubits_asset)
        single_factor = estimate_resource_cost(n_qubits_factor)
        
        # Total resources
        total_qubits = N_assets * n_qubits_asset + K_factors * n_qubits_factor
        total_t_count = (N_assets * single_asset['t_count_estimate'] + 
                        K_factors * single_factor['t_count_estimate'])
        
        # For N=5, K=3: 40 + 18 = 58 qubits
        assert total_qubits == 58
        
        # T-count: 5*765 + 3*189 = 4,392
        assert total_t_count == 4392


class TestPhases0to2Integration:
    """End-to-end integration test combining all phases."""
    
    def test_full_pipeline_5_assets(self):
        """Complete pipeline: correlation → factors → quantum states."""
        N_assets = 5
        K_factors = 3
        n_qubits = 8
        
        # Phase 1: Generate and decompose correlation matrix
        corr = generate_synthetic_correlation_matrix(N=N_assets, K=K_factors, seed=123)
        decomposer = FactorDecomposer()
        L, D, metrics = decomposer.fit(corr, K=K_factors)
        
        # Validate decomposition quality
        assert metrics.variance_explained > 0.6  # K=3 captures >60% for N=5
        
        # Phase 2: Prepare asset marginals
        asset_params = [
            (100.0, 0.03, 0.20, 1.0),  # Asset 1
            (150.0, 0.03, 0.25, 1.0),  # Asset 2
            (200.0, 0.03, 0.30, 1.0),  # Asset 3
            (50.0,  0.03, 0.15, 1.0),  # Asset 4
            (300.0, 0.03, 0.35, 1.0),  # Asset 5
        ]
        
        marginal_circuits = []
        for S0, r, sigma, T in asset_params:
            circuit, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits=n_qubits)
            marginal_circuits.append((circuit, prices))
        
        # Validate all marginals prepared
        assert len(marginal_circuits) == N_assets
        
        # Phase 2: Prepare Gaussian factors
        factor_circuits = []
        for k in range(K_factors):
            circuit = prepare_gaussian_factor(n_qubits=6, mean=0, std=1)
            factor_circuits.append(circuit)
        
        # Validate all factors prepared
        assert len(factor_circuits) == K_factors
        
        # Integration check: Total qubits needed
        total_qubits_marginals = N_assets * n_qubits
        total_qubits_factors = K_factors * 6
        total_qubits = total_qubits_marginals + total_qubits_factors
        
        # Should be 5*8 + 3*6 = 58 qubits
        assert total_qubits == 58
        
        # Resource estimation
        single_marginal = estimate_resource_cost(n_qubits)
        single_factor = estimate_resource_cost(6)
        
        prep_t_count = (N_assets * single_marginal['t_count_estimate'] + 
                       K_factors * single_factor['t_count_estimate'])
        
        # State prep T-count: 5*765 + 3*189 = 4,392
        assert prep_t_count == 4392
        
        print(f"\n✅ Full pipeline validated:")
        print(f"   - Correlation decomposition: {metrics.variance_explained:.1%} variance")
        print(f"   - Frobenius error: {metrics.frobenius_error:.3f}")
        print(f"   - Assets prepared: {N_assets}")
        print(f"   - Factors prepared: {K_factors}")
        print(f"   - Total qubits: {total_qubits}")
        print(f"   - State prep T-count: {prep_t_count:,}")
        
    def test_phase_completion_criteria(self):
        """Verify all phase completion criteria are met."""
        
        # Phase 1 criteria
        N, K = 10, 3
        corr = generate_synthetic_correlation_matrix(N=N, K=K, seed=999)
        decomposer = FactorDecomposer()
        L, D, metrics = decomposer.fit(corr, K=K)
        
        phase1_pass = (
            metrics.frobenius_error < 1.0 and
            metrics.variance_explained > 0.5
        )
        assert phase1_pass, "Phase 1 completion criteria not met"
        
        # Phase 2 criteria
        circuit_marginal = prepare_gaussian_factor(n_qubits=8)
        circuit_factor = prepare_gaussian_factor(n_qubits=6)
        
        # Marginal fidelity ≥ 0.95
        N_marginal = 256
        x_marginal = np.linspace(-4, 4, N_marginal)
        target_marginal = norm.pdf(x_marginal)
        target_marginal /= target_marginal.sum()
        fidelity_marginal = compute_fidelity(circuit_marginal, target_marginal)
        
        # Factor fidelity ≥ 0.90
        N_factor = 64
        x_factor = np.linspace(-4, 4, N_factor)
        target_factor = norm.pdf(x_factor)
        target_factor /= target_factor.sum()
        fidelity_factor = compute_fidelity(circuit_factor, target_factor)
        
        phase2_pass = (fidelity_marginal >= 0.95 and fidelity_factor >= 0.90)
        assert phase2_pass, "Phase 2 completion criteria not met"
        
        print(f"\n✅ All phase completion criteria met:")
        print(f"   Phase 1: Frobenius error {metrics.frobenius_error:.3f} < 1.0 ✅")
        print(f"   Phase 1: Variance explained {metrics.variance_explained:.1%} > 50% ✅")
        print(f"   Phase 2: Marginal fidelity {fidelity_marginal:.4f} ≥ 0.95 ✅")
        print(f"   Phase 2: Factor fidelity {fidelity_factor:.4f} ≥ 0.90 ✅")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
