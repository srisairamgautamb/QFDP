import numpy as np
import pytest

from qfdp_multiasset.sparse_copula import (
    FactorDecomposer,
    generate_synthetic_correlation_matrix,
    analyze_eigenvalue_decay,
)


def test_identity_matrix_k0_invalid():
    N = 5
    corr = np.eye(N)
    decomposer = FactorDecomposer()
    with pytest.raises(ValueError):
        decomposer.fit(corr, K=0)


def test_identity_matrix_k1_ok():
    N = 5
    corr = np.eye(N)
    decomposer = FactorDecomposer()
    L, D, metrics = decomposer.fit(corr, K=1)
    # Identity with K=1: captures 1/N variance, rest goes to D
    # Reconstruction: L @ L.T + D ≈ I (by definition)
    reconstructed = L @ L.T + D
    assert np.allclose(reconstructed, corr, atol=1e-8)
    # Frobenius error should be reasonable (not zero for K=1 < N)
    assert metrics.frobenius_error >= 0.0


def test_symmetric_positive_definite_validation():
    N = 4
    corr = np.eye(N)
    corr[0, 1] = 0.5
    corr[1, 0] = 0.5
    decomposer = FactorDecomposer()
    L, D, metrics = decomposer.fit(corr, K=1)
    assert L.shape == (N, 1)
    assert D.shape == (N, N)
    assert 0.0 <= metrics.variance_explained <= 1.0


def test_generate_synthetic_correlation_matrix_properties():
    corr = generate_synthetic_correlation_matrix(N=10, K=3, seed=123)
    assert corr.shape == (10, 10)
    assert np.allclose(corr, corr.T, atol=1e-8)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-8)


def test_analyze_eigenvalue_decay_monotonic():
    corr = generate_synthetic_correlation_matrix(N=12, K=3, seed=42)
    analysis = analyze_eigenvalue_decay(corr)
    eigvals = analysis['eigenvalues']
    # Sorted descending
    assert np.all(eigvals[:-1] >= eigvals[1:] - 1e-12)
    var = analysis['variance_explained']
    # Cumulative variance must be non-decreasing and <= 1
    assert np.all(var[:-1] <= var[1:] + 1e-12)
    assert var[-1] <= 1.0 + 1e-12


def test_portfolio_variance_error_bound_reasonable():
    N, K = 8, 3
    corr = generate_synthetic_correlation_matrix(N=N, K=K, seed=7)
    decomposer = FactorDecomposer()
    L, D, metrics = decomposer.fit(corr, K=K)
    w = np.ones(N) / N  # equal-weight
    results = decomposer.compute_portfolio_variance_error(L, D, w)
    # Bound from Lemma B (heuristic check): absolute_error <= ||Σ - Σ_K||_F / N
    frob = metrics.frobenius_error
    assert results['absolute_error'] <= frob / N + 1e-6
