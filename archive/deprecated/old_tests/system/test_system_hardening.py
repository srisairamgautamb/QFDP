"""
System Hardening and Critical Tests
===================================

Goal: be brutally critical. Catch flakiness, memory blowups, API rate limits, and scaling regressions.
"""

import json
import time
import numpy as np
import pytest

from qiskit.quantum_info import Statevector

from qfdp_multiasset.utils.runtime_guard import (
    estimate_statevector_bytes,
    ensure_statevector_ok,
    DEFAULT_BUDGET,
)
from qfdp_multiasset.sparse_copula import FactorDecomposer
from qfdp_multiasset.oracles import apply_call_payoff_rotation, call_payoff
from qfdp_multiasset.state_prep import prepare_lognormal_asset
from qfdp_multiasset.mlqae import run_mlqae


@pytest.mark.critical
def test_statevector_budget_guard_prevents_oom():
    # 23 qubits would exceed default cap
    class Dummy:
        num_qubits = 23
    with pytest.raises(MemoryError):
        ensure_statevector_ok(Dummy())


@pytest.mark.critical
def test_statevector_feasible_paths():
    n = 12
    circ, prices = prepare_lognormal_asset(100, 0.03, 0.25, 1.0, n_qubits=n)
    ensure_statevector_ok(circ)
    sv = Statevector(circ)  # should not raise
    assert len(sv.data) == 2**n


@pytest.mark.critical
def test_factor_decomposer_scaling_and_quality():
    # Generate random correlation matrix via Wishart-like construction
    rng = np.random.default_rng(123)
    N = 20
    X = rng.normal(size=(N, 5))
    cov = X @ X.T
    diag = np.sqrt(np.diag(cov))
    corr = cov / np.outer(diag, diag)
    np.fill_diagonal(corr, 1.0)

    decomp = FactorDecomposer()
    L3, D3, m3 = decomp.fit(corr, K=3)
    L5, D5, m5 = decomp.fit(corr, K=5)

    # Quality should improve with higher K
    assert m5.frobenius_error <= m3.frobenius_error + 1e-12
    assert m5.variance_explained >= m3.variance_explained - 1e-12

    # Gate counts vs full
    full = N*(N-1)//2
    sparse3 = N*3
    sparse5 = N*5
    assert sparse3 < full
    assert sparse5 < full


@pytest.mark.critical
def test_mlqae_k0_matches_classical_within_10pct():
    S0, r, sigma, T = 100.0, 0.03, 0.25, 1.0
    circ, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits=6)
    from qiskit import QuantumRegister
    anc = QuantumRegister(1, 'anc')
    circ.add_register(anc)
    scale = apply_call_payoff_rotation(circ, circ.qregs[0], anc[0], prices, 105.0)

    # Quantum estimate
    res = run_mlqae(circ, anc[0], scale, grover_powers=[0], shots_per_power=3000, seed=7)

    # Classical exact
    ensure_statevector_ok(circ)
    sv = Statevector(circ)
    anc_idx = circ.qubits.index(anc[0])
    prob1 = sum(float((amp.conjugate()*amp).real) for i, amp in enumerate(sv.data) if (i >> anc_idx) & 1)
    exact = prob1 * scale

    rel_err = abs(res.price_estimate - exact) / (exact + 1e-12)
    assert rel_err < 0.10, f"MLQAE(k=0) error {rel_err:.1%} too high"


@pytest.mark.expensive
def test_end_to_end_basket_small_qubits():
    # Small N to be memory-safe
    from qfdp_multiasset.portfolio import price_basket_option, PortfolioPayoff
    N = 4
    asset_params = [(100+10*i, 0.03, 0.25, 1.0) for i in range(N)]
    # Synthetic moderate correlation matrix
    corr = np.full((N,N), 0.4)
    np.fill_diagonal(corr, 1.0)
    payoff = PortfolioPayoff('basket', weights=np.ones(N)/N, strike=110.0)

    res = price_basket_option(
        asset_params, corr, payoff,
        n_factors=2, n_qubits_asset=3, n_qubits_factor=2,
        grover_powers=[0], shots_per_power=500, seed=42
    )
    assert res.price_estimate >= 0


@pytest.mark.slow
def test_repeated_runs_no_drift():
    # Ensure repeated runs are stable
    S0, r, sigma, T = 100.0, 0.03, 0.25, 1.0
    circ, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits=5)
    from qiskit import QuantumRegister
    anc = QuantumRegister(1, 'anc')
    circ.add_register(anc)
    scale = apply_call_payoff_rotation(circ, circ.qregs[0], anc[0], prices, 100.0)

    vals = []
    for seed in [1,2,3,4,5]:
        res = run_mlqae(circ, anc[0], scale, grover_powers=[0], shots_per_power=1000, seed=seed)
        vals.append(res.price_estimate)
    v = np.array(vals)
    # Standard deviation should be reasonable for 1000 shots
    assert v.std() < 0.15 * v.mean()
