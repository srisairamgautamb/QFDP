"""
BRUTAL STRESS TEST
==================

Test the system limits:
1. Maximum qubit circuits (find memory ceiling)
2. Parallel test execution (CPU saturation)
3. Large-scale Monte Carlo (statistical validation)
4. Combined multi-asset pricing (full pipeline stress)

Run: python3 -m pytest tests/stress/test_brutal_stress.py -v -n auto
"""

import numpy as np
import pytest
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from qfdp_multiasset.state_prep import prepare_lognormal_asset, prepare_gaussian_factor
from qfdp_multiasset.sparse_copula import (
    generate_synthetic_correlation_matrix,
    encode_sparse_copula_with_decomposition,
)
from qfdp_multiasset.oracles import apply_call_payoff_rotation, apply_piecewise_constant_payoff, call_payoff
from qfdp_multiasset.mlqae import run_mlqae


class TestMemoryLimits:
    """Find the maximum qubit count before OOM."""
    
    @pytest.mark.parametrize("n_qubits", [10, 11, 12, 13, 14])
    def test_max_single_asset_qubits(self, n_qubits):
        """Single asset state prep up to 2^14 = 16K amplitudes."""
        start = time.time()
        circ, prices = prepare_lognormal_asset(100.0, 0.03, 0.25, 1.0, n_qubits=n_qubits)
        sv = Statevector(circ)
        elapsed = time.time() - start
        
        assert circ.num_qubits == n_qubits
        assert len(sv.data) == 2**n_qubits
        print(f"  → {n_qubits} qubits: {len(sv.data):,} amps, {elapsed:.2f}s")
    
    def test_max_multi_asset_circuit(self):
        """3 assets × 8 qubits + 2 factors × 4 qubits = 28 qubits (256M amps = 2GB)."""
        # This will likely fail with memory error - that's the point
        asset_params = [
            (100.0, 0.03, 0.20, 1.0),
            (150.0, 0.03, 0.25, 1.0),
            (200.0, 0.03, 0.30, 1.0),
        ]
        corr = generate_synthetic_correlation_matrix(N=3, K=2, seed=999)
        
        start = time.time()
        try:
            circ, metrics = encode_sparse_copula_with_decomposition(
                asset_params, corr, n_factors=2,
                n_qubits_asset=8, n_qubits_factor=4
            )
            # If we get here, try statevector (likely to fail)
            sv = Statevector(circ)
            elapsed = time.time() - start
            print(f"  → 28 qubits PASSED: {len(sv.data):,} amps, {elapsed:.2f}s")
            assert True
        except Exception as e:
            elapsed = time.time() - start
            print(f"  → 28 qubits FAILED after {elapsed:.2f}s: {type(e).__name__}")
            # This is expected - record the failure
            assert "Memory" in str(e) or "allocate" in str(e).lower()


class TestParallelExecution:
    """Saturate all CPU cores with parallel pricing."""
    
    def _price_option(self, seed):
        """Single option pricing task."""
        S0 = 100.0 + seed % 50
        K = S0 * 1.05
        circ, prices = prepare_lognormal_asset(S0, 0.03, 0.25, 1.0, n_qubits=5)
        anc = QuantumRegister(1, 'anc')
        circ.add_register(anc)
        scale = apply_call_payoff_rotation(circ, circ.qregs[0], anc[0], prices, K)
        result = run_mlqae(circ, anc[0], scale, grover_powers=[0], shots_per_power=500, seed=seed)
        return result.price_estimate
    
    def test_parallel_pricing_10_options(self):
        """Price 10 options in parallel."""
        start = time.time()
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._price_option, i) for i in range(10)]
            prices = [f.result() for f in as_completed(futures)]
        elapsed = time.time() - start
        
        assert len(prices) == 10
        assert all(p > 0 for p in prices)
        print(f"  → Priced 10 options in {elapsed:.2f}s ({10/elapsed:.1f} options/s)")
    
    def test_parallel_pricing_50_options(self):
        """Price 50 options in parallel (full CPU saturation)."""
        start = time.time()
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self._price_option, i) for i in range(50)]
            prices = [f.result() for f in as_completed(futures)]
        elapsed = time.time() - start
        
        assert len(prices) == 50
        print(f"  → Priced 50 options in {elapsed:.2f}s ({50/elapsed:.1f} options/s)")


class TestMonteCarloValidation:
    """Large-scale statistical validation."""
    
    def test_1000_mlqae_runs_convergence(self):
        """Run MLQAE 1000 times, verify mean converges."""
        S0, r, sigma, T = 100.0, 0.03, 0.25, 1.0
        K = 105.0
        circ, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits=4)
        anc = QuantumRegister(1, 'anc')
        circ.add_register(anc)
        scale = apply_call_payoff_rotation(circ, circ.qregs[0], anc[0], prices, K)
        
        # Classical ground truth
        sv = Statevector(circ)
        ancilla_idx = circ.qubits.index(anc[0])
        prob_1 = sum(
            float((amp.conjugate() * amp).real)
            for i, amp in enumerate(sv.data)
            if (i >> ancilla_idx) & 1
        )
        true_price = prob_1 * scale
        
        # Run MLQAE 1000 times
        start = time.time()
        estimates = []
        for seed in range(1000):
            result = run_mlqae(circ, anc[0], scale, grover_powers=[0], shots_per_power=100, seed=seed)
            estimates.append(result.price_estimate)
        elapsed = time.time() - start
        
        # Statistical validation
        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates)
        
        # Central Limit Theorem: mean should be within 3σ/√N
        error = abs(mean_estimate - true_price)
        expected_std_of_mean = std_estimate / np.sqrt(1000)
        
        print(f"  → 1000 runs in {elapsed:.2f}s")
        print(f"  → Mean: {mean_estimate:.3f}, True: {true_price:.3f}, Error: {error:.3f}")
        print(f"  → Std: {std_estimate:.3f}, Std of mean: {expected_std_of_mean:.4f}")
        
        assert error < 3 * expected_std_of_mean, f"Mean diverged: {error:.3f} > {3*expected_std_of_mean:.4f}"


class TestFullPipelineStress:
    """Combined multi-asset pricing with all phases."""
    
    def test_2_asset_100_price_paths(self):
        """Price 2-asset basket with 100 different strikes."""
        asset_params = [
            (100.0, 0.03, 0.25, 1.0),
            (150.0, 0.03, 0.20, 1.0),
        ]
        corr = generate_synthetic_correlation_matrix(N=2, K=1, seed=42)
        
        start = time.time()
        results = []
        for i in range(100):
            K = 100.0 + i  # Strike from 100 to 199
            circ, metrics = encode_sparse_copula_with_decomposition(
                asset_params, corr, n_factors=1,
                n_qubits_asset=4, n_qubits_factor=2
            )
            asset_regs = [qreg for qreg in circ.qregs if qreg.name.startswith('asset_')]
            asset0_prices = np.linspace(50, 150, 16)
            payoff = call_payoff(asset0_prices, K)
            
            anc = QuantumRegister(1, 'anc')
            circ.add_register(anc)
            scale = apply_piecewise_constant_payoff(circ, asset_regs[0], anc[0], asset0_prices, payoff, n_segments=8)
            
            result = run_mlqae(circ, anc[0], scale, grover_powers=[0], shots_per_power=200, seed=i)
            results.append(result.price_estimate)
        
        elapsed = time.time() - start
        print(f"  → Priced 100 basket options in {elapsed:.2f}s ({100/elapsed:.1f} options/s)")
        assert len(results) == 100
        assert all(p >= 0 for p in results)
    
    def test_phase_0_through_7_integration(self):
        """Full pipeline: Phase 0 (structure) → Phase 7 (MLQAE pricing)."""
        # Phase 1: Factor decomposition
        corr = generate_synthetic_correlation_matrix(N=2, K=1, seed=123)
        
        # Phase 2: State prep
        asset_params = [(100.0, 0.03, 0.25, 1.0), (150.0, 0.03, 0.20, 1.0)]
        
        # Phase 3: Sparse copula encoding
        start = time.time()
        circ, metrics = encode_sparse_copula_with_decomposition(
            asset_params, corr, n_factors=1,
            n_qubits_asset=5, n_qubits_factor=2
        )
        t1 = time.time()
        
        # Phase 5: Payoff oracle
        asset_regs = [qreg for qreg in circ.qregs if qreg.name.startswith('asset_')]
        asset0_prices = np.linspace(50, 150, 32)
        anc = QuantumRegister(1, 'anc')
        circ.add_register(anc)
        scale = apply_call_payoff_rotation(circ, asset_regs[0], anc[0], asset0_prices, K=105)
        t2 = time.time()
        
        # Phase 7: MLQAE pricing
        result = run_mlqae(circ, anc[0], scale, grover_powers=[0], shots_per_power=1000, seed=999)
        t3 = time.time()
        
        print(f"  → Phase 3 (encoding): {t1-start:.3f}s")
        print(f"  → Phase 5 (oracle): {t2-t1:.3f}s")
        print(f"  → Phase 7 (MLQAE): {t3-t2:.3f}s")
        print(f"  → TOTAL: {t3-start:.3f}s")
        print(f"  → Price: ${result.price_estimate:.2f}")
        
        assert result.price_estimate > 0
        assert metrics.total_qubits == 2*5 + 1*2  # assets + factors (ancilla added after)
        assert circ.num_qubits == 2*5 + 1*2 + 1  # Total including ancilla


class TestRobustness:
    """Edge cases and error handling."""
    
    def test_extreme_volatility(self):
        """σ = 200% (extreme volatility)."""
        circ, prices = prepare_lognormal_asset(100.0, 0.03, 2.0, 1.0, n_qubits=6)
        assert circ.num_qubits == 6
        assert prices.min() > 0  # No negative prices
    
    def test_zero_volatility_clamped(self):
        """σ = 0 (deterministic) should be clamped to ε."""
        circ, prices = prepare_lognormal_asset(100.0, 0.03, 0.0, 1.0, n_qubits=5)
        assert not np.any(np.isnan(prices))
        sv = Statevector(circ)
        assert not np.any(np.isnan(sv.data))
    
    def test_deep_otm_option(self):
        """Strike = 10× spot (deep out-of-money)."""
        S0 = 100.0
        K = 1000.0  # 10× spot
        circ, prices = prepare_lognormal_asset(S0, 0.03, 0.25, 1.0, n_qubits=5)
        anc = QuantumRegister(1, 'anc')
        circ.add_register(anc)
        scale = apply_call_payoff_rotation(circ, circ.qregs[0], anc[0], prices, K)
        result = run_mlqae(circ, anc[0], scale, grover_powers=[0], shots_per_power=500, seed=42)
        
        # Should price near zero
        assert result.price_estimate < 1.0
    
    def test_deep_itm_option(self):
        """Strike = 0.1× spot (deep in-the-money)."""
        S0 = 100.0
        K = 10.0  # 0.1× spot
        circ, prices = prepare_lognormal_asset(S0, 0.03, 0.25, 1.0, n_qubits=5)
        anc = QuantumRegister(1, 'anc')
        circ.add_register(anc)
        scale = apply_call_payoff_rotation(circ, circ.qregs[0], anc[0], prices, K)
        result = run_mlqae(circ, anc[0], scale, grover_powers=[0], shots_per_power=500, seed=42)
        
        # Should price near S0 - K
        assert result.price_estimate > 50.0  # At least half intrinsic value
