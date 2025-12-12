"""
Factor-Based Quantum Fourier Derivative Pricing (FB-IQFT)
===========================================================

**NOVEL ALGORITHM**: Combines factor-model dimensionality reduction with
quantum Fourier pricing, achieving shallow IQFT depth.

Key Innovation:
- Perform IQFT in K-dimensional factor space (not N-dimensional asset space)
- Depth reduction: O(log²K) vs O(log²N) where K << N
- NISQ-feasible: 2-3 qubits for typical portfolios

Mathematical Framework:
-----------------------
1. Factor decomposition (classical): Σ = LL^T + D
2. Factor-space characteristic function: φ_factor(u) = E[exp(i·u·w^T·L·f)]
3. Carr-Madan in factor space: C(K) = (1/π) ∫ ψ_factor(u) e^(-iuk) du
4. **Shallow IQFT**: log2(K) qubits only
5. Hybrid mapping: Factor payoff → Asset payoff via L

Author: QFDP Unified Research Team
Date: November 30, 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

# Import from unified package
import sys
sys.path.insert(0, '/Volumes/Hippocampus/QFDP')
from unified_qfdp.factor_char_func import (
    compute_factor_loading_transform,
    factor_space_char_func_gaussian,
    estimate_factor_volatilities,
    build_factor_frequency_grid
)
from unified_qfdp.fb_iqft_circuit import (
    build_fb_iqft_circuit,
    compute_factor_qubits,
    FBIQFTCircuitResult
)
from qfdp_multiasset.sparse_copula import FactorDecomposer


@dataclass
class FBIQFTPricingResult:
    """Result from FB-IQFT pricing."""
    price: float
    circuit_depth: int
    depth_reduction: float
    K: int
    n_factor_qubits: int
    factor_weights: np.ndarray
    variance_explained: float
    classical_price_baseline: Optional[float] = None


def factor_based_qfdp(
    portfolio_weights: np.ndarray,
    asset_volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    spot_value: float,
    strike: float,
    risk_free_rate: float,
    maturity: float,
    K: int = 4,
    use_approximate_iqft: bool = False,
    validate_vs_classical: bool = True,
    run_on_hardware: bool = False,
    backend_name: Optional[str] = None
) -> FBIQFTPricingResult:
    """
    Factor-Based Quantum Fourier Derivative Pricing (FB-IQFT).
    
    CORE ALGORITHM: Combines factor decomposition + shallow IQFT for
    NISQ-feasible Fourier-based quantum option pricing.
    
    Parameters
    ----------
    portfolio_weights : np.ndarray (N,)
        Portfolio weights (must sum to 1)
    asset_volatilities : np.ndarray (N,)
        Individual asset volatilities (annualized)
    correlation_matrix : np.ndarray (N, N)
        Asset correlation matrix
    spot_value : float
        Current portfolio value
    strike : float
        Option strike price
    risk_free_rate : float
        Risk-free rate (annualized)
    maturity : float
        Time to maturity (years)
    K : int
        Number of factors (default: 4)
        Typical: K=2-8 for portfolios
    use_approximate_iqft : bool
        Use approximate IQFT to further reduce depth
    validate_vs_classical : bool
        Compare to classical Monte Carlo baseline
    run_on_hardware : bool
        Execute on real IBM Quantum hardware
    backend_name : str, optional
        IBM backend name (e.g., 'ibm_fez')
        
    Returns
    -------
    FBIQFTPricingResult
        Pricing result with circuit depth and reduction metrics
        
    Algorithm Steps:
    ----------------
    1. Factor decomposition: Σ → L (N×K)
    2. Transform to factor space: β = L^T · w
    3. Factor-space characteristic function: φ_factor(u)
    4. Build shallow IQFT circuit (log2(K) qubits)
    5. Execute (quantum or simulator)
    6. Extract option price
    
    Examples
    --------
    >>> weights = np.array([0.3, 0.4, 0.3])
    >>> vols = np.array([0.25, 0.30, 0.20])
    >>> corr = np.eye(3)
    >>> result = factor_based_qfdp(
    ...     weights, vols, corr,
    ...     spot_value=100, strike=105,
    ...     risk_free_rate=0.05, maturity=1.0,
    ...     K=2
    ... )
    >>> print(f"Price: ${result.price:.2f}")
    >>> print(f"Depth reduction: {result.depth_reduction:.1f}×")
    """
    N = len(portfolio_weights)
    
    # Validate inputs
    assert abs(portfolio_weights.sum() - 1.0) < 1e-6, "Weights must sum to 1"
    assert correlation_matrix.shape == (N, N), "Correlation matrix must be N×N"
    assert 1 <= K <= N, f"K must be between 1 and {N}"
    
    print("="*70)
    print("FACTOR-BASED QUANTUM FOURIER DERIVATIVE PRICING (FB-IQFT)")
    print("="*70)
    print()
    print(f"Portfolio: N={N} assets, K={K} factors")
    print(f"Option: Strike={strike}, Spot={spot_value}, T={maturity}Y")
    print()
    
    # STEP 1: Factor Decomposition (Classical)
    print("Step 1: Factor Decomposition...")
    decomposer = FactorDecomposer()
    factor_loading, idiosyncratic, metrics = decomposer.fit(
        correlation_matrix, K=K
    )
    variance_explained = metrics.variance_explained
    print(f"  Variance explained: {variance_explained*100:.1f}%")
    print(f"  Frobenius error: {metrics.frobenius_error:.4f}")
    print()
    
    # STEP 2: Transform to Factor Space
    print("Step 2: Transform to Factor Space...")
    factor_weights = compute_factor_loading_transform(
        portfolio_weights, factor_loading
    )
    print(f"  Effective factor exposure β = L^T·w")
    print(f"  β = {factor_weights}")
    print()
    
    # Estimate factor volatilities
    factor_vols = estimate_factor_volatilities(
        asset_volatilities, factor_loading
    )
    print(f"  Factor volatilities: {factor_vols}")
    print()
    
    # STEP 3: Factor-Space Characteristic Function
    print("Step 3: Compute Factor-Space Characteristic Function...")
    u, dx = build_factor_frequency_grid(K, du=0.25, N_points=64)
    
    char_func_values = factor_space_char_func_gaussian(
        u, factor_weights, factor_vols,
        risk_free_rate, maturity, spot_value,
        damping_alpha=1.5
    )
    print(f"  Frequency points: {len(u)}")
    print(f"  Characteristic function evaluated in K={K} dimensions")
    print()
    
    # STEP 4: Build Shallow IQFT Circuit
    print("Step 4: Build FB-IQFT Circuit...")
    circuit_result = build_fb_iqft_circuit(
        K, char_func_values, strike, spot_value,
        use_approximate_iqft=use_approximate_iqft
    )
    
    print(f"  Circuit qubits: {circuit_result.n_factor_qubits}")
    print(f"  Circuit depth: {circuit_result.circuit_depth}")
    print(f"  Circuit gates: {circuit_result.circuit_gates}")
    print(f"  Depth reduction vs traditional: {circuit_result.depth_reduction_vs_traditional:.1f}×")
    print()
    
    # STEP 5: Execute Circuit
    print("Step 5: Execute Quantum Circuit...")
    
    if run_on_hardware:
        # Run on real IBM Quantum hardware
        from qfdp_multiasset.hardware import IBMQuantumRunner
        
        runner = IBMQuantumRunner(
            backend_name=backend_name,
            use_simulator=False
        )
        
        hw_result = runner.run(circuit_result.circuit, shots=2048)
        
        if hw_result.success:
            print(f"  ✅ Hardware execution successful!")
            print(f"  Backend: {hw_result.backend_name}")
            print(f"  Execution time: {hw_result.execution_time:.2f}s")
            
            # Extract price from hardware results
            # Count ancilla=1 outcomes
            ancilla_ones = sum(
                count for bitstring, count in hw_result.counts.items()
                if bitstring[-1] == '1'
            )
            amplitude = ancilla_ones / hw_result.shots
            
        else:
            print(f"  ❌ Hardware execution failed: {hw_result.error_message}")
            print(f"  Falling back to simulator")
            run_on_hardware = False
    
    if not run_on_hardware:
        # Run on simulator
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import Sampler
        
        simulator = AerSimulator()
        sampler = Sampler()
        
        job = sampler.run(circuit_result.circuit, shots=2048)
        result = job.result()
        
        # Extract amplitude from measurements
        counts = result.quasi_dists[0].binary_probabilities()
        amplitude = sum(
            prob for bitstring, prob in counts.items()
            if bitstring[-1] == '1'
        )
        
        print(f"  Simulator: Amplitude = {amplitude:.6f}")
        print()
    
    # STEP 6: Compute Option Price
    print("Step 6: Compute Option Price...")
    
    # Scale by max payoff
    max_payoff = max(1.5 * spot_value - strike, 0)
    price = amplitude * max_payoff * np.exp(-risk_free_rate * maturity)
    
    print(f"  Option price (FB-IQFT): ${price:.4f}")
    
    # Classical baseline for validation
    classical_price = None
    if validate_vs_classical:
        print()
        print("Validation: Classical Monte Carlo Baseline...")
        classical_price = _classical_mc_baseline(
            portfolio_weights, asset_volatilities, correlation_matrix,
            spot_value, strike, risk_free_rate, maturity
        )
        print(f"  Classical MC price: ${classical_price:.4f}")
        
        error = abs(price - classical_price) / classical_price * 100
        print(f"  Error: {error:.2f}%")
    
    print()
    print("="*70)
    
    return FBIQFTPricingResult(
        price=price,
        circuit_depth=circuit_result.circuit_depth,
        depth_reduction=circuit_result.depth_reduction_vs_traditional,
        K=K,
        n_factor_qubits=circuit_result.n_factor_qubits,
        factor_weights=factor_weights,
        variance_explained=variance_explained,
        classical_price_baseline=classical_price
    )


def _classical_mc_baseline(
    weights: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    spot: float,
    strike: float,
    r: float,
    T: float,
    num_paths: int = 100000
) -> float:
    """Classical Monte Carlo baseline for validation."""
    N = len(weights)
    
    # Cholesky decomposition
    L = np.linalg.cholesky(correlation)
    
    # Sample correlated returns
    rng = np.random.default_rng(42)
    epsilon = rng.standard_normal((num_paths, N))
    Z = epsilon @ L.T
    
    # Asset returns
    returns = Z * volatilities[None, :] * np.sqrt(T)
    
    # Portfolio returns
    portfolio_returns = returns @ weights
    
    # Portfolio values at maturity
    drift = (r - 0.5 * np.sum((weights * volatilities) ** 2)) * T
    portfolio_values = spot * np.exp(drift + portfolio_returns)
    
    # Payoffs
    payoffs = np.maximum(portfolio_values - strike, 0)
    
    # Discounted price
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price


def compare_fb_iqft_vs_traditional(
    N_values: list = [10, 20, 50, 100]
) -> None:
    """
    Compare FB-IQFT depth vs traditional QFDP for different portfolio sizes.
    
    Demonstrates the depth advantage of factor-space IQFT.
    """
    print("="*70)
    print("FB-IQFT vs Traditional QFDP: Depth Comparison")
    print("="*70)
    print()
    
    K = 4  # Typical factor count
    
    print(f"| N Assets | Traditional | FB-IQFT (K={K}) | Reduction |")
    print("|----------|-------------|-----------------|-----------|")
    
    for N in N_values:
        # Traditional QFDP
        n_trad = int(np.ceil(np.log2(N)))
        trad_depth = int(0.5 * n_trad ** 2)
        
        # FB-IQFT
        n_factor = compute_factor_qubits(K)
        fb_depth = int(0.5 * n_factor ** 2)
        
        reduction = trad_depth / fb_depth if fb_depth > 0 else 1
        
        print(f"| {N:8d} | {trad_depth:11d} | {fb_depth:15d} | {reduction:8.1f}× |")
    
    print()
    print("Key insight: FB-IQFT depth is constant in N (depends only on K)")
    print(f"For any portfolio size with K={K} factors: depth ≈ {fb_depth} gates")
    print()
