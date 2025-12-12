"""
Multi-Asset Portfolio Pricing
==============================

Basket options, worst-of, best-of, rainbow derivatives using MLQAE + sparse copula.

Payoff Types:
-------------
1. Basket Call: max(Σ w_i S_i - K, 0)
2. Worst-of Call: max(min(S_1, ..., S_N) - K, 0)
3. Best-of Call: max(max(S_1, ..., S_N) - K, 0)
4. Rainbow (2-color): max(max(S_1, S_2) - α·min(S_1, S_2) - K, 0)

Integration:
- Phase 3: Sparse copula for correlation
- Phase 5/6: Multi-asset payoff oracles
- Phase 7: MLQAE amplitude estimation

Author: QFDP Research
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from qiskit import QuantumCircuit, QuantumRegister

from ..sparse_copula import encode_sparse_copula_with_decomposition
from ..oracles import apply_piecewise_constant_payoff
from ..mlqae import run_mlqae, MLQAEResult


@dataclass
class PortfolioPayoff:
    """Multi-asset payoff specification."""
    payoff_type: str  # 'basket', 'worst-of', 'best-of', 'rainbow'
    weights: Optional[np.ndarray] = None  # For basket (default equal-weighted)
    strike: float = 100.0
    alpha: float = 0.0  # Rainbow parameter


def compute_basket_payoff(
    asset_prices: List[np.ndarray],
    weights: np.ndarray,
    strike: float
) -> np.ndarray:
    """
    Basket call payoff: max(Σ w_i S_i - K, 0).
    
    Args:
        asset_prices: List of N price grids (each 2^n points)
        weights: Portfolio weights (sum to 1)
        strike: Strike price K
        
    Returns:
        Payoff array on joint state space (2^(N×n) values)
    """
    # Build joint state space via tensor product
    meshgrids = np.meshgrid(*asset_prices, indexing='ij')
    
    # Weighted sum: portfolio value
    portfolio_value = sum(w * S for w, S in zip(weights, meshgrids))
    
    # Call payoff
    payoff = np.maximum(portfolio_value - strike, 0.0)
    
    return payoff.flatten()


def compute_worst_of_payoff(
    asset_prices: List[np.ndarray],
    strike: float
) -> np.ndarray:
    """
    Worst-of call: max(min(S_1, ..., S_N) - K, 0).
    
    Returns:
        Payoff array on joint state space
    """
    meshgrids = np.meshgrid(*asset_prices, indexing='ij')
    
    # Element-wise minimum across assets
    worst_price = np.minimum.reduce(meshgrids)
    
    payoff = np.maximum(worst_price - strike, 0.0)
    return payoff.flatten()


def compute_best_of_payoff(
    asset_prices: List[np.ndarray],
    strike: float
) -> np.ndarray:
    """
    Best-of call: max(max(S_1, ..., S_N) - K, 0).
    
    Returns:
        Payoff array on joint state space
    """
    meshgrids = np.meshgrid(*asset_prices, indexing='ij')
    
    # Element-wise maximum across assets
    best_price = np.maximum.reduce(meshgrids)
    
    payoff = np.maximum(best_price - strike, 0.0)
    return payoff.flatten()


def compute_rainbow_payoff(
    asset1_prices: np.ndarray,
    asset2_prices: np.ndarray,
    alpha: float,
    strike: float
) -> np.ndarray:
    """
    Rainbow 2-color call: max(max(S_1, S_2) - α·min(S_1, S_2) - K, 0).
    
    α = 0: best-of
    α = 1: spread option
    
    Returns:
        Payoff array on joint state space (2D flattened)
    """
    S1_grid, S2_grid = np.meshgrid(asset1_prices, asset2_prices, indexing='ij')
    
    max_S = np.maximum(S1_grid, S2_grid)
    min_S = np.minimum(S1_grid, S2_grid)
    
    payoff = np.maximum(max_S - alpha * min_S - strike, 0.0)
    return payoff.flatten()


def price_basket_option(
    asset_params: List[Tuple[float, float, float, float]],
    correlation_matrix: np.ndarray,
    payoff_spec: PortfolioPayoff,
    n_factors: int = 3,
    n_qubits_asset: int = 4,
    n_qubits_factor: int = 2,
    n_segments: int = 16,
    grover_powers: Optional[List[int]] = None,
    shots_per_power: int = 1000,
    seed: Optional[int] = None
) -> MLQAEResult:
    """
    Price multi-asset option using sparse copula + MLQAE.
    
    Args:
        asset_params: List of (S0, r, sigma, T) for each asset
        correlation_matrix: N×N correlation matrix
        payoff_spec: Portfolio payoff specification
        n_factors: Number of copula factors K
        n_qubits_asset: Qubits per asset
        n_qubits_factor: Qubits per factor
        n_segments: Piecewise segments for payoff oracle
        grover_powers: MLQAE Grover iterations
        shots_per_power: Measurements per iteration
        seed: RNG seed
        
    Returns:
        MLQAEResult with option price estimate
        
    Example:
        >>> asset_params = [(100, 0.03, 0.25, 1.0), (150, 0.03, 0.20, 1.0)]
        >>> corr = np.array([[1.0, 0.7], [0.7, 1.0]])
        >>> payoff = PortfolioPayoff('basket', weights=np.array([0.5, 0.5]), strike=125)
        >>> result = price_basket_option(asset_params, corr, payoff)
        >>> print(f"Basket option price: ${result.price_estimate:.2f}")
    """
    N = len(asset_params)
    
    # Phase 3: Encode correlated asset state
    circ, metrics = encode_sparse_copula_with_decomposition(
        asset_params, correlation_matrix, n_factors=n_factors,
        n_qubits_asset=n_qubits_asset, n_qubits_factor=n_qubits_factor
    )
    
    # Extract asset registers and price grids
    asset_regs = [qreg for qreg in circ.qregs if qreg.name.startswith('asset_')]
    
    # Reconstruct price grids for each asset (match state prep discretization)
    asset_price_grids = []
    for S0, r, sigma, T in asset_params:
        sigma_clamped = max(sigma, 1e-6)
        mu = (r - 0.5*sigma_clamped**2) * T
        sigma_r = sigma_clamped * np.sqrt(T)
        
        log_S_min = np.log(S0) + mu - 3 * sigma_r
        log_S_max = np.log(S0) + mu + 3 * sigma_r
        log_prices = np.linspace(log_S_min, log_S_max, 2**n_qubits_asset)
        prices = np.exp(log_prices)
        asset_price_grids.append(prices)
    
    # Compute payoff on joint state space
    if payoff_spec.payoff_type == 'basket':
        weights = payoff_spec.weights if payoff_spec.weights is not None else np.ones(N) / N
        payoff = compute_basket_payoff(asset_price_grids, weights, payoff_spec.strike)
    elif payoff_spec.payoff_type == 'worst-of':
        payoff = compute_worst_of_payoff(asset_price_grids, payoff_spec.strike)
    elif payoff_spec.payoff_type == 'best-of':
        payoff = compute_best_of_payoff(asset_price_grids, payoff_spec.strike)
    elif payoff_spec.payoff_type == 'rainbow':
        if N != 2:
            raise ValueError("Rainbow payoff requires exactly 2 assets")
        payoff = compute_rainbow_payoff(
            asset_price_grids[0], asset_price_grids[1],
            payoff_spec.alpha, payoff_spec.strike
        )
    else:
        raise ValueError(f"Unknown payoff type: {payoff_spec.payoff_type}")
    
    # Phase 6: Encode payoff with piecewise oracle
    # ⚠️ WARNING: MARGINAL APPROXIMATION ⚠️
    # We encode the MARGINAL payoff on first asset only, averaging over others.
    # This LOSES correlation structure and is NOT true basket pricing.
    # True implementation requires N-register controlled payoff oracle (complex).
    anc = QuantumRegister(1, 'ancilla')
    circ.add_register(anc)
    
    # Marginal approximation: encode E[payoff | S_1]
    marginal_payoff = payoff.reshape([2**n_qubits_asset] * N).mean(axis=tuple(range(1, N)))
    
    scale = apply_piecewise_constant_payoff(
        circ, asset_regs[0], anc[0],
        asset_price_grids[0], marginal_payoff,
        n_segments=n_segments
    )
    
    # Phase 7: MLQAE pricing
    result = run_mlqae(
        circ, anc[0], scale,
        grover_powers=grover_powers,
        shots_per_power=shots_per_power,
        seed=seed
    )
    
    return result


def price_basket_option_exact(
    asset_params: List[Tuple[float, float, float, float]],
    correlation_matrix: np.ndarray,
    payoff_spec: PortfolioPayoff,
    n_factors: int = 3,
    n_qubits_asset: int = 4,
    n_qubits_factor: int = 2
) -> float:
    """
    Classical exact pricing (statevector expectation) for benchmarking.
    
    Returns:
        Exact option price
    """
    from qiskit.quantum_info import Statevector
    
    # Build correlated state
    circ, _ = encode_sparse_copula_with_decomposition(
        asset_params, correlation_matrix, n_factors=n_factors,
        n_qubits_asset=n_qubits_asset, n_qubits_factor=n_qubits_factor
    )
    
    # Get statevector
    sv = Statevector(circ)
    probs = np.abs(sv.data) ** 2
    
    # Reconstruct price grids
    asset_price_grids = []
    for S0, r, sigma, T in asset_params:
        sigma_clamped = max(sigma, 1e-6)
        mu = (r - 0.5*sigma_clamped**2) * T
        sigma_r = sigma_clamped * np.sqrt(T)
        log_S_min = np.log(S0) + mu - 3 * sigma_r
        log_S_max = np.log(S0) + mu + 3 * sigma_r
        log_prices = np.linspace(log_S_min, log_S_max, 2**n_qubits_asset)
        asset_price_grids.append(np.exp(log_prices))
    
    # Compute payoff
    N = len(asset_params)
    if payoff_spec.payoff_type == 'basket':
        weights = payoff_spec.weights if payoff_spec.weights is not None else np.ones(N) / N
        payoff = compute_basket_payoff(asset_price_grids, weights, payoff_spec.strike)
    elif payoff_spec.payoff_type == 'worst-of':
        payoff = compute_worst_of_payoff(asset_price_grids, payoff_spec.strike)
    elif payoff_spec.payoff_type == 'best-of':
        payoff = compute_best_of_payoff(asset_price_grids, payoff_spec.strike)
    elif payoff_spec.payoff_type == 'rainbow':
        payoff = compute_rainbow_payoff(
            asset_price_grids[0], asset_price_grids[1],
            payoff_spec.alpha, payoff_spec.strike
        )
    else:
        raise ValueError(f"Unknown payoff type: {payoff_spec.payoff_type}")
    
    # Marginalize over factors (last n_factors × n_qubits_factor qubits)
    n_asset_qubits = N * n_qubits_asset
    n_factor_qubits = n_factors * n_qubits_factor
    
    # Reshape probabilities: (assets, factors)
    probs_reshaped = probs.reshape(2**n_asset_qubits, 2**n_factor_qubits)
    marginal_probs = probs_reshaped.sum(axis=1)  # Sum over factors
    
    # Expected payoff
    expected_payoff = np.dot(marginal_probs, payoff)
    
    return expected_payoff
