"""
Quantum-to-Classical Calibration and Price Reconstruction

This module implements Steps 11-12 of the FB-IQFT flowchart:
- Calibrate quantum probabilities to classical FFT baseline
- Reconstruct option prices from calibrated measurements

The calibration corrects for quantum noise and normalization differences
between the quantum and classical Fourier transforms.
"""

import numpy as np
from typing import Dict, Tuple


def calibrate_quantum_to_classical(
    quantum_probs: Dict[int, float],
    classical_prices: np.ndarray,
    k_grid: np.ndarray
) -> Tuple[float, float]:
    """
    Step 11: Calibrate quantum probabilities P(m) to classical prices.
    
    IMPROVED: Now supports local calibration windows for better per-strike accuracy.
    
    We fit a linear model:
    
        C_m^classical = A · P(m) + B
    
    using least-squares regression. This accounts for:
    1. Normalization differences (quantum state norm vs Fourier scale)
    2. Quantum noise (hardware errors, finite shots)
    3. Global offset (if any)
    
    The parameters (A, B) are then used to reconstruct prices from
    quantum measurements.
    
    Args:
        quantum_probs: {m: P(m)} from Step 10 (can be local window)
        classical_prices: C_m from classical FFT baseline (can be local window)
        k_grid: Log-strike grid (for validation, local or global)
    
    Returns:
        A: Scaling factor (typically ~1e3-1e4 order of magnitude)
        B: Offset term (typically small, ~0-1)
    
    Notes:
        - Uses ordinary least squares (OLS)
        - Robust to noise via averaging over M points
        - Alternative: use weighted LS to emphasize ATM strikes
    
    Example:
        >>> probs = {0: 0.01, 1: 0.15, 2: 0.30, ..., 15: 0.01}
        >>> C_classical = np.array([...])  # From FFT
        >>> A, B = calibrate_quantum_to_classical(probs, C_classical, k_grid)
        >>> print(f"Calibration: A={A:.2f}, B={B:.4f}")
    """
    M = len(classical_prices)
    
    # Build arrays for regression
    # IMPROVED: Handle both global and local window dicts
    m_indices = sorted(quantum_probs.keys())
    min_idx = min(m_indices) if m_indices else 0
    
    # Extract quantum probabilities aligned with classical prices
    # Assume classical_prices corresponds to indices [min_idx, min_idx+M)
    P_vec = np.array([quantum_probs.get(min_idx + i, 0.0) for i in range(M)])
    C_vec = classical_prices  # Already ordered
    
    # Design matrix: [P(m), 1] (intercept term)
    X = np.column_stack([P_vec, np.ones(M)])
    y = C_vec
    
    # Least squares: (XᵀX)⁻¹Xᵀy
    # Solve: [A, B]ᵀ = (XᵀX)⁻¹Xᵀy
    params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    A, B = params
    
    return A, B


def reconstruct_option_prices(
    quantum_probs: Dict[int, float],
    A: float,
    B: float,
    k_grid: np.ndarray,
    B_0: float
) -> np.ndarray:
    """
    Step 12: Reconstruct option prices from calibrated quantum measurements.
    
    Apply the calibration model:
    
        C_m^quantum = A · P(m) + B
    
    to convert quantum probabilities to option prices across the strike grid.
    
    Args:
        quantum_probs: {m: P(m)} from quantum measurement (Step 10)
        A: Scaling factor from calibration (Step 11)
        B: Offset term from calibration (Step 11)
        k_grid: Log-strike grid [k_0, ..., k_{M-1}]
        B_0: Initial basket value (for validation, optional use)
    
    Returns:
        option_prices: Array of call option prices C(K_m) for K_m = B_0·exp(k_m)
    
    Notes:
        - Prices should satisfy: 0 ≤ C(K) ≤ B_0·exp(rT) (forward price)
        - Monotonicity: C(K₁) ≥ C(K₂) if K₁ < K₂ (call option property)
        - If violations occur, it indicates hardware noise → increase shots
    
    Example:
        >>> probs = {0: 0.01, 1: 0.15, ..., 15: 0.01}
        >>> A, B = 5000.0, 0.12
        >>> prices = reconstruct_option_prices(probs, A, B, k_grid, B_0=100)
        >>> print(f"ATM price: ${prices[len(prices)//2]:.2f}")
    """
    M = len(k_grid)
    
    # Sort probabilities by index m
    m_indices = sorted(quantum_probs.keys())
    P_vec = np.array([quantum_probs[m] for m in m_indices])
    
    # Apply calibration model
    option_prices = A * P_vec + B
    
    # Optional: clip to valid range [0, forward_price]
    # (Uncomment if hardware noise causes violations)
    # option_prices = np.clip(option_prices, 0, B_0 * np.exp(r * T))
    
    return option_prices


def validate_prices(
    prices: np.ndarray,
    k_grid: np.ndarray,
    B_0: float,
    r: float,
    T: float,
    tol: float = 0.01
) -> Dict[str, bool]:
    """
    Validate reconstructed option prices for consistency.
    
    Checks:
    1. Non-negativity: C(K) ≥ 0 for all strikes
    2. Upper bound: C(K) ≤ B_0·exp(rT) (forward price)
    3. Monotonicity: C(K₁) ≥ C(K₂) if K₁ < K₂
    4. Put-call parity (optional, if puts are also computed)
    
    Args:
        prices: Option prices from Step 12
        k_grid: Log-strike grid
        B_0: Initial basket value
        r: Risk-free rate
        T: Time to maturity
        tol: Tolerance for numerical violations (fraction of price)
    
    Returns:
        checks: Dictionary of validation results
            - 'non_negative': bool
            - 'bounded': bool
            - 'monotonic': bool
    
    Example:
        >>> checks = validate_prices(prices, k_grid, B_0, r=0.05, T=1.0)
        >>> if not all(checks.values()):
        >>>     print(f"Validation warnings: {checks}")
    """
    forward_price = B_0 * np.exp(r * T)
    
    # Check 1: Non-negativity
    non_negative = np.all(prices >= -tol * np.abs(prices))
    
    # Check 2: Upper bound
    bounded = np.all(prices <= forward_price * (1 + tol))
    
    # Check 3: Monotonicity (decreasing in strike)
    strikes = B_0 * np.exp(k_grid)
    price_diffs = np.diff(prices)  # prices[i+1] - prices[i]
    # Should be non-positive (decreasing) or small positive (tolerance)
    monotonic = np.all(price_diffs <= tol * np.abs(prices[:-1]))
    
    return {
        'non_negative': bool(non_negative),
        'bounded': bool(bounded),
        'monotonic': bool(monotonic)
    }
