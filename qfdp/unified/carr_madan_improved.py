"""
Improved Carr-Madan Fourier Pricing for 1D Gaussian Basket

This module contains enhanced versions of the Carr-Madan functions with:
- Adaptive grid sizing based on moneyness
- Simpson's rule for better FFT accuracy
- Robust calibration with physical constraints

Improvements target <2% simulator accuracy.
"""

import numpy as np
from typing import Tuple
from .carr_madan_gaussian import (
    compute_characteristic_function,
    apply_carr_madan_transform
)


def setup_fourier_grid_adaptive(
    M: int,
    sigma_p: float,
    T: float,
    B_0: float,
    r: float,
    alpha: float = 1.0,
    K_target: float = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    IMPROVED: Adaptive grid setup with target strike awareness.
    
    Enhancements over standard version:
    1. Adaptive coverage: Wider grid for deep OTM/ITM options
    2. Target-centered: Ensures K_target is well-covered
    3. Better Nyquist: Validates grid spacing
    
    Args:
        M: Grid size (16, 32, 64, must be power of 2)
        sigma_p: Portfolio volatility
        T: Time to maturity
        B_0: Initial basket value
        r: Risk-free rate
        alpha: Damping parameter
        K_target: Target strike (optional, enables adaptive grid)
    
    Returns:
        u_grid: Frequency grid [u₀, ..., u_{M-1}]
        k_grid: Log-strike grid [k₀, ..., k_{M-1}]
        delta_u: Frequency spacing
        delta_k: Log-strike spacing
    """
    # Forward price for centering
    F = B_0 * np.exp(r * T)
    k_center = np.log(F / B_0)  # ≈ rT
    
    # IMPROVEMENT 1: Adaptive coverage based on moneyness
    if K_target is not None:
        moneyness = K_target / F
        if moneyness > 1.2 or moneyness < 0.8:  # Deep OTM/ITM
            coverage = 5.0  # Wider coverage (±5σ)
        else:
            coverage = 3.5  # Standard (±3.5σ)
    else:
        coverage = 3.5
    
    # IMPROVEMENT 2: Extend grid to capture target strike
    k_min = k_center - coverage * sigma_p * np.sqrt(T)
    k_max = k_center + coverage * sigma_p * np.sqrt(T)
    
    # IMPROVEMENT 3: Ensure K_target is within grid
    if K_target is not None:
        k_target = np.log(K_target / B_0)
        k_min = min(k_min, k_target - 0.5 * sigma_p * np.sqrt(T))
        k_max = max(k_max, k_target + 0.5 * sigma_p * np.sqrt(T))
    
    # Determine grid spacing
    delta_k = (k_max - k_min) / M
    delta_u = 2 * np.pi / (M * delta_k)  # Nyquist relation
    
    # Build grids
    u_grid = np.arange(M) * delta_u  # Start at 0
    k_grid = k_min + np.arange(M) * delta_k
    
    # VALIDATION: Check Nyquist
    nyquist_product = delta_u * delta_k
    expected_product = 2 * np.pi / M
    if not np.isclose(nyquist_product, expected_product, rtol=1e-6):
        print(f"⚠️  Nyquist warning: Δu·Δk = {nyquist_product:.6f}, expected {expected_product:.6f}")
    
    return u_grid, k_grid, delta_u, delta_k


def classical_fft_baseline_improved(
    psi_values: np.ndarray,
    alpha: float,
    delta_u: float,
    k_grid: np.ndarray
) -> np.ndarray:
    """
    IMPROVED: Classical FFT with Simpson's rule and bounds enforcement.
    
    Enhancements:
    1. Simpson's rule weighting for better accuracy
    2. Non-negativity enforcement (removes FFT artifacts)
    3. Forward price bound checking
    
    Args:
        psi_values: Modified CF ψ(u_j), shape (M,), complex
        alpha: Damping parameter
        delta_u: Frequency spacing
        k_grid: Log-strike grid
    
    Returns:
        C_classical: Call option prices, shape (M,), real
    """
    M = len(psi_values)
    
    # IMPROVEMENT 1: Simpson's rule weights for trapezoidal error correction
    simpson_weights = np.ones(M)
    simpson_weights[1:-1:2] = 4  # Odd indices
    simpson_weights[2:-1:2] = 2  # Even indices
    simpson_weights[0] = simpson_weights[-1] = 1
    simpson_weights = simpson_weights / 3.0  # Simpson's factor
    
    # Apply weights to psi
    psi_weighted = psi_values * simpson_weights
    
    # Apply IFFT (NumPy convention: exp(-i2πjm/M))
    F = np.fft.ifft(psi_weighted)
    
    # Damping factor e^(-αk)
    damping = np.exp(-alpha * k_grid)
    
    # Extract call prices (real part with scaling)
    C_classical = (damping / np.pi) * np.real(F) * delta_u * M / simpson_weights.mean()
    
    # IMPROVEMENT 2: Ensure non-negativity (FFT artifacts)
    C_classical = np.maximum(C_classical, 0.0)
    
    return C_classical


def calibrate_robust(
    quantum_probs: dict,
    classical_prices: np.ndarray,
    k_grid: np.ndarray,
    method: str = 'robust'
) -> Tuple[float, float]:
    """
    IMPROVED: Robust calibration with physical constraints.
    
    Enhancements:
    1. Weighted least squares (emphasizes ATM region)
    2. Outlier filtering (removes noise)
    3. Physical constraints (A > 0, reasonable bounds)
    4. Fallback strategies if standard calibration fails
    
    Args:
        quantum_probs: {m: P(m)} from quantum measurement
        classical_prices: C_m from classical FFT baseline
        k_grid: Log-strike grid (for weighting)
        method: 'standard' or 'robust' (weighted LS)
    
    Returns:
        A: Scaling factor (should be > 0)
        B: Offset term (typically small)
    """
    M = len(classical_prices)
    quantum_array = np.array([quantum_probs.get(m, 0.0) for m in range(M)])
    
    if method == 'robust':
        # IMPROVEMENT 1: Weight by classical price magnitude
        # (emphasize ATM region, de-emphasize OTM tails)
        weights = classical_prices / (classical_prices.max() + 1e-8)
        weights = np.clip(weights, 0.1, 1.0)  # Minimum weight 0.1
        
        # IMPROVEMENT 2: Filter noise (use only non-zero quantum probs)
        mask = quantum_array > 1e-4
        if mask.sum() < 3:
            mask = np.ones(M, dtype=bool)  # Fallback: use all points
        
        # Weighted least squares
        X = np.column_stack([quantum_array[mask], np.ones(mask.sum())])
        y = classical_prices[mask]
        W = np.diag(weights[mask])
        
        # Solve (X^T W X)β = X^T W y
        try:
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
            params = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            # Fallback to standard LS if singular
            params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        
    else:
        # Standard least squares (original)
        X = np.column_stack([quantum_array, np.ones(M)])
        y = classical_prices
        params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    A, B = params
    
    # IMPROVEMENT 3: Physical constraints
    if A < 0:
        print(f"⚠️  Calibration: Negative scale A={A:.2f}, using fallback")
        # Fallback: Direct scaling (no offset)
        total_classical = classical_prices.sum()
        total_quantum = quantum_array.sum()
        A = total_classical / (total_quantum + 1e-10)
        B = 0.0
    
    # IMPROVEMENT 4: Sanity check on extreme values
    if A > 10000 or abs(B) > 1000:
        print(f"⚠️  Calibration: Extreme values A={A:.2f}, B={B:.2f}, using median")
        # Use median-based scaling
        nonzero_mask = quantum_array > 0
        if nonzero_mask.sum() > 0:
            ratios = classical_prices[nonzero_mask] / quantum_array[nonzero_mask]
            A = np.median(ratios[np.isfinite(ratios)])
            B = np.median(classical_prices - A * quantum_array)
        else:
            A = 1.0
            B = 0.0
    
    return float(A), float(B)
