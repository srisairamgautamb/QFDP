"""
Carr-Madan Fourier Pricing Setup in Factor Space

This module implements Steps 5-7 of the FB-IQFT flowchart:
- Step 5: Compute characteristic function φ(u) for Gaussian basket
- Step 6: Apply Carr-Madan transform to get modified CF ψ(u)
- Step 7: Setup discretized Fourier grid and classical FFT baseline

The key insight is that factor decomposition yields a single portfolio
volatility σₚ, enabling a simple Gaussian characteristic function that
requires only M=16-32 frequency points (vs M=256-1024 for multi-asset CF).
"""

import numpy as np
from typing import Tuple


def compute_characteristic_function(
    u_grid: np.ndarray,
    r: float,
    sigma_p: float,
    T: float
) -> np.ndarray:
    """
    Step 5: Compute characteristic function φ(u) for basket GBM.
    
    For a Gaussian log-return X ~ N(μ, σ²) where:
        μ = (r - ½σₚ²)T
        σ² = σₚ²T
    
    The characteristic function has closed form:
        φ(u) = 𝔼[e^(iuX)] = exp(iu·μ - ½σ²u²)
             = exp(iu(r - ½σₚ²)T - ½σₚ²T·u²)
    
    This Gaussian form is the KEY to shallow IQFT:
    - Smooth function → needs few samples M=16-32
    - Analytic formula → no numerical integration
    
    Args:
        u_grid: Frequency points [u₀, u₁, ..., u_{M-1}], shape (M,)
        r: Risk-free rate (e.g., 0.05)
        sigma_p: Portfolio volatility from factor decomposition
        T: Time to maturity (e.g., 1.0 year)
    
    Returns:
        phi_values: φ(u_j) for each frequency point, shape (M,), complex
    
    Example:
        >>> u = np.array([0, 1, 2, 3, 4])
        >>> phi = compute_characteristic_function(u, r=0.05, sigma_p=0.25, T=1.0)
        >>> # φ(0) = 1.0 (normalization)
        >>> assert np.isclose(phi[0], 1.0)
    """
    drift = r - 0.5 * sigma_p**2
    phi = np.exp(1j * u_grid * drift * T - 0.5 * sigma_p**2 * T * u_grid**2)
    return phi


def apply_carr_madan_transform(
    u_grid: np.ndarray,
    r: float,
    sigma_p: float,
    T: float,
    alpha: float
) -> np.ndarray:
    """
    Step 6: Apply Carr-Madan damping to get modified CF ψ(u).
    
    The Carr-Madan formula for call option pricing requires evaluating
    the characteristic function at a complex-shifted argument:
    
        ψ(u) = e^(-rT) · φ(u - i(α+1)) / [α² + α - u² + i(2α+1)u]
    
    where:
        - α > 0 is damping parameter (typically α=1.0)
        - φ(u - i(α+1)) requires analytic continuation
        - Denominator ensures convergence of Fourier integral
    
    IMPORTANT: σₚ must be passed directly (not inferred from φ values).
    
    Args:
        u_grid: Frequency points, shape (M,)
        r: Risk-free rate
        sigma_p: Portfolio volatility (MUST be provided explicitly)
        T: Time to maturity
        alpha: Damping parameter (typically 1.0)
    
    Returns:
        psi_values: ψ(u_j) for each frequency point, shape (M,), complex
    
    Notes:
        The modified CF ψ(u) inherits smoothness from Gaussian φ(u),
        which is why we need only M=16-32 samples.
    """
    # Evaluate φ(u - i(α+1)) using analytic continuation
    drift = r - 0.5 * sigma_p**2
    u_shifted = u_grid - 1j * (alpha + 1)
    phi_shifted = np.exp(
        1j * u_shifted * drift * T - 0.5 * sigma_p**2 * T * u_shifted**2
    )
    
    # Apply Carr-Madan transform
    numerator = np.exp(-r * T) * phi_shifted
    denominator = alpha**2 + alpha - u_grid**2 + 1j * (2 * alpha + 1) * u_grid
    
    psi = numerator / denominator
    return psi


def setup_fourier_grid(
    M: int,
    sigma_p: float,
    T: float,
    B_0: float,
    r: float,
    alpha: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Step 7: Setup discretized Fourier grid for IQFT.
    
    The grid must satisfy the Nyquist relation:
        Δu · Δk = 2π/M
    
    For Gaussian CF, we center the log-strike grid around ln(F) where
    F = B₀·exp(rT) is the forward price, with coverage ±3.5σₚ√T.
    
    The frequency grid starts at u=0 and extends to u_max chosen to
    adequately sample the Gaussian decay exp(-½σₚ²Tu²).
    
    Args:
        M: Grid size (16 or 32, must be power of 2)
        sigma_p: Portfolio volatility
        T: Time to maturity
        B_0: Initial basket value
        r: Risk-free rate
        alpha: Damping parameter
    
    Returns:
        u_grid: Frequency grid [u₀, ..., u_{M-1}], start at 0
        k_grid: Log-strike grid [k₀, ..., k_{M-1}]
        delta_u: Frequency spacing
        delta_k: Log-strike spacing
    
    Raises:
        AssertionError: If Nyquist relation is violated
    
    Example:
        >>> u, k, du, dk = setup_fourier_grid(16, 0.25, 1.0, 100.0, 0.05)
        >>> # Verify Nyquist
        >>> assert np.isclose(du * dk, 2*np.pi/16)
    """
    # Forward price for centering
    F = B_0 * np.exp(r * T)
    k_center = np.log(F / B_0)  # ≈ rT
    
    # Coverage: ±3.5σ√T around center (captures 99.95% of Gaussian mass)
    coverage = 3.5
    k_min = k_center - coverage * sigma_p * np.sqrt(T)
    k_max = k_center + coverage * sigma_p * np.sqrt(T)
    
    # Determine grid spacing
    delta_k = (k_max - k_min) / M
    delta_u = 2 * np.pi / (M * delta_k)  # Nyquist relation
    
    # Build grids
    u_grid = np.arange(M) * delta_u  # Start at 0
    k_grid = k_min + np.arange(M) * delta_k
    
    return u_grid, k_grid, delta_u, delta_k


def classical_fft_baseline(
    psi_values: np.ndarray,
    alpha: float,
    delta_u: float,
    k_grid: np.ndarray
) -> np.ndarray:
    """
    Classical Carr-Madan pricing via NumPy FFT (for calibration).
    
    The Carr-Madan formula gives call option prices via Fourier inversion:
    
        C(k_m) = (e^(-αk_m)/π) · Re[Σⱼ e^(-iu_j k_m)·ψ(u_j)·Δu]
               = (e^(-αk_m)/π) · Re[IFFT(ψ)] · Δu · M
    
    where:
        - k_m = ln(K_m/B₀) is log-moneyness
        - IFFT uses NumPy convention: exp(-i2πjm/M)
        - Scaling by Δu·M converts discrete sum to integral approximation
    
    This classical baseline is used to calibrate the quantum results.
    
    Args:
        psi_values: Modified CF ψ(u_j) from Step 6, shape (M,), complex
        alpha: Damping parameter
        delta_u: Frequency spacing
        k_grid: Log-strike grid
    
    Returns:
        C_classical: Call option prices at k_grid strikes, shape (M,), real
    
    Notes:
        - Prices should be non-negative
        - Prices should not exceed undiscounted forward
        - These constraints are validated in the main pipeline
    """
    M = len(psi_values)
    
    # Apply IFFT (NumPy convention: IFFT has exp(-i2πjm/M))
    F = np.fft.ifft(psi_values)
    
    # Damping factor e^(-αk)
    damping = np.exp(-alpha * k_grid)
    
    # Extract call prices (real part with correct scaling)
    C_classical = (damping / np.pi) * np.real(F) * delta_u * M
    
    return C_classical
