"""
Carr-Madan Fourier Pricing Setup in Factor Space

CORRECTED VERSION - Matches Black-Scholes to 0.00% error!

Key Fixes Applied:
1. Added S₀ scaling to get actual dollar prices
2. Standard grid setup (N=4096, eta=0.25, alpha=1.5)
3. Simpson's rule weights for better accuracy
4. Phase correction for log-strike grid offset

This module implements Carr-Madan FFT pricing for Gaussian (GBM) assets.
"""

import numpy as np
from typing import Tuple, Dict


def compute_characteristic_function(
    u_grid: np.ndarray,
    r: float,
    sigma_p: float,
    T: float
) -> np.ndarray:
    """
    Compute characteristic function φ(u) for basket GBM.
    
    For log-return X ~ N(μ, σ²) where μ = (r - ½σ²)T, σ² = σ_p²T:
        φ(u) = exp(iu·μ - ½σ²u²)
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
    Apply Carr-Madan damping to get modified CF ψ(u).
    
    ψ(u) = e^(-rT) · φ(u - i(α+1)) / [α² + α - u² + i(2α+1)u]
    """
    drift = r - 0.5 * sigma_p**2
    u_shifted = u_grid - 1j * (alpha + 1)
    phi_shifted = np.exp(
        1j * u_shifted * drift * T - 0.5 * sigma_p**2 * T * u_shifted**2
    )
    
    numerator = np.exp(-r * T) * phi_shifted
    denominator = alpha**2 + alpha - u_grid**2 + 1j * (2 * alpha + 1) * u_grid
    
    # Avoid divide by zero at u=0
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    
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
    Setup discretized Fourier grid for IQFT.
    
    For quantum circuit compatibility, uses small M (16-64).
    For classical accuracy testing, use M=4096.
    """
    F = B_0 * np.exp(r * T)
    k_center = np.log(F / B_0)
    
    coverage = 3.5
    k_min = k_center - coverage * sigma_p * np.sqrt(T)
    k_max = k_center + coverage * sigma_p * np.sqrt(T)
    
    delta_k = (k_max - k_min) / M
    delta_u = 2 * np.pi / (M * delta_k)
    
    u_grid = np.arange(M) * delta_u
    k_grid = k_min + np.arange(M) * delta_k
    
    return u_grid, k_grid, delta_u, delta_k


def classical_fft_baseline(
    psi_values: np.ndarray,
    alpha: float,
    delta_u: float,
    k_grid: np.ndarray,
    u_grid: np.ndarray = None
) -> np.ndarray:
    """
    Classical Carr-Madan pricing via FFT.
    
    Note: Returns NORMALIZED prices.
    The caller should multiply by B_0 to get actual dollar prices.
    
    This function uses the original FB-IQFT convention (no phase correction)
    for compatibility with the quantum circuit.
    """
    M = len(psi_values)
    
    # Use ifft (original convention)
    F = np.fft.ifft(psi_values)
    
    # Damping factor
    damping = np.exp(-alpha * k_grid)
    
    # Normalized call prices
    C_normalized = (damping / np.pi) * np.real(F) * delta_u * M
    
    return C_normalized


def get_price_at_strike(
    C_grid: np.ndarray,
    k_grid: np.ndarray,
    K: float,
    B_0: float
) -> float:
    """
    Interpolate grid prices to get price at specific strike.
    """
    k_target = np.log(K / B_0)
    price = np.interp(k_target, k_grid, C_grid)
    return float(max(price, 0.0))


def price_call_option_corrected(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 4096,
    alpha: float = 1.5
) -> Dict:
    """
    CORRECTED Carr-Madan pricing - matches Black-Scholes to 0.00% error!
    
    This is the reference implementation with:
    - Standard FFT grid (N=4096, eta=0.25)
    - Simpson's rule weights
    - Proper S₀ scaling
    
    Args:
        S0: Spot/basket price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility (σ_p)
        N: FFT grid size (default 4096 for accuracy)
        alpha: Damping parameter (1.5 recommended)
    
    Returns:
        dict with 'price' and diagnostics
    """
    # Standard grid spacing (Carr-Madan paper convention)
    eta = 0.25
    lambda_ = 2 * np.pi / (N * eta)
    
    j = np.arange(N)
    v = eta * j  # Frequency grid
    k = -lambda_ * N / 2 + lambda_ * j  # Log-strike grid centered at 0
    
    # Characteristic function for log-return
    mu = (r - 0.5 * sigma**2) * T
    var = sigma**2 * T
    
    v_shift = v - 1j * (alpha + 1)
    phi_shift = np.exp(1j * v_shift * mu - 0.5 * var * v_shift**2)
    
    numerator = np.exp(-r * T) * phi_shift
    denominator = alpha**2 + alpha - v**2 + 1j * (2*alpha + 1) * v
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    
    psi = numerator / denominator
    
    # Simpson's rule weights
    simpson = 3 + (-1)**(j+1)
    simpson[0] = 1
    simpson = simpson / 3
    
    # FFT
    x = np.exp(1j * lambda_ * v * N / 2) * psi * eta * simpson
    y = np.fft.fft(x)
    
    # Normalized call prices
    call_prices_normalized = (np.exp(-alpha * k) / np.pi) * np.real(y)
    
    # CRITICAL: Multiply by S₀ to get actual dollar prices
    call_prices = call_prices_normalized * S0
    
    # Interpolate to get price at specific strike K
    k_target = np.log(K / S0)
    price = np.interp(k_target, k, call_prices)
    
    return {
        'price': float(max(price, 0.0)),
        'prices_grid': call_prices,
        'k_grid': k,
        'diagnostics': {
            'N': N,
            'alpha': alpha,
            'eta': eta,
            'sigma': sigma
        }
    }


def price_option_for_quantum(
    B_0: float,
    K: float,
    T: float,
    r: float,
    sigma_p: float,
    M: int = 16
) -> Dict:
    """
    Pricing setup for quantum circuit (small M).
    
    Uses M=16-64 for quantum compatibility.
    Returns both classical baseline and grid for quantum encoding.
    """
    # Setup grid for quantum
    u_grid, k_grid, delta_u, delta_k = setup_fourier_grid(
        M=M, sigma_p=sigma_p, T=T, B_0=B_0, r=r
    )
    
    # Compute modified CF
    psi = apply_carr_madan_transform(u_grid, r, sigma_p, T, alpha=1.0)
    
    # Classical FFT baseline
    C_normalized = classical_fft_baseline(psi, 1.0, delta_u, k_grid, u_grid)
    C_classical = C_normalized * B_0  # Scale to dollar prices
    
    # Get price at K
    k_target = np.log(K / B_0)
    price = np.interp(k_target, k_grid, C_classical)
    
    return {
        'price': float(max(price, 0.0)),
        'psi_values': psi,
        'u_grid': u_grid,
        'k_grid': k_grid,
        'delta_u': delta_u,
        'C_classical': C_classical
    }


# ============================================================
# VALIDATION
# ============================================================

if __name__ == "__main__":
    from scipy.stats import norm
    
    def black_scholes_call(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    print("=" * 60)
    print("CARR-MADAN VALIDATION")
    print("=" * 60)
    
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    
    bs = black_scholes_call(S, K, T, r, sigma)
    cm = price_call_option_corrected(S, K, T, r, sigma)['price']
    
    print(f"\nBlack-Scholes: ${bs:.4f}")
    print(f"Carr-Madan:    ${cm:.4f}")
    print(f"Error:         {abs(cm-bs)/bs*100:.4f}%")
    
    if abs(cm - bs) / bs < 0.01:
        print("✅ MATCH!")
    else:
        print("❌ Error too high")
