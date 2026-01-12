"""
Empirical Characteristic Function for QML-IQFT
===============================================

Compute empirical characteristic function from market data for QNN training.

Implements Section 5.1 from QML_QHDP.pdf:
- Empirical CF computation from returns
- Frequency grid generation
- QNN training data preparation

Author: QFDP Research Team
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class EmpiricalCFResult:
    """
    Result from empirical characteristic function computation.
    
    Attributes
    ----------
    cf_real : np.ndarray
        Real part of CF values
    cf_imag : np.ndarray
        Imaginary part of CF values
    frequency_grid : np.ndarray
        Frequency points u
    n_samples : int
        Number of samples used
    """
    cf_real: np.ndarray
    cf_imag: np.ndarray
    frequency_grid: np.ndarray
    n_samples: int


def compute_empirical_cf(
    returns: np.ndarray,
    frequency_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical characteristic function from returns data.
    
    φ_T(u) = (1/T) Σ_{t=1}^T exp(i u^T R_t)
    
    Parameters
    ----------
    returns : np.ndarray
        Log returns (T × N) or (T,) for univariate
    frequency_grid : np.ndarray
        Frequency points u (M,) or (M × N) for multivariate
        
    Returns
    -------
    cf_real : np.ndarray
        Real part of CF values (M,)
    cf_imag : np.ndarray
        Imaginary part of CF values (M,)
        
    Notes
    -----
    For portfolio returns (scalar), the CF is:
    φ(u) = E[exp(i·u·r)] ≈ (1/T) Σ exp(i·u·r_t)
    
    Examples
    --------
    >>> u_grid = np.linspace(-5, 5, 50)
    >>> cf_real, cf_imag = compute_empirical_cf(returns, u_grid)
    """
    # Handle DataFrame
    if hasattr(returns, 'values'):
        returns = returns.values
    
    # Ensure 1D for portfolio returns
    if returns.ndim == 2:
        # Sum across assets for portfolio characteristic
        portfolio_returns = returns.sum(axis=1)
    else:
        portfolio_returns = returns
    
    T = len(portfolio_returns)
    M = len(frequency_grid)
    
    cf_real = np.zeros(M)
    cf_imag = np.zeros(M)
    
    for i, u in enumerate(frequency_grid):
        # φ(u) = (1/T) Σ exp(i·u·r_t)
        exponents = np.exp(1j * u * portfolio_returns)
        cf = np.mean(exponents)
        
        cf_real[i] = cf.real
        cf_imag[i] = cf.imag
    
    return cf_real, cf_imag


def compute_multivariate_empirical_cf(
    returns: np.ndarray,
    frequency_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute multivariate empirical characteristic function.
    
    φ_T(u) = (1/T) Σ_{t=1}^T exp(i u^T R_t)
    
    Parameters
    ----------
    returns : np.ndarray
        Log returns (T × N)
    frequency_matrix : np.ndarray
        Frequency points (M × N)
        
    Returns
    -------
    cf_real : np.ndarray
        Real part of CF values (M,)
    cf_imag : np.ndarray
        Imaginary part of CF values (M,)
    """
    if hasattr(returns, 'values'):
        returns = returns.values
    
    T, N = returns.shape
    M = frequency_matrix.shape[0]
    
    cf_real = np.zeros(M)
    cf_imag = np.zeros(M)
    
    for i, u in enumerate(frequency_matrix):
        # u^T @ R_t for each time t
        dot_products = returns @ u  # (T,)
        exponents = np.exp(1j * dot_products)
        cf = np.mean(exponents)
        
        cf_real[i] = cf.real
        cf_imag[i] = cf.imag
    
    return cf_real, cf_imag


def generate_frequency_grid(
    u_max: float = 5.0,
    n_frequencies: int = 50,
    include_negative: bool = True
) -> np.ndarray:
    """
    Generate frequency grid for CF computation.
    
    Parameters
    ----------
    u_max : float
        Maximum frequency value
    n_frequencies : int
        Number of frequency points
    include_negative : bool
        Include negative frequencies (symmetric grid)
        
    Returns
    -------
    frequency_grid : np.ndarray
        Frequency points (n_frequencies,)
    """
    if include_negative:
        return np.linspace(-u_max, u_max, n_frequencies)
    else:
        return np.linspace(0, u_max, n_frequencies)


def prepare_qnn_training_data(
    returns: np.ndarray,
    n_frequencies: int = 50,
    u_max: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training data for Quantum Neural Network.
    
    Parameters
    ----------
    returns : np.ndarray or pd.DataFrame
        Log returns (T × N) or (T,)
    n_frequencies : int
        Number of frequency points
    u_max : float
        Maximum frequency value
        
    Returns
    -------
    X : np.ndarray
        Input features - frequency values (M × 1)
    y : np.ndarray
        Target - CF values [real, imag] (M × 2)
    frequency_grid : np.ndarray
        Frequency grid used
        
    Examples
    --------
    >>> X, y, u_grid = prepare_qnn_training_data(returns)
    >>> print(f"Training samples: {len(X)}")
    """
    # Generate frequency grid
    frequency_grid = generate_frequency_grid(u_max, n_frequencies)
    
    # Compute empirical CF
    cf_real, cf_imag = compute_empirical_cf(returns, frequency_grid)
    
    # Format for training
    X = frequency_grid.reshape(-1, 1)
    y = np.column_stack([cf_real, cf_imag])
    
    print(f"✅ Prepared QNN training data:")
    print(f"   Frequency points: {n_frequencies}")
    print(f"   Frequency range: [{-u_max:.1f}, {u_max:.1f}]")
    print(f"   Input shape X: {X.shape}")
    print(f"   Target shape y: {y.shape}")
    
    return X, y, frequency_grid


def validate_cf_properties(
    cf_real: np.ndarray,
    cf_imag: np.ndarray,
    frequency_grid: np.ndarray,
    verbose: bool = True
) -> dict:
    """
    Validate characteristic function properties.
    
    Properties checked:
    1. φ(0) = 1 (normalization)
    2. |φ(u)| ≤ 1 (bounded)
    3. Hermitian symmetry: φ(-u) = φ(u)*
    
    Parameters
    ----------
    cf_real : np.ndarray
        Real part of CF
    cf_imag : np.ndarray
        Imaginary part of CF
    frequency_grid : np.ndarray
        Frequency points
    verbose : bool
        Print validation results
        
    Returns
    -------
    checks : dict
        Validation results for each property
    """
    checks = {}
    
    # Property 1: φ(0) = 1
    zero_idx = np.argmin(np.abs(frequency_grid))
    cf_at_zero = cf_real[zero_idx] + 1j * cf_imag[zero_idx]
    checks['normalization'] = np.abs(cf_at_zero - 1.0) < 0.1
    
    # Property 2: |φ(u)| ≤ 1
    magnitudes = np.sqrt(cf_real**2 + cf_imag**2)
    checks['bounded'] = np.all(magnitudes <= 1.0 + 1e-6)
    
    # Property 3: Hermitian symmetry (approximate)
    # For symmetric grid, φ(-u) ≈ φ(u)*
    if np.isclose(frequency_grid[0], -frequency_grid[-1]):
        cf_pos = cf_real[len(cf_real)//2:] + 1j * cf_imag[len(cf_imag)//2:]
        cf_neg = cf_real[:len(cf_real)//2][::-1] + 1j * cf_imag[:len(cf_imag)//2][::-1]
        symmetry_error = np.mean(np.abs(cf_neg - np.conj(cf_pos)))
        checks['hermitian_symmetry'] = symmetry_error < 0.1
    else:
        checks['hermitian_symmetry'] = None
    
    if verbose:
        print("\n📊 CF Property Validation:")
        print(f"   φ(0) = 1: {'✅' if checks['normalization'] else '❌'}")
        print(f"   |φ(u)| ≤ 1: {'✅' if checks['bounded'] else '❌'}")
        if checks['hermitian_symmetry'] is not None:
            print(f"   Hermitian symmetry: {'✅' if checks['hermitian_symmetry'] else '❌'}")
    
    return checks


if __name__ == '__main__':
    # Test with synthetic Gaussian returns
    np.random.seed(42)
    T = 1000
    returns = 0.001 + 0.02 * np.random.randn(T)  # μ=0.1%, σ=2%
    
    X, y, u_grid = prepare_qnn_training_data(returns, n_frequencies=50)
    
    # Validate properties
    checks = validate_cf_properties(y[:, 0], y[:, 1], u_grid)
    
    print(f"\n✅ Characteristic function test complete")
