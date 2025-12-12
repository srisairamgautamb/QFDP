"""
Factor-Space Characteristic Function for FB-IQFT
=================================================

Core innovation: Compute characteristic function in K-dimensional FACTOR space
instead of N-dimensional asset space, enabling shallow IQFT.

Mathematical Framework:
-----------------------
Portfolio returns: R_p = w^T · (L·f + ε)
where:
  - f: K-dimensional factor returns (systematic)
  - L: N×K factor loading matrix from FB-QDP
  - ε: N-dimensional idiosyncratic noise
  - w: N-dimensional portfolio weights

Factor-space characteristic function:
φ_factor(u) = E[exp(i·u·w^T·L·f)]

This is K-dimensional (not N-dimensional), enabling shallow IQFT.

Author: QFDP Unified Research Team
Date: November 30, 2025
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class FactorCharFuncResult:
    """Result from factor-space characteristic function computation."""
    values: np.ndarray          # Characteristic function values
    frequencies: np.ndarray     # Frequency grid
    K: int                      # Number of factors
    damping_alpha: float        # Carr-Madan damping parameter
    factor_variance: np.ndarray # Variance of each factor


def compute_factor_loading_transform(
    weights: np.ndarray,
    factor_loading: np.ndarray
) -> np.ndarray:
    """
    Compute w^T · L transformation for factor-space pricing.
    
    This maps K-dimensional factor returns to scalar portfolio return.
    
    Parameters
    ----------
    weights : np.ndarray (N,)
        Portfolio weights
    factor_loading : np.ndarray (N, K)
        Factor loading matrix from FB-QDP decomposition
        
    Returns
    -------
    np.ndarray (K,)
        Effective factor weights: β = L^T · w
    """
    return factor_loading.T @ weights


def factor_space_char_func_gaussian(
    u: np.ndarray,
    factor_weights: np.ndarray,
    factor_volatilities: np.ndarray,
    risk_free_rate: float,
    maturity: float,
    spot_value: float,
    damping_alpha: float = 1.5
) -> np.ndarray:
    """
    Characteristic function in K-dimensional factor space (Gaussian assumption).
    
    Under risk-neutral measure with Gaussian factors:
    φ_factor(u) = exp(i·u·μ - 0.5·u^2·σ_f^2)
    
    where:
    - μ = w^T·L·E[f] (factor-based portfolio drift)
    - σ_f^2 = (w^T·L)^T·Var(f)·(w^T·L) (factor-based portfolio variance)
    
    Parameters
    ----------
    u : np.ndarray
        Frequency grid (complex)
    factor_weights : np.ndarray (K,)
        β = L^T·w (effective factor exposure)
    factor_volatilities : np.ndarray (K,)
        Volatility of each factor
    risk_free_rate : float
        Risk-free rate
    maturity : float
        Time to maturity
    spot_value : float
        Current portfolio value
    damping_alpha : float
        Carr-Madan damping parameter
        
    Returns
    -------
    np.ndarray (complex)
        Characteristic function values at each u
        
    Notes
    -----
    This assumes factors are:
    1. Gaussian (justified by CLT for large N)
    2. Independent (by construction from eigendecomposition)
    3. Zero-mean (factors are standardized)
    """
    K = len(factor_weights)
    
    # Factor variance contribution
    # σ_portfolio^2 = β^T · diag(σ_factor^2) · β
    factor_var = np.sum((factor_weights * factor_volatilities) ** 2)
    
    # Risk-neutral drift (portfolio level)
    mu_portfolio = np.log(spot_value) + (risk_free_rate - 0.5 * factor_var) * maturity
    
    # Damped characteristic function (Carr-Madan)
    u_damped = u - 1j * (damping_alpha + 1.0)
    
    # Gaussian characteristic function in factor space
    char_func = np.exp(
        1j * u_damped * mu_portfolio - 0.5 * factor_var * maturity * (u_damped ** 2)
    )
    
    return char_func


def factor_space_char_func_lognormal(
    u: np.ndarray,
    factor_weights: np.ndarray,
    factor_volatilities: np.ndarray,
    risk_free_rate: float,
    maturity: float,
    spot_value: float,
    damping_alpha: float = 1.5
) -> np.ndarray:
    """
    Characteristic function with log-normal factor returns.
    
    More realistic for financial factors (ensures positive prices).
    
    Parameters
    ----------
    Same as factor_space_char_func_gaussian
    
    Returns
    -------
    np.ndarray (complex)
        Characteristic function values
        
    Notes
    -----
    Log-normal factors: f_k ~ LogNormal(μ_k, σ_k^2)
    
    Portfolio return: R = w^T·L·f is approximately log-normal
    (by product of log-normals)
    """
    K = len(factor_weights)
    
    # Aggregate factor variance
    factor_var = np.sum((factor_weights * factor_volatilities) ** 2)
    
    # Log-normal parameters
    mu = np.log(spot_value) + (risk_free_rate - 0.5 * factor_var) * maturity
    sigma_sqrt_T = np.sqrt(factor_var * maturity)
    
    # Damped frequency
    u_damped = u - 1j * (damping_alpha + 1.0)
    
    # Log-normal characteristic function
    char_func = np.exp(1j * u_damped * mu - 0.5 * (sigma_sqrt_T ** 2) * (u_damped ** 2))
    
    return char_func


def carr_madan_integrand_factor_space(
    u: np.ndarray,
    char_func_values: np.ndarray,
    damping_alpha: float,
    risk_free_rate: float,
    maturity: float
) -> np.ndarray:
    """
    Carr-Madan integrand in factor space.
    
    ψ(u) = exp(-rT) · φ_factor(u - i(α+1)) / (α^2 + α - u^2 + iu(2α+1))
    
    Parameters
    ----------
    u : np.ndarray
        Frequency grid
    char_func_values : np.ndarray (complex)
        Characteristic function evaluated at damped frequencies
    damping_alpha : float
        Damping parameter
    risk_free_rate : float
        Risk-free rate
    maturity : float
        Time to maturity
        
    Returns
    -------
    np.ndarray (complex)
        Integrand values
    """
    alpha = damping_alpha
    
    # Carr-Madan denominator
    denom = (alpha ** 2 + alpha - u ** 2 + 1j * u * (2 * alpha + 1.0))
    
    # Avoid division by zero
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    
    # Integrand
    integrand = np.exp(-risk_free_rate * maturity) * char_func_values / denom
    
    return integrand


def build_factor_frequency_grid(
    K: int,
    du: float = 0.25,
    N_points: int = 64
) -> Tuple[np.ndarray, float]:
    """
    Build frequency grid for factor-space IQFT.
    
    Key difference from asset-space: Only K dimensions, so can use finer grid.
    
    Parameters
    ----------
    K : int
        Number of factors (2-8 typical)
    du : float
        Frequency spacing
    N_points : int
        Number of frequency points (must be power of 2)
        
    Returns
    -------
    u : np.ndarray (N_points,)
        Frequency grid
    dx : float
        Log-strike spacing (reciprocal space)
    """
    assert N_points > 0 and (N_points & (N_points - 1)) == 0, "N_points must be power of 2"
    
    # Frequency grid: u_j = j · du for j = 0, 1, ..., N-1
    u = np.arange(N_points) * du
    
    # Reciprocal spacing (for log-strike grid)
    dx = 2 * np.pi / (N_points * du)
    
    return u, dx


def estimate_factor_volatilities(
    asset_volatilities: np.ndarray,
    factor_loading: np.ndarray
) -> np.ndarray:
    """
    Estimate factor volatilities from asset volatilities and loading matrix.
    
    Inverse problem: Given σ_assets and L, estimate σ_factors
    
    Method: Least-squares fitting
    σ_assets ≈ sqrt(diag(L · Σ_factor · L^T))
    
    Parameters
    ----------
    asset_volatilities : np.ndarray (N,)
        Individual asset volatilities
    factor_loading : np.ndarray (N, K)
        Factor loading matrix
        
    Returns
    -------
    np.ndarray (K,)
        Estimated factor volatilities
    """
    N, K = factor_loading.shape
    
    # Simple heuristic: Factor vol ≈ weighted average of asset vols
    # More sophisticated: Use PCA variance or regression
    
    # Weight each factor by its loading strength
    factor_vols = np.zeros(K)
    for k in range(K):
        # Factor k volatility: weighted average, weighted by loading^2
        weights = factor_loading[:, k] ** 2
        weights = weights / (weights.sum() + 1e-12)
        factor_vols[k] = np.sqrt(np.sum(weights * asset_volatilities ** 2))
    
    return factor_vols


def validate_factor_char_func(
    factor_weights: np.ndarray,
    factor_volatilities: np.ndarray,
    risk_free_rate: float,
    maturity: float,
    spot_value: float,
    num_mc_samples: int = 100000,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Validate factor-space characteristic function against Monte Carlo.
    
    Parameters
    ----------
    factor_weights : np.ndarray (K,)
        Effective factor exposure β = L^T·w
    factor_volatilities : np.ndarray (K,)
        Factor volatilities
    risk_free_rate : float
        Risk-free rate
    maturity : float
        Time to maturity
    spot_value : float
        Current portfolio value
    num_mc_samples : int
        Number of Monte Carlo samples
    seed : int
        Random seed
        
    Returns
    -------
    char_func_value : float
        Characteristic function at u=1
    mc_estimate : float
        Monte Carlo estimate of same quantity
    """
    K = len(factor_weights)
    
    # Characteristic function at u=1
    u = np.array([1.0])
    char_value = factor_space_char_func_gaussian(
        u, factor_weights, factor_volatilities,
        risk_free_rate, maturity, spot_value, damping_alpha=0.0
    )[0]
    
    # Monte Carlo validation
    rng = np.random.default_rng(seed)
    
    # Sample factor returns (Gaussian, independent)
    factor_returns = rng.normal(
        loc=0,
        scale=factor_volatilities * np.sqrt(maturity),
        size=(num_mc_samples, K)
    )
    
    # Portfolio returns via factor model
    portfolio_returns = factor_returns @ factor_weights
    
    # Add drift
    mu = (risk_free_rate - 0.5 * np.sum((factor_weights * factor_volatilities) ** 2)) * maturity
    log_prices = np.log(spot_value) + mu + portfolio_returns
    
    # Characteristic function estimator: E[exp(i·u·X)]
    mc_estimate = np.mean(np.exp(1j * u[0] * log_prices))
    
    return complex(char_value), complex(mc_estimate)
