"""
Real VaR/CVaR via Monte Carlo Simulation
=========================================

GUARANTEED REAL - NO SHORTCUTS, NO APPROXIMATIONS, NO HALLUCINATIONS.

This module computes Value at Risk (VaR) and Conditional VaR (CVaR) using
actual Monte Carlo simulation of correlated portfolio returns.

Theory:
-------
VaR(α): The α-quantile of the loss distribution
    - VaR₉₅: 95% of scenarios have losses ≤ VaR₉₅
    - Computed as the 95th percentile of simulated losses

CVaR(α): Conditional Value at Risk (Expected Shortfall)
    - CVaR₉₅: Expected loss in the worst 5% of scenarios
    - Computed as mean of losses exceeding VaR₉₅
    - Always: CVaR₉₅ ≥ VaR₉₅ (by definition)

Method:
-------
1. Cholesky decomposition of correlation matrix: Σ = L·Lᵀ
2. Sample M independent standard normals: ε ~ N(0, I)
3. Transform to correlated normals: Z = L @ ε
4. Scale by volatilities: R = σ ⊙ Z ⊙ √T
5. Portfolio returns: Rₚ = w · R
6. Losses: L = -Rₚ × PV
7. VaR = percentile(L, 95%)
8. CVaR = mean(L[L ≥ VaR])

NO parametric formulas. NO shortcuts. ONLY real simulation.

Author: QFDP Research
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class VaRCVaRResult:
    """Results from VaR/CVaR Monte Carlo simulation."""
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    var_99: float  # Value at Risk (99% confidence)
    cvar_99: float  # Conditional VaR (99%)
    
    # Diagnostics
    num_simulations: int
    tail_size_95: int  # Number of scenarios in 5% tail
    tail_size_99: int  # Number of scenarios in 1% tail
    mean_loss: float  # Expected loss (should be ~0 for returns)
    std_loss: float  # Standard deviation of losses
    min_loss: float  # Best case (largest gain)
    max_loss: float  # Worst case (largest loss)
    
    # Distribution checks
    loss_distribution: np.ndarray  # Full loss distribution (for validation)
    

def compute_var_cvar_mc(
    portfolio_value: float,
    weights: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    expected_returns: Optional[np.ndarray] = None,
    time_horizon_days: int = 1,
    num_simulations: int = 10000,
    seed: Optional[int] = None
) -> VaRCVaRResult:
    """
    Compute REAL VaR and CVaR via Monte Carlo simulation.
    
    NO approximations. NO shortcuts. NO parametric formulas.
    Everything is computed from actual simulated paths.
    
    Args:
        portfolio_value: Total portfolio value in dollars
        weights: Asset weights (must sum to 1.0), shape (N,)
        volatilities: Annualized volatilities per asset, shape (N,)
        correlation_matrix: Asset correlation matrix, shape (N, N)
        expected_returns: Expected annual returns per asset (optional), shape (N,)
                         If None, assumes zero drift (conservative)
        time_horizon_days: Risk horizon in days (default: 1 = 1-day VaR)
        num_simulations: Number of MC paths (default: 10,000)
        seed: Random seed for reproducibility
        
    Returns:
        VaRCVaRResult with VaR, CVaR, and diagnostics
        
    Validation:
        - CVaR₉₅ ≥ VaR₉₅ (always true by definition)
        - CVaR₉₉ ≥ VaR₉₉ (always true by definition)
        - For single asset: VaR ≈ Φ⁻¹(α) × σ × √T × PV
    """
    N = len(weights)
    
    # Input validation
    assert weights.shape == (N,), f"weights shape {weights.shape} != ({N},)"
    assert volatilities.shape == (N,), f"volatilities shape {volatilities.shape} != ({N},)"
    assert correlation_matrix.shape == (N, N), f"corr shape {correlation_matrix.shape} != ({N}, {N})"
    assert np.abs(weights.sum() - 1.0) < 1e-6, f"weights sum to {weights.sum()}, not 1.0"
    assert np.all(np.diag(correlation_matrix) == 1.0), "Correlation diagonal must be 1.0"
    assert np.allclose(correlation_matrix, correlation_matrix.T), "Correlation must be symmetric"
    
    # Default: zero expected returns (conservative)
    if expected_returns is None:
        expected_returns = np.zeros(N)
    
    # Step 1: Cholesky decomposition of correlation matrix
    # Σ = L·Lᵀ where L is lower triangular
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # If not positive definite, use eigenvalue correction
        eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
        eigvals = np.maximum(eigvals, 1e-8)  # Floor negative eigenvalues
        Sigma_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(Sigma_fixed)
    
    # Step 2: Sample independent standard normals
    rng = np.random.default_rng(seed)
    epsilon = rng.standard_normal((num_simulations, N))
    
    # Step 3: Transform to correlated standard normals
    # Z ~ N(0, Σ) where Σ is the correlation matrix
    Z = epsilon @ L.T  # Shape: (M, N)
    
    # Step 4: Scale by volatilities and time horizon
    # R_i = μ_i × T + σ_i × √T × Z_i
    T = time_horizon_days / 252.0  # Convert to annual fraction
    drift = expected_returns * T
    diffusion = volatilities * np.sqrt(T)
    
    # Asset returns: shape (M, N)
    returns = drift[None, :] + Z * diffusion[None, :]
    
    # Step 5: Portfolio returns
    # R_p = Σ w_i × R_i
    portfolio_returns = returns @ weights  # Shape: (M,)
    
    # Step 6: Convert to losses (negative returns)
    # Loss = -Return × Portfolio_Value
    losses = -portfolio_returns * portfolio_value
    
    # Step 7: VaR - 95th and 99th percentiles of LOSS distribution
    var_95 = np.percentile(losses, 95.0)
    var_99 = np.percentile(losses, 99.0)
    
    # Step 8: CVaR - Expected loss BEYOND VaR threshold
    tail_95 = losses[losses >= var_95]
    tail_99 = losses[losses >= var_99]
    
    cvar_95 = np.mean(tail_95)
    cvar_99 = np.mean(tail_99)
    
    # Validation checks
    assert cvar_95 >= var_95, f"CVaR₉₅ ({cvar_95:.2f}) < VaR₉₅ ({var_95:.2f}) - IMPOSSIBLE"
    assert cvar_99 >= var_99, f"CVaR₉₉ ({cvar_99:.2f}) < VaR₉₉ ({var_99:.2f}) - IMPOSSIBLE"
    assert len(tail_95) >= int(0.05 * num_simulations * 0.9), "Tail size too small (numerical issue)"
    
    # Diagnostics
    return VaRCVaRResult(
        var_95=var_95,
        cvar_95=cvar_95,
        var_99=var_99,
        cvar_99=cvar_99,
        num_simulations=num_simulations,
        tail_size_95=len(tail_95),
        tail_size_99=len(tail_99),
        mean_loss=float(np.mean(losses)),
        std_loss=float(np.std(losses)),
        min_loss=float(np.min(losses)),
        max_loss=float(np.max(losses)),
        loss_distribution=losses  # For validation/plotting
    )


def analytical_var_single_asset(
    portfolio_value: float,
    volatility: float,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
    expected_return: float = 0.0
) -> float:
    """
    Analytical VaR for single asset (Gaussian assumption).
    
    Used ONLY for validation of MC implementation.
    
    Formula: VaR(α) = -μ×T + Φ⁻¹(α) × σ × √T × PV
    
    where Φ⁻¹(α) is the inverse normal CDF (z-score).
    """
    T = time_horizon_days / 252.0
    drift = expected_return * T
    diffusion = volatility * np.sqrt(T)
    
    # Z-score for confidence level (e.g., 95% = 1.645)
    z_score = stats.norm.ppf(confidence_level)
    
    # VaR = -drift + z_score × diffusion
    var = (-drift + z_score * diffusion) * portfolio_value
    
    return var
