"""
PCA Factor Model for QML-IQFT
==============================

PCA-based factor decomposition with risk-neutral drift adjustment.

Implements Section 3 from QML_QHDP.pdf:
- Eigenvalue decomposition of correlation matrix
- Risk-neutral drift transformation
- Factor space projection

Author: QFDP Research Team
Date: January 2026
"""

import numpy as np
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class FactorModelResult:
    """
    Result from PCA factor model fitting.
    
    Attributes
    ----------
    n_factors : int
        Number of factors (K)
    explained_variance_ratio : np.ndarray
        Variance explained by each factor
    total_variance_explained : float
        Cumulative variance explained
    components : np.ndarray
        Factor loading matrix Q_K (K × N)
    eigenvalues : np.ndarray
        Eigenvalues (variances) of factors
    risk_neutral_drift : np.ndarray
        Risk-neutral drift μ_F (K,)
    """
    n_factors: int
    explained_variance_ratio: np.ndarray
    total_variance_explained: float
    components: np.ndarray
    eigenvalues: np.ndarray
    risk_neutral_drift: np.ndarray


class PCAFactorModel:
    """
    PCA-based factor decomposition with risk-neutral adjustment.
    
    Performs dimensionality reduction from N assets to K factors,
    enabling efficient QML characteristic function learning.
    
    Mathematical Framework:
    -----------------------
    1. Eigendecomposition: Σ = V Λ V^T
    2. Factor loading: Q_K = V[:, :K] (top K eigenvectors)
    3. Risk-neutral drift (Eq 3.1):
       μ_F = Q_K @ (r·1 - 0.5·diag(Σ))
    
    Parameters
    ----------
    n_factors : int, optional
        Number of factors. If None, auto-select based on variance_threshold
    variance_threshold : float
        Minimum variance to explain (used if n_factors is None)
        
    Examples
    --------
    >>> model = PCAFactorModel(n_factors=3)
    >>> result = model.fit(returns, risk_free_rate=0.05)
    >>> factors = model.transform(returns)
    """
    
    def __init__(
        self,
        n_factors: Optional[int] = None,
        variance_threshold: float = 0.85
    ):
        self.n_factors = n_factors
        self.variance_threshold = variance_threshold
        self.pca = None
        self._fitted = False
        self._result = None
    
    def fit(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.05,
        annualization_factor: float = 252
    ) -> FactorModelResult:
        """
        Fit PCA factor model to returns data.
        
        Parameters
        ----------
        returns : np.ndarray or pd.DataFrame
            Log returns (T × N)
        risk_free_rate : float
            Annual risk-free rate
        annualization_factor : float
            Trading days per year (for annualization)
            
        Returns
        -------
        FactorModelResult
            Fitted model results
            
        Notes
        -----
        Risk-neutral drift adjustment per Eq 3.1:
        μ_F = Q_K^T @ (r·1 - 0.5·diag(Σ))
        """
        # Convert to numpy if DataFrame
        if hasattr(returns, 'values'):
            returns = returns.values
        
        N = returns.shape[1]
        
        # Determine number of factors
        if self.n_factors is None:
            n_factors = self._auto_select_factors(returns)
        else:
            n_factors = min(self.n_factors, N)
        
        # Fit PCA
        self.pca = PCA(n_components=n_factors)
        self.pca.fit(returns)
        
        # Extract results
        explained_variance_ratio = self.pca.explained_variance_ratio_
        total_variance_explained = explained_variance_ratio.sum()
        components = self.pca.components_  # Q_K: (K × N)
        eigenvalues = self.pca.explained_variance_
        
        # Compute risk-neutral drift (Eq 3.1)
        # μ_F = Q_K @ (r·1 - 0.5·diag(Σ))
        daily_rate = risk_free_rate / annualization_factor
        Sigma = np.cov(returns.T)
        drift_adjustment = daily_rate * np.ones(N) - 0.5 * np.diag(Sigma)
        risk_neutral_drift = components @ drift_adjustment
        
        self._result = FactorModelResult(
            n_factors=n_factors,
            explained_variance_ratio=explained_variance_ratio,
            total_variance_explained=total_variance_explained,
            components=components,
            eigenvalues=eigenvalues,
            risk_neutral_drift=risk_neutral_drift
        )
        
        self._fitted = True
        
        print(f"✅ Fitted {n_factors} factors")
        print(f"   Explained variance: {total_variance_explained:.2%}")
        print(f"   Individual ratios: {explained_variance_ratio}")
        
        return self._result
    
    def _auto_select_factors(self, returns: np.ndarray) -> int:
        """Auto-select K based on variance threshold."""
        # Fit full PCA to get eigenvalue spectrum
        full_pca = PCA()
        full_pca.fit(returns)
        
        cumulative_var = np.cumsum(full_pca.explained_variance_ratio_)
        n_factors = np.searchsorted(cumulative_var, self.variance_threshold) + 1
        
        # Clamp to reasonable range
        n_factors = max(2, min(n_factors, returns.shape[1] - 1, 8))
        
        print(f"   Auto-selected K={n_factors} factors "
              f"(explains {cumulative_var[n_factors-1]:.1%} variance)")
        
        return n_factors
    
    def transform(self, returns: np.ndarray) -> np.ndarray:
        """
        Project returns onto factor space.
        
        Parameters
        ----------
        returns : np.ndarray
            Log returns (T × N)
            
        Returns
        -------
        factors : np.ndarray
            Factor values (T × K)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if hasattr(returns, 'values'):
            returns = returns.values
        
        return self.pca.transform(returns)
    
    def inverse_transform(self, factors: np.ndarray) -> np.ndarray:
        """
        Reconstruct returns from factor values.
        
        Parameters
        ----------
        factors : np.ndarray
            Factor values (T × K)
            
        Returns
        -------
        returns : np.ndarray
            Reconstructed returns (T × N)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.pca.inverse_transform(factors)
    
    def get_factor_weights(self, portfolio_weights: np.ndarray) -> np.ndarray:
        """
        Compute effective factor exposure for a portfolio.
        
        β = Q_K^T @ w
        
        Parameters
        ----------
        portfolio_weights : np.ndarray
            Asset weights (N,)
            
        Returns
        -------
        factor_weights : np.ndarray
            Factor exposure (K,)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self._result.components @ portfolio_weights
    
    @property
    def result(self) -> Optional[FactorModelResult]:
        """Get fitted result."""
        return self._result


def estimate_factor_volatilities(
    asset_volatilities: np.ndarray,
    factor_loading: np.ndarray
) -> np.ndarray:
    """
    Estimate factor volatilities from asset volatilities.
    
    Uses least-squares fitting:
    σ_assets ≈ |L| @ σ_factors
    
    Parameters
    ----------
    asset_volatilities : np.ndarray
        Asset volatilities (N,)
    factor_loading : np.ndarray
        Factor loading matrix (K × N) or (N × K)
        
    Returns
    -------
    factor_volatilities : np.ndarray
        Estimated factor volatilities (K,)
    """
    # Ensure factor_loading is (K × N)
    if factor_loading.shape[0] > factor_loading.shape[1]:
        factor_loading = factor_loading.T
    
    K = factor_loading.shape[0]
    
    # Solve: |L|^T @ σ_f ≈ σ_assets
    # Using least squares
    L_abs = np.abs(factor_loading.T)  # (N × K)
    factor_vols, _, _, _ = np.linalg.lstsq(L_abs, asset_volatilities, rcond=None)
    
    # Ensure positive
    factor_vols = np.abs(factor_vols)
    
    return factor_vols


if __name__ == '__main__':
    # Test with synthetic data
    np.random.seed(42)
    T, N = 1000, 5
    
    # Generate correlated returns
    true_factors = np.random.randn(T, 2)
    loadings = np.random.randn(N, 2)
    noise = 0.1 * np.random.randn(T, N)
    returns = true_factors @ loadings.T + noise
    
    # Fit model
    model = PCAFactorModel(n_factors=3)
    result = model.fit(returns, risk_free_rate=0.05)
    
    # Transform
    factors = model.transform(returns)
    print(f"\n✅ Factor model test complete")
    print(f"   Factor shape: {factors.shape}")
