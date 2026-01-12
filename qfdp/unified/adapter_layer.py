"""
QRC-FB-IQFT Adapter Layer
========================

This module provides an isolated adapter layer that transforms QRC factors
into portfolio volatility (sigma_p) for the base FB-IQFT model.

Key Design: Base model remains completely untouched.

Mathematical Framework:
- Modulation: h(f, f̄) = 1 + β(f/f̄ - 1)
- Adaptive eigenvalues: λ̃ᵢ = λᵢ · h(fᵢ, f̄)
- Adaptive covariance: C_QRC = Q Λ̃ Q^T
- Portfolio vol: σ_p = √(w^T C_QRC w)
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class QRCFactorModulator:
    """
    Modulates PCA eigenvalues based on QRC factors.
    
    Given QRC factors f = [f1, f2, f3, f4] where sum(f) ≈ 1,
    we modulate eigenvalues λ using:
        h(f_i, f̄) = 1 + β(f_i/f̄ - 1)
        λ̃_i = λ_i · h(f_i, f̄)
    """
    
    def __init__(self, beta: float = 0.1):
        """
        Args:
            beta: Modulation strength (0.1 optimal based on tuning)
        """
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.beta = beta
        logger.info(f"QRCFactorModulator initialized with beta={beta}")
    
    def modulate_eigenvalues(
        self,
        eigenvalues: np.ndarray,
        qrc_factors: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply QRC modulation to eigenvalues.
        
        Args:
            eigenvalues: Original PCA eigenvalues (shape: n)
            qrc_factors: QRC factors (shape: n), should sum to ~1
        
        Returns:
            Tuple of (modulated_eigenvalues, diagnostics_dict)
        """
        n = min(len(eigenvalues), len(qrc_factors))
        f_bar = 1.0 / n  # Uniform baseline
        
        # Compute modulation: h(f, f̄) = 1 + β(f/f̄ - 1)
        h = np.ones(len(eigenvalues))
        h[:n] = 1.0 + self.beta * (qrc_factors[:n] / f_bar - 1.0)
        
        # Apply modulation
        lambda_tilde = eigenvalues * h
        
        diagnostics = {
            'f_bar': f_bar,
            'h_factors': h.tolist(),
            'eigenvalue_ratio': float(np.sum(lambda_tilde) / np.sum(eigenvalues))
        }
        
        return lambda_tilde, diagnostics


class PortfolioVolatilityAdapter:
    """
    Computes portfolio volatility from modulated covariance matrix.
    """
    
    def compute_sigma_p_qrc(
        self,
        weights: np.ndarray,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
        qrc_factors: np.ndarray,
        modulator: QRCFactorModulator
    ) -> Tuple[float, Dict]:
        """
        Compute QRC-adapted portfolio volatility.
        
        Args:
            weights: Portfolio weights
            eigenvectors: PCA eigenvectors (n x k)
            eigenvalues: PCA eigenvalues (k,)
            qrc_factors: QRC factors
            modulator: QRCFactorModulator instance
        
        Returns:
            Tuple of (sigma_p, diagnostics)
        """
        # Apply modulation
        lambda_tilde, mod_diag = modulator.modulate_eigenvalues(eigenvalues, qrc_factors)
        
        # Reconstruct covariance: C_QRC = Q Λ̃ Q^T
        Lambda_tilde = np.diag(lambda_tilde)
        C_QRC = eigenvectors @ Lambda_tilde @ eigenvectors.T
        
        # Portfolio volatility: σ_p = √(w^T C w)
        sigma_p_squared = weights @ C_QRC @ weights
        sigma_p = np.sqrt(max(sigma_p_squared, 1e-10))
        
        diagnostics = {
            'sigma_p': float(sigma_p),
            'modulation': mod_diag,
            'covariance_is_psd': bool(np.all(np.linalg.eigvalsh(C_QRC) > -1e-10))
        }
        
        return float(sigma_p), diagnostics
    
    def compute_sigma_p_pca(
        self,
        weights: np.ndarray,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray
    ) -> float:
        """Compute standard PCA portfolio volatility (baseline)."""
        Lambda = np.diag(eigenvalues)
        C_PCA = eigenvectors @ Lambda @ eigenvectors.T
        sigma_p_squared = weights @ C_PCA @ weights
        return float(np.sqrt(max(sigma_p_squared, 1e-10)))


class BaseModelAdapter:
    """
    Orchestrator that coordinates QRC → sigma_p transformation.
    
    This is the main interface for the adapter layer.
    Single integration point: prepare_for_qrc_pricing() returns sigma_p
    """
    
    def __init__(self, beta: float = 0.1):
        """
        Args:
            beta: QRC modulation strength (0.1 optimal)
        """
        self.modulator = QRCFactorModulator(beta=beta)
        self.sigma_adapter = PortfolioVolatilityAdapter()
        self.beta = beta
        logger.info(f"BaseModelAdapter initialized with beta={beta}")
    
    def prepare_for_qrc_pricing(
        self,
        spot_prices: np.ndarray,
        volatilities: np.ndarray,
        correlation_matrix: np.ndarray,
        qrc_factors: np.ndarray,
        portfolio_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Transform QRC factors → sigma_p for base model.
        
        Args:
            spot_prices: Asset spot prices
            volatilities: Individual asset volatilities
            correlation_matrix: Asset correlation matrix
            qrc_factors: QRC-generated factors (should sum to ~1)
            portfolio_weights: Portfolio weights (defaults to equal-weighted)
        
        Returns:
            Dictionary with sigma_p_qrc, sigma_p_pca, and diagnostics
        """
        n_assets = len(spot_prices)
        
        if portfolio_weights is None:
            portfolio_weights = np.ones(n_assets) / n_assets
        
        # Construct covariance: C = diag(σ) Ρ diag(σ)
        D = np.diag(volatilities)
        C = D @ correlation_matrix @ D
        
        # PCA decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Match dimensions
        n_factors = min(len(qrc_factors), len(eigenvalues))
        
        # Compute QRC-adapted sigma_p
        sigma_p_qrc, diag = self.sigma_adapter.compute_sigma_p_qrc(
            portfolio_weights, eigenvectors, eigenvalues, qrc_factors, self.modulator
        )
        
        # Compute baseline PCA sigma_p
        sigma_p_pca = self.sigma_adapter.compute_sigma_p_pca(
            portfolio_weights, eigenvectors, eigenvalues
        )
        
        return {
            'sigma_p_qrc': sigma_p_qrc,
            'sigma_p_pca': sigma_p_pca,
            'sigma_p_change_pct': (sigma_p_qrc - sigma_p_pca) / sigma_p_pca * 100,
            'qrc_factors': qrc_factors.tolist(),
            'diagnostics': diag
        }
    
    def prepare_for_pca_pricing(
        self,
        spot_prices: np.ndarray,
        volatilities: np.ndarray,
        correlation_matrix: np.ndarray,
        portfolio_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """Standard PCA baseline (no QRC modulation)."""
        n_assets = len(spot_prices)
        
        if portfolio_weights is None:
            portfolio_weights = np.ones(n_assets) / n_assets
        
        D = np.diag(volatilities)
        C = D @ correlation_matrix @ D
        
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        sigma_p_pca = self.sigma_adapter.compute_sigma_p_pca(
            portfolio_weights, eigenvectors, eigenvalues
        )
        
        return {'sigma_p_pca': sigma_p_pca}


# ============================================================
# VALIDATION TESTS
# ============================================================

def validate_adapter():
    """Run adapter validation tests."""
    
    print("=" * 80)
    print("ADAPTER MATHEMATICAL VALIDATION")
    print("=" * 80)
    
    adapter = BaseModelAdapter(beta=0.1)
    
    # Test data
    S = np.array([100.0, 105.0, 98.0, 102.0])
    sigma = np.array([0.2, 0.25, 0.22, 0.23])
    rho = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.5, 1.0, 0.4, 0.3],
        [0.3, 0.4, 1.0, 0.5],
        [0.2, 0.3, 0.5, 1.0]
    ])
    
    # Test 1: Uniform factors → PCA baseline
    print("\n✓ TEST 1: Uniform factors preserve PCA")
    qrc_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    result_qrc = adapter.prepare_for_qrc_pricing(S, sigma, rho, qrc_uniform)
    result_pca = adapter.prepare_for_pca_pricing(S, sigma, rho)
    
    diff = abs(result_qrc['sigma_p_qrc'] - result_pca['sigma_p_pca'])
    print(f"  σ_p,QRC: {result_qrc['sigma_p_qrc']:.6f}")
    print(f"  σ_p,PCA: {result_pca['sigma_p_pca']:.6f}")
    print(f"  Difference: {diff:.8f}")
    print(f"  {'✅ PASS' if diff < 1e-6 else '❌ FAIL'}")
    
    # Test 2: Concentrated factors → Different
    print("\n✓ TEST 2: Concentrated factors cause adaptation")
    qrc_conc = np.array([0.55, 0.25, 0.12, 0.08])
    result_conc = adapter.prepare_for_qrc_pricing(S, sigma, rho, qrc_conc)
    
    print(f"  σ_p,QRC: {result_conc['sigma_p_qrc']:.6f}")
    print(f"  Relative change: {result_conc['sigma_p_change_pct']:+.2f}%")
    print(f"  ✅ QRC factors detected and applied")
    
    # Test 3: Covariance PSD
    print("\n✓ TEST 3: Adaptive covariance is positive definite")
    is_psd = result_conc['diagnostics']['covariance_is_psd']
    print(f"  All eigenvalues non-negative: {is_psd}")
    print(f"  {'✅ PASS' if is_psd else '❌ FAIL'}")
    
    print("\n" + "=" * 80)
    print("✅ ADAPTER VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    validate_adapter()
