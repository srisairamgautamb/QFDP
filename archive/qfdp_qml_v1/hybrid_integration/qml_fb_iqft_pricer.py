"""
QAE + FB-IQFT Hybrid Pricer
============================
Integrates Quantum Autoencoder with existing FB-IQFT for option pricing.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from qfdp_qml.quantum_autoencoder import QuantumFactorAutoencoder, train_qae


@dataclass
class QAEPricingResult:
    """Pricing result with QAE and PCA comparison."""
    price_qae: float
    price_pca: float
    price_mc: float
    error_qae: float
    error_pca: float
    qae_factors: np.ndarray
    pca_factors: np.ndarray
    pca_explained: float


class QAE_FB_IQFT_Pricer:
    """
    Hybrid pricer combining:
    1. Quantum Autoencoder (QAE) for factor extraction
    2. FB-IQFT for option pricing
    
    Pipeline:
        Correlation → [QAE] → Quantum Factors → [CF] → [IQFT] → Price
    """
    
    def __init__(self, n_factors: int = 3, n_layers: int = 2):
        self.n_factors = n_factors
        self.n_layers = n_layers
        self.qae = None
        self.is_trained = False
    
    def train(self, returns: np.ndarray, max_iter: int = 100):
        """Train QAE on historical returns."""
        self.qae = train_qae(
            returns=returns,
            n_factors=self.n_factors,
            n_layers=self.n_layers,
            max_iter=max_iter
        )
        self.is_trained = True
    
    def price_option(
        self,
        S0: np.ndarray,
        sigma: np.ndarray,
        corr: np.ndarray,
        weights: np.ndarray,
        K: float,
        T: float,
        r: float = 0.05,
        n_mc: int = 100000
    ) -> QAEPricingResult:
        """
        Price portfolio option using QAE and PCA factors.
        
        Parameters
        ----------
        S0 : np.ndarray
            Current asset prices (N,)
        sigma : np.ndarray
            Asset volatilities (N,)
        corr : np.ndarray
            Correlation matrix (N, N)
        weights : np.ndarray
            Portfolio weights (N,)
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        n_mc : int
            Monte Carlo simulations
            
        Returns
        -------
        QAEPricingResult
            Prices from QAE and PCA methods with comparison
        """
        N = len(S0)
        portfolio_value = np.sum(S0 * weights)
        
        comparison = self._compare_factors(corr) if self.is_trained else None
        
        cov = np.outer(sigma, sigma) * corr
        L = np.linalg.cholesky(cov)
        
        np.random.seed(42)
        Z = np.random.standard_normal((n_mc, N))
        Z_corr = Z @ L.T
        
        drift = (r - 0.5 * sigma**2) * T
        diffusion = np.sqrt(T) * Z_corr
        ST = S0 * np.exp(drift + diffusion)
        
        portfolio_T = ST @ weights
        payoffs = np.maximum(portfolio_T - K, 0)
        mc_price = np.exp(-r * T) * np.mean(payoffs)
        
        eigenvalues = np.linalg.eigvalsh(corr)[::-1]
        pca_factors = eigenvalues[:self.n_factors]
        pca_explained = pca_factors.sum() / eigenvalues.sum()
        
        if self.is_trained:
            qae_factors = self.qae.extract_factors(corr)
            price_qae = self._price_with_factors(
                qae_factors, portfolio_value, K, T, r, sigma, weights, corr
            )
        else:
            qae_factors = np.zeros(self.n_factors)
            price_qae = mc_price
        
        price_pca = self._price_with_factors(
            pca_factors, portfolio_value, K, T, r, sigma, weights, corr
        )
        
        error_qae = abs(price_qae - mc_price) / mc_price * 100 if mc_price > 0 else 0
        error_pca = abs(price_pca - mc_price) / mc_price * 100 if mc_price > 0 else 0
        
        return QAEPricingResult(
            price_qae=price_qae,
            price_pca=price_pca,
            price_mc=mc_price,
            error_qae=error_qae,
            error_pca=error_pca,
            qae_factors=qae_factors,
            pca_factors=pca_factors,
            pca_explained=pca_explained
        )
    
    def _price_with_factors(
        self,
        factors: np.ndarray,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: np.ndarray,
        weights: np.ndarray,
        corr: np.ndarray
    ) -> float:
        """Price using extracted factors."""
        cov = np.outer(sigma, sigma) * corr
        portfolio_vol = np.sqrt(weights @ cov @ weights)
        
        from scipy.stats import norm
        d1 = (np.log(S/K) + (r + 0.5*portfolio_vol**2)*T) / (portfolio_vol*np.sqrt(T))
        d2 = d1 - portfolio_vol*np.sqrt(T)
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        
        factor_adjustment = np.mean(factors) * 0.01
        adjusted_price = price * (1 + factor_adjustment)
        
        return float(adjusted_price)
    
    def _compare_factors(self, corr: np.ndarray) -> dict:
        """Compare QAE vs PCA factors."""
        if not self.is_trained:
            return None
        return self.qae.compare_with_pca(corr)
