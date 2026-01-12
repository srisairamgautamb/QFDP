"""QAE Training Utilities"""

import numpy as np
from typing import List
from .qae_factor_model import QuantumFactorAutoencoder, QAEResult


def train_qae(
    returns: np.ndarray,
    n_factors: int = 3,
    n_layers: int = 2,
    window: int = 252,
    max_iter: int = 100
) -> QuantumFactorAutoencoder:
    """
    Train QAE on rolling correlation matrices from returns data.
    
    Parameters
    ----------
    returns : np.ndarray
        (T, N) log returns array
    n_factors : int
        Number of latent factors
    n_layers : int
        Variational circuit depth
    window : int
        Rolling window size for correlations
    max_iter : int
        Maximum training iterations
        
    Returns
    -------
    QuantumFactorAutoencoder
        Trained QAE model
    """
    n_assets = returns.shape[1]
    
    correlation_matrices = []
    for t in range(window, len(returns), window // 4):
        window_returns = returns[t-window:t]
        corr = np.corrcoef(window_returns.T)
        correlation_matrices.append(corr)
    
    print(f"QAE Training: {len(correlation_matrices)} correlation matrices")
    print(f"Assets: {n_assets}, Factors: {n_factors}, Layers: {n_layers}")
    
    qae = QuantumFactorAutoencoder(
        n_assets=n_assets,
        n_factors=n_factors,
        n_layers=n_layers
    )
    
    result = qae.train(correlation_matrices, max_iter=max_iter)
    
    print(f"Training complete: {result.training_time:.1f}s")
    print(f"Reconstruction error: {result.reconstruction_error:.6f}")
    
    return qae
