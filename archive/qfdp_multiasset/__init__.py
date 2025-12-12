"""
QFDP Multi-Asset Portfolio Management
======================================

Quantum-enhanced multi-asset derivative pricing and portfolio optimization
using sparse copula correlation encoding.

Main Modules:
    - sparse_copula: Factor model decomposition (O(N²) → O(N×K) breakthrough)
    - state_prep: Quantum state preparation (Grover-Rudolph, variational)
    - iqft: Tensor IQFT for multi-dimensional Fourier transform
    - qsp: Quantum Signal Processing for payoff encoding
    - oracles: Characteristic function and payoff oracles
    - mlqae: Maximum Likelihood Quantum Amplitude Estimation
    - portfolio: Portfolio optimization algorithms
    - analysis: Resource analysis and extrapolation
    - benchmarks: Classical baselines (MC, FFT, nested MC)

Example:
    >>> from qfdp_multiasset.sparse_copula import FactorDecomposer
    >>> decomposer = FactorDecomposer()
    >>> L, D, metrics = decomposer.fit(corr_matrix, K=3)
    >>> print(f"Variance explained: {metrics.variance_explained:.1%}")

Research Gates:
    - GATE 1: Sparse copula fidelity (F ≥ 0.10, Frobenius ≤ 0.5)
    - GATE 2: QSP+MLQAE pricing accuracy (RMSE ≤ 1%)
    - GATE 3: Nested CVA estimation (error ≤ 10%)
"""

__version__ = '0.1.0'
__author__ = 'QFDP Multi-Asset Research Team'

# Import main classes for convenience
from .sparse_copula import FactorDecomposer, DecompositionMetrics

__all__ = [
    'FactorDecomposer',
    'DecompositionMetrics',
]