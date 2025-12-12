"""
Sparse Copula Module
====================

Factor model-based correlation encoding for quantum multi-asset pricing.

Key Innovation:
    Reduces quantum correlation encoding complexity from O(N²) to O(N×K)
    via low-rank matrix approximation.

Main Classes:
    - FactorDecomposer: Eigenvalue-based decomposition Σ ≈ L·L^T + D
    - DecompositionMetrics: Quality metrics (variance explained, errors)

Main Functions:
    - generate_synthetic_correlation_matrix: Create test matrices
    - analyze_eigenvalue_decay: Determine optimal K

Example:
    >>> from qfdp_multiasset.sparse_copula import FactorDecomposer
    >>> decomposer = FactorDecomposer()
    >>> L, D, metrics = decomposer.fit(corr_matrix, K=3)
    >>> print(f"Variance explained: {metrics.variance_explained:.1%}")
"""

from .factor_model import (
    FactorDecomposer,
    DecompositionMetrics,
    generate_synthetic_correlation_matrix,
    analyze_eigenvalue_decay
)

from .copula_circuit import (
    encode_sparse_copula,
    encode_sparse_copula_with_decomposition,
    CopulaEncodingMetrics,
    compute_copula_fidelity,
    estimate_copula_resources
)

__all__ = [
    # Factor decomposition (Phase 1)
    'FactorDecomposer',
    'DecompositionMetrics',
    'generate_synthetic_correlation_matrix',
    'analyze_eigenvalue_decay',
    # Quantum encoding (Phase 3 - BREAKTHROUGH)
    'encode_sparse_copula',
    'encode_sparse_copula_with_decomposition',
    'CopulaEncodingMetrics',
    'compute_copula_fidelity',
    'estimate_copula_resources'
]

__version__ = '0.1.0'
