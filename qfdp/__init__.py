"""
QFDP: Quantum Fourier Derivative Pricing
=========================================

A quantum computing framework for derivative pricing and portfolio management.

Key Features:
- FB-IQFT: Factor-Based Quantum Fourier pricing (BREAKTHROUGH)
- Sparse copula for multi-asset correlation
- MLQAE with amplitude amplification
- IBM Quantum hardware integration
- Real VaR/CVaR risk metrics

Author: QFDP Research Team
Date: November 2025
"""

__version__ = "1.0.0"

# Core exports
from .core.sparse_copula import FactorDecomposer
from .core.hardware import IBMQuantumRunner

# FB-IQFT exports (the breakthrough)
from .fb_iqft.pricing import factor_based_qfdp

__all__ = [
    'FactorDecomposer',
    'IBMQuantumRunner',
    'factor_based_qfdp',
]
