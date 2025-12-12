"""
Factor-Tensorized Quantum Amplitude Estimation (FT-QAE)
========================================================

Novel quantum algorithm combining:
- Factor model decomposition (Σ = LL^T + D)
- Tensor product state preparation (⊗_k |ψ_k⟩)
- Quantum Amplitude Estimation (QAE/MLAE)

Key Innovation:
--------------
- Exploits factor independence to build tensor product states
- State prep complexity: O(Kn) vs O(2^{Kn}) for general states
- Amplitude estimation for O(1/ε) sampling advantage vs Monte Carlo O(1/ε²)

Mathematical Foundation:
-----------------------
Theorem 1: Factor independence ⇒ |Ψ⟩ = ⊗_{k=1}^K |ψ_k⟩
Corollary 1: State preparation requires O(Kn) gates (exponential speedup)

Author: QFDP Unified Research Team
Date: December 3, 2025
"""

from qfdp.ft_qae.tensor_state import (
    prepare_gaussian_factor_state,
    prepare_tensor_product_state,
    TensorStateConfig
)

from qfdp.ft_qae.payoff_oracle import (
    build_payoff_oracle,
    PayoffOracleConfig
)

from qfdp.ft_qae.qae import (
    maximum_likelihood_qae,
    iterative_qae,
    MLQAEResult
)

from qfdp.ft_qae.pricing import (
    ft_qae_price_option,
    FTQAEPricingResult
)

__all__ = [
    # State preparation
    'prepare_gaussian_factor_state',
    'prepare_tensor_product_state',
    'TensorStateConfig',
    
    # Payoff oracle
    'build_payoff_oracle',
    'PayoffOracleConfig',
    
    # QAE algorithms
    'maximum_likelihood_qae',
    'iterative_qae',
    'MLQAEResult',
    
    # Main pricing interface
    'ft_qae_price_option',
    'FTQAEPricingResult'
]
