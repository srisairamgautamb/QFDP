""" 
Quantum State Preparation Module
=================================

Implements quantum amplitude encoding for marginal distributions and
Gaussian factors using Grover-Rudolph and variational methods.

Key functions:
- `prepare_marginal_distribution`: Generic amplitude encoding
- `prepare_lognormal_asset`: Black-Scholes asset price distribution
- `prepare_gaussian_factor`: Standard normal factor for copula
- `compute_fidelity`: Validate state preparation accuracy
- `estimate_resource_cost`: T-count and depth analysis

Examples
--------
>>> from qfdp_multiasset.state_prep import (
...     prepare_lognormal_asset,
...     prepare_gaussian_factor,
...     compute_fidelity
... )
>>> 
>>> # Prepare AAPL price distribution
>>> S0, r, sigma, T = 150.0, 0.03, 0.25, 1.0
>>> circuit, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits=8)
>>> 
>>> # Prepare correlation factor
>>> factor_circuit = prepare_gaussian_factor(n_qubits=6)
"""

from .grover_rudolph import (
    prepare_marginal_distribution,
    prepare_lognormal_asset,
    prepare_gaussian_factor,
    compute_fidelity,
    estimate_resource_cost
)

# Invertible state prep for k>0 MLQAE (research-grade)
from .invertible_prep import (
    prepare_lognormal_invertible,
    prepare_gaussian_invertible,
    build_grover_operator,
    select_adaptive_k,
    estimate_grover_iterations,
    validate_invertibility,
    compute_rotation_angles_tree,
    build_rotation_tree_circuit
)

__all__ = [
    # Standard state prep (uses initialize - k=0 only)
    'prepare_marginal_distribution',
    'prepare_lognormal_asset', 
    'prepare_gaussian_factor',
    'compute_fidelity',
    'estimate_resource_cost',
    # Invertible state prep (enables k>0 MLQAE)
    'prepare_lognormal_invertible',
    'prepare_gaussian_invertible',
    'build_grover_operator',
    'select_adaptive_k',
    'estimate_grover_iterations',
    'validate_invertibility',
    'compute_rotation_angles_tree',
    'build_rotation_tree_circuit'
]
