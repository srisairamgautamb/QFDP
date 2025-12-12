"""
Payoff Oracle for FT-QAE
=========================

Builds the quantum oracle U_payoff that maps factor states to payoff amplitudes:
    |f₁⟩|f₂⟩...|f_K⟩|0⟩ → |f₁⟩|f₂⟩...|f_K⟩(√(1-p)|0⟩ + √p|1⟩)

where p = max(B(f) - K, 0) / B_max is the normalized payoff.

Mathematical Foundation:
------------------------
B(f) = F·exp(√T·Σ_k β_k f_k) = basket value given factors
Π(B) = max(B - K, 0) = call payoff
p = Π(B) / Π_max = normalized payoff ∈ [0,1]

Author: QFDP Research Team
Date: December 3, 2025
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister

from qfdp.ft_qae.arithmetic import (
    build_basket_value_oracle,
    simplified_payoff_amplitude_loading
)


@dataclass
class PayoffOracleConfig:
    """Configuration for payoff oracle construction.
    
    Attributes
    ----------
    n_precision_bits : int
        Fixed-point precision bits for arithmetic
    use_simplified_oracle : bool
        Use simplified demo oracle (True) or full arithmetic (False)
    """
    n_precision_bits: int = 10
    use_simplified_oracle: bool = True  # For NISQ demo


def compute_payoff_max(
    forward_price: float,
    beta_values: np.ndarray,
    maturity: float,
    coverage: float = 4.0
) -> float:
    """
    Compute maximum possible payoff for normalization.
    
    Max payoff occurs when all factors are at maximum value (+coverage·σ).
    
    Parameters
    ----------
    forward_price : float
        Forward basket value F = B₀·e^{rT}
    beta_values : np.ndarray
        Factor exposures
    maturity : float
        Time to maturity
    coverage : float
        Factor coverage in standard deviations
    
    Returns
    -------
    payoff_max : float
        Maximum payoff value
    
    Formula:
    --------
    B_max = F·exp(√T·coverage·Σ|β_k|)
    Π_max = B_max (assuming strike < B_max)
    """
    # Maximum factor contribution (all factors at +coverage)
    max_factor_sum = coverage * np.sum(np.abs(beta_values))
    
    # Maximum basket value
    B_max = forward_price * np.exp(np.sqrt(maturity) * max_factor_sum)
    
    return B_max


def build_payoff_oracle(
    factor_registers: List[QuantumRegister],
    factor_grids: List[np.ndarray],
    beta_values: np.ndarray,
    forward_price: float,
    strike: float,
    maturity: float,
    config: PayoffOracleConfig
) -> Tuple[QuantumCircuit, int]:
    """
    Build complete payoff oracle circuit for FT-QAE.
    
    Oracle Operation:
    -----------------
    |f₁...f_K⟩|0⟩ → |f₁...f_K⟩(√(1-p(f))|0⟩ + √p(f)|1⟩)
    
    where p(f) = max(B(f) - K, 0) / B_max
    
    Parameters
    ----------
    factor_registers : List[QuantumRegister]
        K quantum registers for factors
    factor_grids : List[np.ndarray]
        Grid points for each factor
    beta_values : np.ndarray, shape (K,)
        Factor exposures (weights)
    forward_price : float
        Forward basket value F
    strike : float
        Option strike K
    maturity : float
        Time to maturity T
    config : PayoffOracleConfig
        Oracle configuration
    
    Returns
    -------
    qc : QuantumCircuit
        Payoff oracle circuit
    ancilla_idx : int
        Index of ancilla qubit encoding payoff
    
    Circuit Structure:
    ------------------
    1. Compute weighted sum: S = Σ_k β_k·f_k
    2. Compute basket value: B = F·exp(√T·S)
    3. Compare to strike: flag = (B > K)
    4. Compute payoff: Π = (B - K) if flag else 0
    5. Normalize: p = Π / Π_max
    6. Amplitude encode: |0⟩ → √(1-p)|0⟩ + √p|1⟩
    
    Complexity:
    -----------
    - Qubits: K·n + O(log K) ancillas + 1 output
    - Gates: O(K·n²) for arithmetic
    - Depth: O(K·n)
    
    Examples
    --------
    >>> from qiskit import QuantumRegister
    >>> from qfdp.ft_qae.tensor_state import TensorStateConfig
    >>> 
    >>> # Setup
    >>> config_state = TensorStateConfig(n_qubits_per_factor=4, K_factors=2)
    >>> factor_regs = [QuantumRegister(4, f'f{k}') for k in range(2)]
    >>> grids = [np.linspace(-4, 4, 16) for _ in range(2)]
    >>> beta = np.array([0.6, 0.4])
    >>> 
    >>> # Build oracle
    >>> config_oracle = PayoffOracleConfig(n_precision_bits=8)
    >>> qc, anc_idx = build_payoff_oracle(
    ...     factor_regs, grids, beta, F=100, strike=100, maturity=1.0, config=config_oracle
    ... )
    >>> print(f"Oracle depth: {qc.depth()}")
    """
    K = len(factor_registers)
    n = len(factor_registers[0])
    
    # Create circuit with all registers
    qc = QuantumCircuit()
    for reg in factor_registers:
        qc.add_register(reg)
    
    # Ancilla qubit for payoff amplitude
    ancilla = AncillaRegister(1, 'payoff')
    qc.add_register(ancilla)
    ancilla_idx = qc.num_qubits - 1
    
    if config.use_simplified_oracle:
        # SIMPLIFIED ORACLE for demonstration
        # Uses classical simulation to compute payoff and encode as rotation
        
        # Create auxiliary register for basket value
        basket_register = QuantumRegister(n + 2, 'basket')
        qc.add_register(basket_register)
        
        # Step 1: Build basket value oracle (simplified)
        build_basket_value_oracle(
            qc,
            [list(reg) for reg in factor_registers],
            factor_grids,
            beta_values,
            forward_price,
            maturity,
            list(basket_register),
            n_precision_bits=config.n_precision_bits
        )
        
        # Step 2: Payoff amplitude loading (simplified)
        payoff_max = compute_payoff_max(forward_price, beta_values, maturity)
        
        simplified_payoff_amplitude_loading(
            qc,
            list(basket_register),
            ancilla[0],
            payoff_max,
            strike,
            n_frac_bits=config.n_precision_bits
        )
    
    else:
        # FULL ORACLE (not implemented in this demo version)
        # Would require:
        # - Complete quantum arithmetic library
        # - Proper exponential circuits
        # - Multi-controlled operations
        raise NotImplementedError(
            "Full payoff oracle requires production quantum arithmetic library. "
            "Use config.use_simplified_oracle=True for demo."
        )
    
    return qc, ancilla_idx


__all__ = [
    'PayoffOracleConfig',
    'build_payoff_oracle',
    'compute_payoff_max'
]
