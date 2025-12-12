"""
Quantum Arithmetic for FT-QAE Payoff Oracle
============================================

Implements quantum arithmetic operations needed for basket payoff computation:
- Fixed-point weighted sum: Σ_k β_k f_k
- Quantum exponential: exp(x) via piecewise linear approximation
- Quantum comparison: B > K
- Amplitude loading: controlled rotation based on payoff

All operations use fixed-point arithmetic for NISQ-friendly implementation.

Author: QFDP Research Team
Date: December 3, 2025
"""

import numpy as np
from typing import List, Tuple, Optional
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import RYGate


def float_to_fixed_point(
    value: float,
    n_bits: int,
    n_frac_bits: int
) -> int:
    """
    Convert floating-point value to fixed-point integer representation.
    
    Fixed-point format: Q(n-n_frac).n_frac
    - n_bits total bits
    - n_frac_bits fractional bits
    - (n_bits - n_frac_bits) integer bits
    
    Value range: [-2^(n-n_frac-1), 2^(n-n_frac-1) - 2^(-n_frac)]
    
    Parameters
    ----------
    value : float
        Value to convert
    n_bits : int
        Total number of bits
    n_frac_bits : int
        Number of fractional bits
    
    Returns
    -------
    fixed_point : int
        Fixed-point representation
    
    Examples
    --------
    >>> # 8-bit fixed-point with 4 fractional bits: Q4.4
    >>> x = float_to_fixed_point(3.5, n_bits=8, n_frac_bits=4)
    >>> print(x)  # 3.5 * 2^4 = 56
    56
    """
    scale_factor = 2 ** n_frac_bits
    fixed = int(round(value * scale_factor))
    
    # Check for overflow
    max_val = 2 ** (n_bits - 1) - 1
    min_val = -2 ** (n_bits - 1)
    
    if fixed > max_val or fixed < min_val:
        raise ValueError(
            f"Value {value} overflows {n_bits}-bit fixed-point with {n_frac_bits} fractional bits"
        )
    
    return fixed


def fixed_point_to_float(
    fixed_point: int,
    n_frac_bits: int
) -> float:
    """Convert fixed-point integer to floating-point value."""
    scale_factor = 2 ** n_frac_bits
    return fixed_point / scale_factor


def quantum_adder(
    qc: QuantumCircuit,
    a_register: List,
    b_register: List,
    carry_in: Optional[int] = None
) -> None:
    """
    Add two quantum registers: b += a (in-place addition on b).
    
    Implements ripple-carry adder using controlled-X gates.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit
    a_register : List[Qubit]
        Input register A (not modified)
    b_register : List[Qubit]
        Input/output register B (B := B + A)
    carry_in : Qubit, optional
        Carry-in qubit
    
    Complexity:
    -----------
    - Gates: O(n) where n = number of bits
    - Depth: O(n) (sequential carry propagation)
    
    Note: This is a simplified ripple-carry adder. For production,
    consider Draper adder or quantum Fourier transform adder for better depth.
    """
    n = len(a_register)
    assert len(b_register) >= n, "b_register must be at least as long as a_register"
    
    # Simple ripple-carry adder (can be optimized)
    # For each bit position:
    # 1. Compute carry
    # 2. Compute sum
    # 3. Propagate carry
    
    # This is a placeholder - full implementation requires carry logic
    # For simplicity in demo, use half-adder approximation (no carry)
    for i in range(n):
        qc.cx(a_register[i], b_register[i])


def quantum_weighted_sum_simple(
    qc: QuantumCircuit,
    factor_registers: List[List],
    beta_values: np.ndarray,
    output_register: List,
    n_frac_bits: int = 8
) -> None:
    """
    Compute weighted sum: output = Σ_k β_k × factor_k
    
    Simplified implementation using controlled rotations to approximate
    weighted sum. For production, use full quantum arithmetic.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit
    factor_registers : List[List[Qubit]]
        K factor registers, each with n qubits
    beta_values : np.ndarray, shape (K,)
        Factor weights (normalized)
    output_register : List[Qubit]
        Output register for weighted sum
    n_frac_bits : int
        Number of fractional bits in fixed-point
    
    Algorithm:
    ----------
    For each factor k:
        1. Convert β_k to fixed-point
        2. Multiply factor_k by β_k using shift-and-add
        3. Add to accumulator (output_register)
    
    Complexity:
    -----------
    - Gates: O(K × n × n_frac_bits)
    - Depth: O(K × n) with parallel optimization
    """
    K = len(factor_registers)
    n = len(factor_registers[0])
    
    # Initialize output to zero (reset)
    for qubit in output_register:
        qc.reset(qubit)
    
    # For each factor
    for k in range(K):
        beta_k = beta_values[k]
        
        # Convert to fixed-point
        beta_fixed = float_to_fixed_point(abs(beta_k), n_bits=n+2, n_frac_bits=n_frac_bits)
        
        # Extract binary representation
        beta_bits = [(beta_fixed >> i) & 1 for i in range(n)]
        
        # Multiply factor_k by beta_k using shift-and-add
        for bit_pos, bit_val in enumerate(beta_bits):
            if bit_val == 1:
                # Add factor_k << bit_pos to output
                shift = bit_pos
                for i in range(min(n, len(output_register) - shift)):
                    if i + shift < len(output_register):
                        qc.cx(factor_registers[k][i], output_register[i + shift])


def piecewise_linear_exp(
    x: float,
    n_pieces: int = 8
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Compute piecewise linear approximation of exp(x).
    
    Divides domain into n_pieces segments and fits linear functions
    f_i(x) = a_i + b_i·x for each segment.
    
    Parameters
    ----------
    x : float
        Input value (used to determine domain)
    n_pieces : int
        Number of linear pieces
    
    Returns
    -------
    breakpoints : List[Tuple[float, float]]
        (x_min, x_max) for each piece
    coefficients : List[Tuple[float, float]]
        (a, b) coefficients for each piece: f(x) = a + b·x
    
    Examples
    --------
    >>> bp, coef = piecewise_linear_exp(0.5, n_pieces=4)
    >>> print(f"Number of pieces: {len(coef)}")
    Number of pieces: 4
    """
    # Domain: [-4, 4] typically for normalized factors
    x_min, x_max = -4.0, 4.0
    x_values = np.linspace(x_min, x_max, n_pieces + 1)
    
    breakpoints = []
    coefficients = []
    
    for i in range(n_pieces):
        x0 = x_values[i]
        x1 = x_values[i + 1]
        
        # Compute exp at endpoints
        y0 = np.exp(x0)
        y1 = np.exp(x1)
        
        # Linear fit: y = a + b·x
        # y0 = a + b·x0
        # y1 = a + b·x1
        # ⇒ b = (y1 - y0) / (x1 - x0)
        # ⇒ a = y0 - b·x0
        
        b = (y1 - y0) / (x1 - x0)
        a = y0 - b * x0
        
        breakpoints.append((x0, x1))
        coefficients.append((a, b))
    
    return breakpoints, coefficients


def quantum_comparison(
    qc: QuantumCircuit,
    a_register: List,
    b_register: List,
    result_qubit,
    comparison: str = 'gt'
) -> None:
    """
    Quantum comparison: result = (A > B) or (A >= B) or (A < B) or (A <= B).
    
    Implements comparison using quantum subtraction and sign check.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit
    a_register : List[Qubit]
        First register (A)
    b_register : List[Qubit]
        Second register (B)
    result_qubit : Qubit
        Output qubit: |1⟩ if comparison is true, |0⟩ otherwise
    comparison : str
        Type of comparison: 'gt' (>), 'gte' (>=), 'lt' (<), 'lte' (<=)
    
    Algorithm:
    ----------
    1. Compute A - B using quantum subtractor
    2. Check sign bit of result
    3. Set result_qubit based on sign
    
    Complexity:
    -----------
    - Gates: O(n) for subtraction + O(1) for sign check
    - Depth: O(n)
    
    Note: Simplified implementation for demonstration.
    Production version should use optimized comparator circuits.
    """
    n = len(a_register)
    
    # Simplified: use multi-controlled gates based on most significant bits
    # For A > B: check if MSB(A) > MSB(B) (approximate)
    
    if comparison == 'gt':
        # A > B approximately: check top bits
        # This is a placeholder - full implementation requires subtraction
        qc.cx(a_register[-1], result_qubit)  # Simplified
    elif comparison == 'gte':
        qc.cx(a_register[-1], result_qubit)
        qc.x(result_qubit)  # Invert for >=
    elif comparison == 'lt':
        qc.cx(b_register[-1], result_qubit)
    else:  # lte
        qc.cx(b_register[-1], result_qubit)
        qc.x(result_qubit)


def compute_payoff_rotation_angle(
    payoff_normalized: float
) -> float:
    """
    Compute rotation angle for amplitude encoding of payoff.
    
    For amplitude encoding: |0⟩ → √(1-p)|0⟩ + √p|1⟩
    Rotation angle: θ = 2·arcsin(√p)
    
    Parameters
    ----------
    payoff_normalized : float
        Normalized payoff value in [0, 1]
    
    Returns
    -------
    theta : float
        Rotation angle for RY gate
    
    Mathematical Foundation:
    ------------------------
    RY(θ) |0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
    
    To achieve amplitude √p for |1⟩:
        sin²(θ/2) = p
        ⇒ θ = 2·arcsin(√p)
    
    Examples
    --------
    >>> theta = compute_payoff_rotation_angle(0.25)
    >>> import numpy as np
    >>> print(f"θ = {theta:.4f} rad = {np.degrees(theta):.2f}°")
    """
    # Clip to [0, 1] for numerical safety
    p = np.clip(payoff_normalized, 0, 1)
    
    # Compute rotation angle
    theta = 2.0 * np.arcsin(np.sqrt(p))
    
    return theta


def build_basket_value_oracle(
    qc: QuantumCircuit,
    factor_registers: List[List],
    factor_grids: List[np.ndarray],
    beta_values: np.ndarray,
    forward_price: float,
    maturity: float,
    output_register: List,
    n_precision_bits: int = 10
) -> None:
    """
    Build oracle to compute basket value B(f) = F·exp(√T·Σ_k β_k f_k).
    
    This is a SIMPLIFIED oracle for demonstration. Production implementation
    should use:
    - Proper quantum arithmetic (QFT-based multipliers, Draper adder)
    - Optimized exponential (CORDIC or polynomial approximation)
    - Error-corrected operations for precision
    
    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit
    factor_registers : List[List[Qubit]]
        Factor quantum registers
    factor_grids : List[np.ndarray]
        Grid values for each factor
    beta_values : np.ndarray
        Factor exposures (weights)
    forward_price : float
        Forward basket value F = B₀·e^{rT}
    maturity : float
        Time to maturity T
    output_register : List[Qubit]
        Output register for basket value
    n_precision_bits : int
        Number of bits for fixed-point precision
    
    Algorithm:
    ----------
    1. Compute weighted sum: S = Σ_k β_k·f_k
    2. Scale by √T: S_scaled = √T·S
    3. Compute exp(S_scaled) using piecewise linear
    4. Multiply by F: B = F·exp(S_scaled)
    5. Encode result in output_register
    
    Complexity:
    -----------
    - Gates: O(K·n²) for weighted sum + O(n) for exp
    - Depth: O(K·n) + O(log n) for optimized implementation
    """
    K = len(factor_registers)
    n = len(factor_registers[0])
    
    # Step 1: Compute weighted sum (simplified)
    # In practice, use full quantum arithmetic
    sum_register = QuantumRegister(n + int(np.ceil(np.log2(K))), 'sum')
    qc.add_register(sum_register)
    
    quantum_weighted_sum_simple(
        qc,
        factor_registers,
        beta_values,
        list(sum_register),
        n_frac_bits=n_precision_bits
    )
    
    # Step 2-4: Exponential and scaling
    # Simplified: copy sum to output (placeholder for full exp implementation)
    for i in range(min(len(sum_register), len(output_register))):
        qc.cx(sum_register[i], output_register[i])


def simplified_payoff_amplitude_loading(
    qc: QuantumCircuit,
    payoff_register: List,
    ancilla_qubit,
    payoff_max: float,
    strike: float,
    n_frac_bits: int = 10
) -> None:
    """
    Load payoff amplitude into ancilla qubit via controlled rotation.
    
    SIMPLIFIED VERSION for demonstration. Production should use:
    - Exact arithmetic for payoff computation
    - Proper normalization
    - Multi-controlled rotations based on full payoff value
    
    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit
    payoff_register : List[Qubit]
        Register holding basket value
    ancilla_qubit : Qubit
        Ancilla qubit for amplitude encoding
    payoff_max : float
        Maximum possible payoff (for normalization)
    strike : float
        Strike price K
    n_frac_bits : int
        Fixed-point fractional bits
    
    Algorithm:
    ----------
    1. Compare basket value to strike: flag = (B > K)
    2. Compute normalized payoff: p = (B - K) / payoff_max
    3. Rotation angle: θ = 2·arcsin(√p)
    4. Controlled rotation: if flag, apply RY(θ) to ancilla
    
    Result:
    -------
    |ancilla⟩ = √(1 - p̃) |0⟩ + √p̃ |1⟩
    where p̃ = max(B - K, 0) / payoff_max
    """
    # Placeholder: simplified version
    # Use multi-controlled rotation based on payoff_register bits
    
    # Approximate: rotate ancilla based on most significant bits
    # This demonstrates the concept - production needs full implementation
    
    # Rotation angle (simplified: use fixed angle for demo)
    # In practice, compute based on actual payoff value
    theta_example = np.pi / 4  # 45° ≈ 50% payoff
    
    # Apply controlled rotation (control = payoff_register top bit)
    if len(payoff_register) > 0:
        qc.cry(theta_example, payoff_register[-1], ancilla_qubit)


# Export key functions
__all__ = [
    'float_to_fixed_point',
    'fixed_point_to_float',
    'quantum_weighted_sum_simple',
    'piecewise_linear_exp',
    'quantum_comparison',
    'compute_payoff_rotation_angle',
    'build_basket_value_oracle',
    'simplified_payoff_amplitude_loading'
]
