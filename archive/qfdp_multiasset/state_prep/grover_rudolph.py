"""
Grover-Rudolph Quantum State Preparation
=========================================

Implements amplitude encoding for asset marginal distributions and Gaussian factors.

The Grover-Rudolph algorithm (arXiv:quant-ph/0208112) enables exact amplitude
encoding of arbitrary probability distributions into quantum states via recursive
decomposition into controlled-Ry rotations.

Key Features:
- Exact amplitude loading (deterministic, no training)
- O(2^n) gate complexity for n qubits
- Support for log-normal asset price distributions
- Gaussian factor state preparation for correlation encoding

Author: QFDP Multi-Asset Research Team
Date: November 2025
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from scipy.stats import norm, lognorm
import warnings


def prepare_marginal_distribution(
    prices: np.ndarray,
    probabilities: np.ndarray,
    n_qubits: int,
    validate: bool = True
) -> QuantumCircuit:
    """
    Prepare quantum state encoding asset price distribution.
    
    Creates quantum state |ψ⟩ = Σ_i √p_i |i⟩ where p_i are discretized
    probabilities over price grid.
    
    Parameters
    ----------
    prices : np.ndarray, shape (M,)
        Price grid (not directly used, but for reference)
    probabilities : np.ndarray, shape (M,)
        Probability distribution over prices (must sum to 1)
    n_qubits : int
        Number of qubits (N = 2^n grid points)
    validate : bool, default=True
        Validate probability distribution properties
    
    Returns
    -------
    circuit : QuantumCircuit
        State preparation circuit
    
    Examples
    --------
    >>> import numpy as np
    >>> from qfdp_multiasset.state_prep import prepare_marginal_distribution
    >>> 
    >>> # Log-normal distribution for asset price
    >>> S0, sigma, T = 100, 0.2, 1.0
    >>> n_qubits = 8  # 256 price points
    >>> N = 2**n_qubits
    >>> 
    >>> # Discretize price range (3 std devs)
    >>> S_min = S0 * np.exp(-3*sigma*np.sqrt(T))
    >>> S_max = S0 * np.exp(+3*sigma*np.sqrt(T))
    >>> prices = np.linspace(S_min, S_max, N)
    >>> 
    >>> # Log-normal pdf
    >>> returns = np.log(prices / S0)
    >>> pdf = norm.pdf(returns, loc=-0.5*sigma**2*T, scale=sigma*np.sqrt(T))
    >>> probabilities = pdf / pdf.sum()
    >>> 
    >>> # Prepare state
    >>> circuit = prepare_marginal_distribution(prices, probabilities, n_qubits)
    >>> print(f"Circuit depth: {circuit.depth()}")
    """
    if validate:
        # Check normalization
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
            warnings.warn(f"Probabilities sum to {probabilities.sum():.6f}, not 1.0. Normalizing.")
            probabilities = probabilities / probabilities.sum()
        
        # Check non-negative
        if np.any(probabilities < 0):
            raise ValueError("Probabilities must be non-negative")
    
    # Convert probabilities to amplitudes
    amplitudes = np.sqrt(probabilities)
    
    # Pad to power of 2
    N = 2**n_qubits
    if len(amplitudes) < N:
        padded = np.zeros(N)
        padded[:len(amplitudes)] = amplitudes
        amplitudes = padded
    elif len(amplitudes) > N:
        amplitudes = amplitudes[:N]
    
    # Renormalize after padding/truncation
    norm_factor = np.linalg.norm(amplitudes)
    if norm_factor > 0:
        amplitudes = amplitudes / norm_factor
    
    # Build circuit using Qiskit's initialize (uses isometry decomposition)
    # This is more efficient than manual Grover-Rudolph for simulation
    circuit = QuantumCircuit(n_qubits)
    circuit.initialize(amplitudes, range(n_qubits))
    
    return circuit


def _grover_rudolph_recursive(
    amplitudes: np.ndarray,
    n_qubits: int,
    qubit_offset: int = 0
) -> QuantumCircuit:
    """
    Recursive Grover-Rudolph decomposition.
    
    At each level, split the amplitude vector in half and apply controlled
    rotations to distribute probability correctly.
    
    Parameters
    ----------
    amplitudes : np.ndarray
        Target amplitudes (must be normalized)
    n_qubits : int
        Number of qubits
    qubit_offset : int
        Qubit index offset for recursion
    
    Returns
    -------
    circuit : QuantumCircuit
        State preparation circuit
    """
    N = len(amplitudes)
    assert N == 2**n_qubits, f"Amplitudes length {N} must equal 2^{n_qubits}"
    
    circuit = QuantumCircuit(n_qubits)
    
    if n_qubits == 0:
        # Base case: no qubits
        return circuit
    
    if n_qubits == 1:
        # Base case: single qubit rotation
        # |0⟩ → α|0⟩ + β|1⟩
        prob_0 = np.abs(amplitudes[0])**2
        prob_1 = np.abs(amplitudes[1])**2
        total = prob_0 + prob_1
        
        if total > 1e-12:
            theta = 2 * np.arcsin(np.sqrt(prob_1 / total))
            circuit.ry(theta, 0)
        
        return circuit
    
    # Split amplitudes into left and right halves
    mid = N // 2
    left_amps = amplitudes[:mid]
    right_amps = amplitudes[mid:]
    
    # Compute total probability in each half
    left_prob = np.sum(np.abs(left_amps)**2)
    right_prob = np.sum(np.abs(right_amps)**2)
    total_prob = left_prob + right_prob
    
    if total_prob < 1e-12:
        # No amplitude in this branch, return empty circuit
        return circuit
    
    # Compute rotation angle for first qubit
    # We want: |0⟩ → √(left_prob/total_prob) |0⟩ + √(right_prob/total_prob) |1⟩
    theta = 2 * np.arcsin(np.sqrt(right_prob / total_prob))
    
    # Apply rotation on first qubit (most significant bit)
    circuit.ry(theta, 0)
    
    # Normalize left and right amplitudes
    if left_prob > 1e-12:
        left_normalized = left_amps / np.sqrt(left_prob)
    else:
        left_normalized = np.zeros_like(left_amps)
    
    if right_prob > 1e-12:
        right_normalized = right_amps / np.sqrt(right_prob)
    else:
        right_normalized = np.zeros_like(right_amps)
    
    # Build subcircuits for remaining qubits
    left_circuit = _grover_rudolph_recursive(left_normalized, n_qubits - 1)
    right_circuit = _grover_rudolph_recursive(right_normalized, n_qubits - 1)
    
    # Apply left subcircuit controlled on qubit 0 = |0⟩ (C-NOT controlled gates)
    # Add controlled versions of each gate in the subcircuit
    for instruction in left_circuit.data:
        gate = instruction.operation
        target_qubits = [left_circuit.qubits.index(q) + 1 for q in instruction.qubits]
        
        # Apply gate controlled on qubit 0 being |0⟩
        # Use X-controlled-X pattern: X on control, controlled gate, X on control
        circuit.x(0)
        if hasattr(gate, 'to_instruction'):
            ctrl_gate = gate.control(1)
            circuit.append(ctrl_gate, [0] + target_qubits)
        else:
            ctrl_gate = gate.control(1)
            circuit.append(ctrl_gate, [0] + target_qubits)
        circuit.x(0)
    
    # Apply right subcircuit controlled on qubit 0 = |1⟩
    for instruction in right_circuit.data:
        gate = instruction.operation
        target_qubits = [right_circuit.qubits.index(q) + 1 for q in instruction.qubits]
        
        # Apply gate controlled on qubit 0 being |1⟩
        if hasattr(gate, 'to_instruction'):
            ctrl_gate = gate.control(1)
            circuit.append(ctrl_gate, [0] + target_qubits)
        else:
            ctrl_gate = gate.control(1)
            circuit.append(ctrl_gate, [0] + target_qubits)
    
    return circuit


def prepare_gaussian_factor(
    n_qubits: int = 6,
    mean: float = 0.0,
    std: float = 1.0,
    n_std: float = 4.0
) -> QuantumCircuit:
    """
    Prepare quantum state encoding Gaussian N(mean, std²) factor.
    
    Discretizes Gaussian over [-n_std·std, +n_std·std] range and
    encodes into quantum amplitudes.
    
    Parameters
    ----------
    n_qubits : int, default=6
        Number of qubits (64 bins for default)
    mean : float, default=0.0
        Gaussian mean
    std : float, default=1.0
        Gaussian standard deviation
    n_std : float, default=4.0
        Range in standard deviations (covers ±4σ ≈ 99.99%)
    
    Returns
    -------
    circuit : QuantumCircuit
        Gaussian factor state preparation circuit
    
    Examples
    --------
    >>> from qfdp_multiasset.state_prep import prepare_gaussian_factor
    >>> 
    >>> # Standard normal factor
    >>> circuit = prepare_gaussian_factor(n_qubits=6, mean=0, std=1)
    >>> 
    >>> # Validate by statevector
    >>> from qiskit.quantum_info import Statevector
    >>> statevector = Statevector(circuit)
    >>> probs = np.abs(statevector.data)**2
    >>> 
    >>> # Check statistics
    >>> N = 2**6
    >>> grid = np.linspace(-4, 4, N)
    >>> estimated_mean = np.sum(grid * probs)
    >>> print(f"Estimated mean: {estimated_mean:.3f} (target: 0)")
    """
    N = 2**n_qubits
    
    # Discretize Gaussian over range
    x_min = mean - n_std * std
    x_max = mean + n_std * std
    x_grid = np.linspace(x_min, x_max, N)
    
    # Gaussian pdf
    pdf = norm.pdf(x_grid, loc=mean, scale=std)
    
    # Normalize to sum to 1
    probabilities = pdf / pdf.sum()
    
    # Prepare using Grover-Rudolph
    circuit = prepare_marginal_distribution(x_grid, probabilities, n_qubits, validate=True)
    
    return circuit


def prepare_lognormal_asset(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_qubits: int = 8,
    n_std: float = 3.0
) -> Tuple[QuantumCircuit, np.ndarray]:
    """
    Prepare log-normal asset price distribution under Black-Scholes.
    
    Models S_T ~ LogNormal with:
        E[S_T] = S0 * exp(r*T)
        Var[log(S_T/S0)] = sigma^2 * T
    
    Parameters
    ----------
    S0 : float
        Initial asset price
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to maturity (years)
    n_qubits : int, default=8
        Number of qubits (256 price points)
    n_std : float, default=3.0
        Range in standard deviations
    
    Returns
    -------
    circuit : QuantumCircuit
        State preparation circuit
    prices : np.ndarray
        Price grid corresponding to basis states |i⟩
    
    Examples
    --------
    >>> from qfdp_multiasset.state_prep import prepare_lognormal_asset
    >>> 
    >>> # AAPL-like parameters
    >>> S0, r, sigma, T = 150.0, 0.03, 0.25, 1.0
    >>> circuit, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits=8)
    >>> 
    >>> print(f"Price range: [{prices[0]:.2f}, {prices[-1]:.2f}]")
    >>> print(f"Circuit depth: {circuit.depth()}")
    """
    N = 2**n_qubits
    
    # Log-returns distribution: r ~ N(μ, σ²)
    # where μ = (r - 0.5*σ²)*T, σ_r = σ*√T
    # Clamp σ=0 to avoid degenerate distribution
    sigma_clamped = max(sigma, 1e-6)
    mu = (r - 0.5*sigma_clamped**2) * T
    sigma_r = sigma_clamped * np.sqrt(T)
    
    # Price range (in log-space)
    log_S_min = np.log(S0) + mu - n_std * sigma_r
    log_S_max = np.log(S0) + mu + n_std * sigma_r
    
    # Price grid (exponential spacing)
    log_prices = np.linspace(log_S_min, log_S_max, N)
    prices = np.exp(log_prices)
    
    # Log-normal pdf (transform from log-space)
    # p(S) = p(log S) / S
    log_returns = log_prices - np.log(S0)
    pdf_log = norm.pdf(log_returns, loc=mu, scale=sigma_r)
    pdf_price = pdf_log / prices  # Jacobian transformation
    
    # Normalize
    probabilities = pdf_price / pdf_price.sum()
    
    # Prepare quantum state
    circuit = prepare_marginal_distribution(prices, probabilities, n_qubits, validate=True)
    
    return circuit, prices


def compute_fidelity(
    circuit: QuantumCircuit,
    target_probabilities: np.ndarray
) -> float:
    """
    Compute fidelity between prepared state and target distribution.
    
    Fidelity F = (Σ √(p_i q_i))² where p_i are target probabilities
    and q_i are measured probabilities from quantum state.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        State preparation circuit
    target_probabilities : np.ndarray
        Target probability distribution
    
    Returns
    -------
    fidelity : float
        Fidelity in [0, 1], where 1 = perfect match
    
    Examples
    --------
    >>> circuit = prepare_gaussian_factor(n_qubits=6)
    >>> 
    >>> # Target: standard normal
    >>> N = 64
    >>> x = np.linspace(-4, 4, N)
    >>> target_probs = norm.pdf(x)
    >>> target_probs /= target_probs.sum()
    >>> 
    >>> fidelity = compute_fidelity(circuit, target_probs)
    >>> print(f"Fidelity: {fidelity:.4f}")
    """
    # Get statevector
    statevector = Statevector(circuit)
    measured_amplitudes = statevector.data
    measured_probs = np.abs(measured_amplitudes)**2
    
    # Pad target if needed
    N = len(measured_probs)
    if len(target_probabilities) < N:
        padded = np.zeros(N)
        padded[:len(target_probabilities)] = target_probabilities
        target_probabilities = padded
    elif len(target_probabilities) > N:
        target_probabilities = target_probabilities[:N]
    
    # Renormalize
    target_probabilities = target_probabilities / target_probabilities.sum()
    
    # Compute fidelity (Bhattacharyya coefficient squared)
    fidelity = np.sum(np.sqrt(target_probabilities * measured_probs))**2
    
    return float(fidelity)


def estimate_resource_cost(n_qubits: int) -> dict:
    """
    Estimate T-count and depth for Grover-Rudolph state preparation.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits
    
    Returns
    -------
    resources : dict
        - 'ry_gates': Number of Ry rotations
        - 't_count_estimate': Estimated T-gates (assumes Ry ≈ 3 T-gates)
        - 'depth_estimate': Estimated circuit depth
        - 'formula': Scaling formula
    
    Notes
    -----
    **Grover-Rudolph Complexity:**
    
    - Ry gates: O(2^n) for n qubits
    - Each Ry decomposes to ~3 T-gates (Solovay-Kitaev)
    - Depth: O(n·2^n) worst case (sequential controlled rotations)
    
    **For multi-asset encoding:**
    
    - N assets × n qubits/asset = N·n total qubits
    - Independent preparation: N × (2^n) Ry gates
    - Total T-count: N × 2^n × 3 ≈ 3N·2^n
    
    Examples
    --------
    >>> from qfdp_multiasset.state_prep import estimate_resource_cost
    >>> 
    >>> # Single asset (8 qubits)
    >>> resources = estimate_resource_cost(8)
    >>> print(f"T-count estimate: {resources['t_count_estimate']:,}")
    >>> 
    >>> # 5 assets (5 × 8 = 40 total qubits, but independent prep)
    >>> N_assets = 5
    >>> single_asset = estimate_resource_cost(8)
    >>> total_t_count = N_assets * single_asset['t_count_estimate']
    >>> print(f"Total T-count (5 assets): {total_t_count:,}")
    """
    N = 2**n_qubits
    
    # Number of Ry rotations (approximately 2^n - 1)
    ry_gates = N - 1
    
    # T-count estimate (Ry ≈ 3 T-gates via Clifford+T decomposition)
    c_prep = 3  # Constant per Ry gate
    t_count_estimate = ry_gates * c_prep
    
    # Depth estimate (pessimistic: sequential application)
    depth_estimate = n_qubits * ry_gates
    
    return {
        'n_qubits': n_qubits,
        'ry_gates': ry_gates,
        't_count_estimate': t_count_estimate,
        't_count_per_qubit': t_count_estimate / n_qubits if n_qubits > 0 else 0,
        'depth_estimate': depth_estimate,
        'formula': f"T-count ≈ {c_prep} × (2^{n_qubits} - 1) = {t_count_estimate}"
    }
