"""
Invertible Quantum State Preparation for k>0 MLQAE
===================================================

Implements INVERTIBLE amplitude encoding using ONLY RY and controlled-RY gates.
This enables construction of proper Grover operator Q = -AS₀A†Sχ for amplitude
amplification with k>0, providing TRUE quantum speedup.

Key Difference from grover_rudolph.py:
--------------------------------------
- grover_rudolph.py: Uses circuit.initialize() → NOT invertible → k=0 only
- invertible_prep.py: Uses manual RY tree → Invertible → k>0 enabled

Mathematical Foundation:
------------------------
Grover-Rudolph decomposition (arXiv:quant-ph/0208112):

For target amplitudes {α₀, α₁, ..., α_{2^n-1}}, build binary tree:

Level 0: RY(θ₀) where tan(θ₀/2) = (α_right / α_left)
Level 1: Controlled-RY gates for left/right subtrees
...
Level n-1: Leaf nodes (individual amplitudes)

Each rotation is:
    θ = 2·arcsin(√(p_right / p_total))

where p = |amplitude|².

This decomposition uses ONLY:
- RY gates (single-qubit Y rotations)
- CRY gates (controlled-RY)
- X gates (for control value negation)

ALL gates have well-defined inverses:
- RY(θ)† = RY(-θ)
- CRY(θ)† = CRY(-θ)
- X† = X

Grover Operator Construction:
------------------------------
Q = -AS₀A†Sχ

Where:
- A: Invertible state preparation (this module)
- S₀: Reflection about |0...0⟩
- Sχ: Reflection about target states (ancilla |1⟩)

Query Complexity:
-----------------
- k=0: 1 query (no amplification)
- k=1: 2 queries (1 forward + 1 Grover)
- k=4: 5 queries (1 forward + 4 Grover)

Speedup: ε accuracy with O(√M) queries vs O(M) classical samples.

Author: QFDP Research Team
Date: November 2025
Research paper requirement: Enable k>0 for true quantum advantage
"""

import numpy as np
from typing import Tuple, List, Optional
from qiskit import QuantumCircuit, QuantumRegister
from scipy import stats
import warnings


def compute_rotation_angles_tree(
    target_probs: np.ndarray,
    min_prob: float = 1e-12
) -> List[List[float]]:
    """
    Compute RY rotation angles for binary tree decomposition.
    
    Implements recursive probability splitting for Grover-Rudolph encoding.
    
    Parameters
    ----------
    target_probs : np.ndarray, shape (2^n,)
        Target probability distribution (must be normalized)
    min_prob : float, default=1e-12
        Minimum probability threshold for numerical stability
    
    Returns
    -------
    angles : List[List[float]]
        Tree structure of rotation angles. Level k has 2^k angles.
        angles[k][j] is the rotation for node j at level k.
    
    Algorithm
    ---------
    For each node at level k with indices [start, end):
        1. Split into left [start, mid) and right [mid, end)
        2. Compute p_left = sum of probabilities in left subtree
        3. Compute p_total = sum of probabilities in [start, end)
        4. Rotation: θ = 2·arcsin(√(p_left / p_total))
        5. Recurse on left and right subtrees
    
    Complexity: O(2^n) for n qubits
    
    Examples
    --------
    >>> # Uniform distribution on 4 states
    >>> probs = np.array([0.25, 0.25, 0.25, 0.25])
    >>> angles = compute_rotation_angles_tree(probs)
    >>> # angles[0] = [π/2]  (50-50 split at root)
    >>> # angles[1] = [π/2, π/2]  (50-50 at each subtree)
    """
    n = int(np.log2(len(target_probs)))
    if 2**n != len(target_probs):
        raise ValueError(f"Probabilities length must be power of 2, got {len(target_probs)}")
    
    if not np.isclose(target_probs.sum(), 1.0, atol=1e-6):
        warnings.warn(f"Probabilities sum to {target_probs.sum():.6f}, normalizing")
        target_probs = target_probs / target_probs.sum()
    
    angles = []
    
    for level in range(n):
        level_angles = []
        nodes_at_level = 2**level
        states_per_node = 2**(n - level)
        
        for node_idx in range(nodes_at_level):
            # Indices for this node's probability range
            start = node_idx * states_per_node
            end = start + states_per_node
            mid = start + states_per_node // 2
            
            # Probabilities in left and right subtrees
            p_left = np.sum(target_probs[start:mid])
            p_total = np.sum(target_probs[start:end])
            
            if p_total < min_prob:
                # No probability in this subtree, use zero rotation
                theta = 0.0
            else:
                # Rotation to split probability
                # We want: |left⟩ with amplitude √(p_left/p_total)
                #          |right⟩ with amplitude √(p_right/p_total)
                # RY(θ): |0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩
                # For p_right probability: sin²(θ/2) = p_right/p_total
                # So: θ = 2·arcsin(√(p_right/p_total))
                p_right = np.sum(target_probs[mid:end])
                theta = 2 * np.arcsin(np.sqrt(p_right / p_total))
            
            level_angles.append(theta)
        
        angles.append(level_angles)
    
    return angles


def build_rotation_tree_circuit(
    angles: List[List[float]],
    qubits: QuantumRegister
) -> QuantumCircuit:
    """
    Build invertible circuit from rotation angle tree.
    
    Applies rotations level by level with appropriate controls.
    
    Parameters
    ----------
    angles : List[List[float]]
        Rotation angles from compute_rotation_angles_tree()
    qubits : QuantumRegister
        Qubits to operate on
    
    Returns
    -------
    circuit : QuantumCircuit
        Invertible state preparation circuit (RY/CRY gates only)
    
    Structure
    ---------
    Level 0: Unconditional RY on qubit 0
    Level 1: CRY controlled by qubit 0, targeting qubit 1
    Level 2: Multi-controlled-RY by qubits 0,1 targeting qubit 2
    ...
    
    For multi-controlled gates with specific control values (e.g., |01⟩),
    we use X gates to flip control qubits as needed.
    """
    n = len(angles)
    circuit = QuantumCircuit(qubits)
    
    if n == 0:
        return circuit
    
    # Level 0: Root rotation (unconditional)
    circuit.ry(angles[0][0], qubits[0])
    
    # Levels 1 to n-1: Controlled rotations
    for level in range(1, n):
        nodes = 2**level
        
        for node_idx in range(nodes):
            angle = angles[level][node_idx]
            
            # Target qubit for this level
            target_qubit = qubits[level]
            
            # Control qubits: all previous levels
            control_qubits = [qubits[i] for i in range(level)]
            
            # Binary representation of node_idx determines control values
            # E.g., node_idx=5 (binary 101) means controls should be |1⟩|0⟩|1⟩
            control_bits = format(node_idx, f'0{level}b')
            
            # Apply X gates to flip controls that should be |0⟩
            for i, bit in enumerate(control_bits):
                if bit == '0':
                    circuit.x(control_qubits[i])
            
            # Apply controlled-RY
            if level == 1:
                # Single-controlled-RY (native gate)
                circuit.cry(angle, control_qubits[0], target_qubit)
            else:
                # Multi-controlled-RY (decompose via mcry)
                circuit.mcry(angle, control_qubits, target_qubit)
            
            # Undo X gates
            for i, bit in enumerate(control_bits):
                if bit == '0':
                    circuit.x(control_qubits[i])
    
    return circuit


def prepare_lognormal_invertible(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_qubits: int = 6,
    n_std: float = 3.0
) -> Tuple[QuantumCircuit, np.ndarray]:
    """
    Prepare log-normal distribution with INVERTIBLE circuit.
    
    This is the research-grade version that enables k>0 MLQAE.
    
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
    n_qubits : int, default=6
        Number of qubits (2^6 = 64 price points)
    n_std : float, default=3.0
        Range coverage in standard deviations
    
    Returns
    -------
    circuit : QuantumCircuit
        INVERTIBLE state preparation (RY/CRY gates only)
    prices : np.ndarray
        Price grid corresponding to |i⟩ basis states
    
    Notes
    -----
    **Key Difference**: No circuit.initialize() - only parametrized gates.
    
    This enables:
    1. circuit.inverse() works correctly
    2. Grover operator Q = -AS₀A†Sχ can be built
    3. k>0 amplitude amplification
    4. True quantum speedup: O(√M) vs O(M)
    
    **Black-Scholes Model**:
    
    log(S_T/S0) ~ N(μ, σ_r²) where:
    - μ = (r - 0.5σ²)T
    - σ_r = σ√T
    
    **Discretization**:
    
    Price range: [S0·e^(μ-3σ_r), S0·e^(μ+3σ_r)]
    Grid: Uniform in log-space, exponential in price-space
    
    Examples
    --------
    >>> from qfdp.core.state_prep import prepare_lognormal_invertible
    >>> 
    >>> # Prepare AAPL-like distribution
    >>> circuit, prices = prepare_lognormal_invertible(
    ...     S0=150, r=0.03, sigma=0.25, T=1.0, n_qubits=6
    ... )
    >>> 
    >>> # Verify invertibility
    >>> circuit_inv = circuit.inverse()
    >>> print(f"Invertible: {circuit_inv is not None}")  # True
    >>> 
    >>> # Use in MLQAE with k>0
    >>> # (See mlqae_pricing.py for full example)
    """
    N = 2**n_qubits
    
    # Log-return distribution parameters
    sigma_clamped = max(sigma, 1e-6)
    mu = (r - 0.5*sigma_clamped**2) * T
    sigma_r = sigma_clamped * np.sqrt(T)
    
    # Price grid (log-space uniform → price-space exponential)
    log_S_min = np.log(S0) + mu - n_std * sigma_r
    log_S_max = np.log(S0) + mu + n_std * sigma_r
    log_prices = np.linspace(log_S_min, log_S_max, N)
    prices = np.exp(log_prices)
    
    # Log-normal PDF via change of variables
    # p(S) = p(log S) / S where p(log S) ~ N(log(S0) + μ, σ_r²)
    log_returns = log_prices - np.log(S0)
    pdf_log = stats.norm.pdf(log_returns, loc=mu, scale=sigma_r)
    pdf_price = pdf_log / prices  # Jacobian: d(log S) / dS = 1/S
    
    # Normalize to probabilities
    probs = pdf_price / pdf_price.sum()
    
    # Compute rotation tree
    angles = compute_rotation_angles_tree(probs)
    
    # Build circuit
    qr = QuantumRegister(n_qubits, 'state')
    circuit = build_rotation_tree_circuit(angles, qr)
    
    return circuit, prices


def prepare_gaussian_invertible(
    n_qubits: int = 6,
    mean: float = 0.0,
    std: float = 1.0,
    n_std: float = 4.0
) -> Tuple[QuantumCircuit, np.ndarray]:
    """
    Prepare Gaussian N(μ, σ²) with invertible circuit.
    
    Used for correlation factors in sparse copula encoding.
    
    Parameters
    ----------
    n_qubits : int, default=6
        Number of qubits
    mean : float, default=0.0
        Gaussian mean
    std : float, default=1.0
        Standard deviation
    n_std : float, default=4.0
        Range coverage (±4σ ≈ 99.99%)
    
    Returns
    -------
    circuit : QuantumCircuit
        Invertible Gaussian state preparation
    grid : np.ndarray
        Value grid corresponding to basis states
    
    Examples
    --------
    >>> circuit, grid = prepare_gaussian_invertible(n_qubits=6)
    >>> # Standard normal N(0,1) on 64 bins
    """
    N = 2**n_qubits
    
    # Discretize Gaussian
    x_min = mean - n_std * std
    x_max = mean + n_std * std
    grid = np.linspace(x_min, x_max, N)
    
    # Gaussian PDF
    pdf = stats.norm.pdf(grid, loc=mean, scale=std)
    probs = pdf / pdf.sum()
    
    # Build invertible circuit
    angles = compute_rotation_angles_tree(probs)
    qr = QuantumRegister(n_qubits, 'gauss')
    circuit = build_rotation_tree_circuit(angles, qr)
    
    return circuit, grid


def build_grover_operator(
    A_circuit: QuantumCircuit,
    ancilla_qubit,
    use_simplified: bool = False
) -> QuantumCircuit:
    """
    Build TRUE Grover operator Q = -AS₀A†Sχ for amplitude amplification.
    
    This is the research-grade implementation that provides quantum speedup.
    
    Parameters
    ----------
    A_circuit : QuantumCircuit
        A-operator (state prep + correlation + payoff encoding)
        MUST be invertible (use prepare_lognormal_invertible, not initialize())
    ancilla_qubit : Qubit
        Ancilla qubit encoding payoff amplitude
    use_simplified : bool, default=False
        If True, use simplified Z-gate reflection (for testing)
        If False, use full Q = -AS₀A†Sχ (for real speedup)
    
    Returns
    -------
    Q_circuit : QuantumCircuit
        Grover operator Q
    
    Mathematical Structure
    ----------------------
    Q = -AS₀A†Sχ
    
    Where:
    - Sχ: Reflection about marked states (ancilla |1⟩)
      Implementation: Z gate on ancilla
    
    - A†: Inverse of state preparation
      Implementation: A_circuit.inverse()
      **Requires**: A_circuit uses ONLY invertible gates (RY, CRY, etc.)
    
    - S₀: Reflection about |0...0⟩ state
      Implementation: 
        1. X on all qubits (flip |0⟩ ↔ |1⟩)
        2. Multi-controlled-Z on all qubits
        3. X on all qubits (flip back)
    
    - A: State preparation (reapply)
    
    - Global phase: -1 (π phase shift)
    
    Complexity
    ----------
    - Gates: 2·depth(A) + O(n) for reflections
    - Depth: 2·depth(A) + O(n)
    - Each Q application: 1 "query" to oracle
    
    Query Savings (vs Classical MC)
    --------------------------------
    Classical: ε accuracy requires O(1/ε²) samples
    Quantum (k>0): ε accuracy requires O(1/ε) Grover iterations
    Speedup: Quadratic reduction in samples
    
    Examples
    --------
    >>> # Build A-operator
    >>> circuit, prices = prepare_lognormal_invertible(100, 0.03, 0.2, 1.0, 6)
    >>> # ... add payoff encoding to ancilla ...
    >>> 
    >>> # Build Grover operator
    >>> Q = build_grover_operator(circuit, ancilla[0])
    >>> 
    >>> # Apply k iterations
    >>> for _ in range(k):
    >>>     circuit.compose(Q, inplace=True)
    """
    if use_simplified:
        # Simplified reflection (statevector simulation only)
        Q = QuantumCircuit(*A_circuit.qregs)
        Q.z(ancilla_qubit)
        return Q
    
    # Full Grover operator
    n_qubits = A_circuit.num_qubits
    Q = QuantumCircuit(*A_circuit.qregs)
    
    # Get all qubits except ancilla
    state_qubits = [q for q in A_circuit.qubits if q != ancilla_qubit]
    
    # Step 1: Sχ - Reflection about ancilla |1⟩
    Q.z(ancilla_qubit)
    
    # Step 2: A† - Invert state preparation
    try:
        A_inv = A_circuit.inverse()
        Q.compose(A_inv, inplace=True)
    except Exception as e:
        raise RuntimeError(
            f"Cannot invert A-operator. Ensure A uses ONLY invertible gates "
            f"(RY, CRY, X, CX, etc.). Error: {e}\n"
            "Use prepare_lognormal_invertible(), NOT circuit.initialize()."
        )
    
    # Step 3: S₀ - Reflection about |0...0⟩
    # Implements: I - 2|0⟩⟨0| = diag([-1, 1, 1, ..., 1])
    # Method: X-all → MCZ → X-all
    
    # Flip all state qubits (not ancilla)
    for qubit in state_qubits:
        Q.x(qubit)
    
    # Multi-controlled-Z on all state qubits
    # This marks |1...1⟩ state, which becomes |0...0⟩ after X-flip
    if len(state_qubits) == 1:
        Q.z(state_qubits[0])
    elif len(state_qubits) == 2:
        Q.h(state_qubits[1])
        Q.cx(state_qubits[0], state_qubits[1])
        Q.h(state_qubits[1])
    else:
        # Multi-controlled Z via H-MCX-H pattern
        Q.h(state_qubits[-1])
        Q.mcx(state_qubits[:-1], state_qubits[-1])
        Q.h(state_qubits[-1])
    
    # Flip back
    for qubit in state_qubits:
        Q.x(qubit)
    
    # Step 4: A - Reapply state preparation
    Q.compose(A_circuit, inplace=True)
    
    # Step 5: Global phase -1
    Q.global_phase = np.pi
    
    return Q


def select_adaptive_k(
    a_initial: float,
    target_accuracy: float = 0.05,
    max_k: int = 8,
    conservative: bool = True
) -> int:
    """
    Select optimal k to avoid over-rotation while maximizing accuracy.
    
    The Grover operator rotates amplitude by angle θ where sin²(θ) = a.
    After k iterations: a_k = sin²((2k+1)θ)
    
    Over-rotation occurs when (2k+1)θ > π/2, causing amplitude to decrease.
    
    Parameters
    ----------
    a_initial : float
        Initial amplitude estimate (from k=0 measurement or classical MC)
    target_accuracy : float, default=0.05
        Desired relative accuracy (5%)
    max_k : int, default=8
        Maximum k to consider
    conservative : bool, default=True
        If True, stay well below over-rotation threshold
    
    Returns
    -------
    k_opt : int
        Optimal k value (0 if amplitude already large or would over-rotate)
    
    Algorithm
    ---------
    1. Compute rotation angle: θ = arcsin(√a)
    2. Find max safe k: (2k+1)θ < π/2 (before over-rotation)
    3. Choose k balancing amplification vs accuracy
    4. For very small a: limit to k=1,2 (higher k risks over-rotation)
    5. For large a (>0.3): use k=0 only (already accurate)
    
    Examples
    --------
    >>> select_adaptive_k(0.01)  # Small amplitude
    1  # Conservative, avoid over-rotation
    
    >>> select_adaptive_k(0.5)  # Large amplitude  
    0  # Already accurate, no amplification needed
    """
    # Handle edge cases
    if a_initial <= 0 or a_initial >= 1:
        return 0
    
    # For large amplitudes, no amplification needed
    if a_initial > 0.3:
        return 0
    
    # Rotation angle
    theta = np.arcsin(np.sqrt(a_initial))
    
    # Maximum k before over-rotation: (2k+1)θ < π/2
    # Solve: k < (π/2θ - 1) / 2
    k_max_theory = int((np.pi / (2 * theta) - 1) / 2)
    
    if conservative:
        # Stay well below theoretical max (80% of threshold)
        k_max_safe = max(0, int(0.8 * k_max_theory))
    else:
        k_max_safe = k_max_theory
    
    # Cap at user-specified max
    k_max_safe = min(k_max_safe, max_k)
    
    # For very small amplitudes, be extra conservative
    if a_initial < 0.05:
        k_max_safe = min(k_max_safe, 2)  # k ∈ {0,1,2} only
    elif a_initial < 0.1:
        k_max_safe = min(k_max_safe, 4)  # k ∈ {0,1,2,4} max
    
    # Choose k that maximizes expected accuracy
    # Higher k → fewer total shots needed for same accuracy
    # But: higher k → closer to over-rotation risk
    
    # Start with k=1 if safe (minimal amplification)
    if k_max_safe >= 1:
        k_opt = 1
    else:
        k_opt = 0
    
    # Can we safely go higher?
    if k_max_safe >= 2 and a_initial < 0.08:
        k_opt = 2  # Moderate amplification for small a
    
    return k_opt

def estimate_grover_iterations(
    target_amplitude: float,
    tolerance: float = 0.01,
    max_k: int = 20
) -> List[int]:
    """
    Estimate optimal Grover iterations for MLQAE.
    
    For amplitude a, after k Grover iterations:
        a_k = sin²((2k+1)·arcsin(√a))
    
    We want diverse k values that maximize information about a.
    
    Parameters
    ----------
    target_amplitude : float
        Rough estimate of amplitude (e.g., from classical MC)
    tolerance : float, default=0.01
        Desired estimation accuracy
    max_k : int, default=20
        Maximum iterations to consider
    
    Returns
    -------
    grover_powers : List[int]
        Recommended k values, e.g., [0, 1, 2, 4, 8]
    
    Strategy
    --------
    Use geometric progression: k ∈ {0, 1, 2, 4, 8, ...}
    
    This provides:
    - k=0: Initial amplitude (baseline)
    - k=1: Small amplification (validation)
    - k≥2: Significant amplification (speedup)
    
    For small amplitudes (a << 1), more iterations needed.
    For large amplitudes (a ≈ 1), fewer iterations sufficient.
    
    Examples
    --------
    >>> # Option with 5% expected return
    >>> powers = estimate_grover_iterations(0.05, tolerance=0.01)
    >>> print(powers)  # [0, 1, 2, 4, 8]
    """
    if target_amplitude <= 0 or target_amplitude >= 1:
        # Conservative default
        return [0, 1, 2, 4]
    
    # Geometric progression up to max_k
    powers = [0]
    k = 1
    while k <= max_k:
        powers.append(k)
        k *= 2
    
    # For very small amplitudes, limit iterations
    # (avoid over-rotation where amplitude wraps around)
    theta = np.arcsin(np.sqrt(target_amplitude))
    max_useful_k = int(np.pi / (4 * theta))
    
    powers = [k for k in powers if k <= max_useful_k]
    
    # Ensure at least [0, 1]
    if len(powers) < 2:
        powers = [0, 1]
    
    return powers


def validate_invertibility(circuit: QuantumCircuit) -> Tuple[bool, str]:
    """
    Validate that circuit is invertible (all gates have inverses).
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to validate
    
    Returns
    -------
    is_invertible : bool
        True if circuit can be inverted
    message : str
        Explanation or error message
    
    Examples
    --------
    >>> circuit = QuantumCircuit(4)
    >>> circuit.initialize([0.5, 0.5, 0.5, 0.5], range(2))
    >>> is_inv, msg = validate_invertibility(circuit)
    >>> print(is_inv)  # False
    >>> print(msg)  # "Circuit uses non-invertible gate: initialize"
    
    >>> circuit2, _ = prepare_lognormal_invertible(100, 0.03, 0.2, 1.0, 4)
    >>> is_inv, msg = validate_invertibility(circuit2)
    >>> print(is_inv)  # True
    """
    non_invertible_gates = {'initialize', 'reset', 'measure'}
    
    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        if gate_name in non_invertible_gates:
            return False, f"Circuit uses non-invertible gate: {gate_name}"
    
    # Try to invert
    try:
        circuit.inverse()
        return True, "Circuit is invertible (all gates have well-defined inverses)"
    except Exception as e:
        return False, f"Circuit inversion failed: {e}"
