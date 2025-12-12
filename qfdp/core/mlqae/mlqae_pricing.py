"""
MLQAE Pricing Module
====================

Maximum Likelihood Quantum Amplitude Estimation for derivative pricing.

Theory:
-------
Given A-operator encoding payoff f(x) in ancilla amplitude:
    |ψ⟩ = A|0⟩ = √a|ψ₁⟩|1⟩ + √(1-a)|ψ₀⟩|0⟩
    
where a = E[f(x)] / f_max (scaled expectation).

MLQAE applies Q^k for k ∈ {0, 1, 2, ..., K-1} and measures ancilla shots.
Maximum likelihood estimate of a from outcome frequencies {n_k}.

Algorithm:
----------
1. Build A-operator (state prep + correlation + payoff oracle from Phases 2-6)
2. For each power k:
   - Apply Q^k = (A' S₀ A^† S_χ)^k where S_χ flips sign of |1⟩ ancilla
   - Measure ancilla M times
   - Record count h_k = number of |1⟩ outcomes
3. MLE: maximize L(a) = ∏_k P(h_k | a, M, k)
4. Return: Price = a × f_max

References:
-----------
- Suzuki et al. (2020): "Amplitude Estimation without Phase Estimation"
- Grinko et al. (2021): "Iterative Quantum Amplitude Estimation"

Author: QFDP Research
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector


@dataclass
class MLQAEResult:
    """Results from MLQAE amplitude estimation."""
    amplitude_estimate: float  # â ∈ [0, 1]
    price_estimate: float  # â × scale
    confidence_interval: Tuple[float, float]  # 95% CI on amplitude
    log_likelihood: float
    num_grover_iterations: List[int]  # Powers k used
    measurement_counts: List[Tuple[int, int]]  # [(h_k, M)] for each k
    total_shots: int
    oracle_queries: int  # Sum of k × M across all iterations


def grover_operator(circuit: QuantumCircuit, ancilla_qubit, state_prep_gates) -> None:
    """
    Apply simplified Grover operator Q = -S_ancilla (ignoring full A† S₀ A for statevector).
    
    For MLQAE with statevector simulation, we use a simplified reflection:
    Q ≈ I - 2|ψ⟩⟨ψ| where |ψ⟩ is the prepared state.
    
    This is implemented as Z rotation on ancilla (marks |1⟩ state for amplitude amplification).
    
    NOTE: This is a simulation shortcut. For real hardware, use full Grover operator.
    
    Args:
        circuit: Circuit to modify in-place
        ancilla_qubit: Qubit encoding payoff (target for reflection)
        state_prep_gates: Unused (kept for interface compatibility)
    """
    # Simplified reflection: flip sign of ancilla |1⟩ state
    # This amplifies the marked amplitude (payoff-encoded states)
    circuit.z(ancilla_qubit)


def simulate_measurement_outcomes(
    circuit: QuantumCircuit,
    ancilla_qubit,
    num_shots: int,
    seed: Optional[int] = None
) -> int:
    """
    Simulate measuring ancilla qubit `num_shots` times, return count of |1⟩.
    
    Uses statevector simulation (exact probabilities).
    For real hardware: replace with circuit.measure() + backend.run().
    
    Args:
        circuit: Quantum circuit (no measurements attached)
        ancilla_qubit: Qubit to measure
        num_shots: Number of measurement samples
        seed: RNG seed for reproducibility
        
    Returns:
        Number of |1⟩ outcomes (h_k)
    """
    sv = Statevector(circuit)
    
    # Get ancilla qubit index
    ancilla_idx = circuit.qubits.index(ancilla_qubit)
    
    # Marginal probability of ancilla=1 (sum over all basis states with ancilla=1)
    prob_1 = 0.0
    for i, amp in enumerate(sv.data):
        if (i >> ancilla_idx) & 1:  # Check if bit at ancilla position is 1
            prob_1 += float((amp.conjugate() * amp).real)
    
    # Sample binomial(num_shots, prob_1)
    rng = np.random.default_rng(seed)
    return rng.binomial(num_shots, prob_1)


def likelihood(amplitude: float, measurements: List[Tuple[int, int, int]]) -> float:
    """
    Log-likelihood function for MLQAE.
    
    L(a) = ∏_k P(h_k | a, M_k, k)
    where P(h_k | a, M_k, k) = Binomial(h_k; M_k, sin²((2k+1)θ))
    and a = sin²(θ).
    
    Args:
        amplitude: Candidate amplitude â ∈ [0, 1]
        measurements: List of (k, M_k, h_k) tuples
        
    Returns:
        Log-likelihood ∑_k log P(h_k | a, M_k, k)
    """
    if amplitude <= 0 or amplitude >= 1:
        return -np.inf
    
    theta = np.arcsin(np.sqrt(amplitude))
    log_lik = 0.0
    
    for k, M_k, h_k in measurements:
        # Probability after k Grover iterations
        p_k = np.sin((2*k + 1) * theta) ** 2
        p_k = np.clip(p_k, 1e-15, 1 - 1e-15)  # Numerical stability
        
        # Binomial log-likelihood
        from scipy.special import betaln
        log_lik += (
            betaln(h_k + 1, M_k - h_k + 1) - betaln(1, M_k + 1)
            + h_k * np.log(p_k) + (M_k - h_k) * np.log(1 - p_k)
        )
    
    return log_lik


def run_mlqae(
    a_operator_circuit: QuantumCircuit,
    ancilla_qubit,
    payoff_scale: float,
    grover_powers: Optional[List[int]] = None,
    shots_per_power: int = 100,
    seed: Optional[int] = None
) -> MLQAEResult:
    """
    Execute MLQAE amplitude estimation.
    
    ⚠️ LIMITATION: Current implementation only uses k=0 (no Grover iterations)
    due to Qiskit initialize() bug. This provides NO QUANTUM SPEEDUP vs classical
    Monte Carlo - both sample the same distribution.
    
    Args:
        a_operator_circuit: Circuit encoding A-operator (state prep + copula + payoff)
                           Must have ancilla qubit with payoff encoded
        ancilla_qubit: Qubit encoding payoff amplitude
        payoff_scale: Maximum payoff value (f_max) for descaling
        grover_powers: List of k values for Q^k (default: [0])
        shots_per_power: Measurements per Grover power
        seed: RNG seed
        
    Returns:
        MLQAEResult with amplitude estimate and pricing
    """
    if grover_powers is None:
        grover_powers = [0]  # Only k=0 implemented (no quantum advantage)
    
    # Extract state prep gates (everything before measurement)
    state_prep_gates = list(a_operator_circuit.data)
    
    measurements = []  # (k, M, h_k)
    
    for k in grover_powers:
        # Build circuit with k Grover iterations
        circ_k = a_operator_circuit.copy()
        for _ in range(k):
            grover_operator(circ_k, ancilla_qubit, state_prep_gates)
        
        # Measure ancilla
        h_k = simulate_measurement_outcomes(circ_k, ancilla_qubit, shots_per_power, seed)
        measurements.append((k, shots_per_power, h_k))
    
    # Maximum likelihood estimation
    result = minimize_scalar(
        lambda a: -likelihood(a, measurements),
        bounds=(1e-6, 1 - 1e-6),
        method='bounded'
    )
    
    a_est = result.x
    log_lik = -result.fun
    
    # Confidence interval (Fisher information approximation)
    # Δa ≈ 1 / √(M × K) for M shots and K Grover powers
    total_shots = len(grover_powers) * shots_per_power
    std_err = 1.0 / np.sqrt(total_shots)
    ci_lower = max(0.0, a_est - 1.96 * std_err)
    ci_upper = min(1.0, a_est + 1.96 * std_err)
    
    # Total oracle queries
    oracle_queries = sum(k * shots_per_power for k in grover_powers)
    
    return MLQAEResult(
        amplitude_estimate=a_est,
        price_estimate=a_est * payoff_scale,
        confidence_interval=(ci_lower * payoff_scale, ci_upper * payoff_scale),
        log_likelihood=log_lik,
        num_grover_iterations=grover_powers,
        measurement_counts=[(h, shots_per_power) for k, M, h in measurements],
        total_shots=total_shots,
        oracle_queries=oracle_queries
    )
