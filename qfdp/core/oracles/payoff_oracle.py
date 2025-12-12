"""
Payoff Oracles (Phase 5 - Initial)
===================================

Implements a digital (threshold) payoff oracle for an asset price register.

Oracle marks an ancilla qubit when the discretized price S >= K.
This provides a foundational building block for amplitude estimation.

Note: This version uses explicit basis-state marking (O(2^n)) suitable for
unit tests (n ≤ 8). Future phases will replace with uniformly controlled
rotations for scalable piecewise approximations of continuous payoffs.

Author: QFDP Multi-Asset Research Team
Date: November 2025
"""

from typing import List, Tuple
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RYGate


def _mark_basis_state(circuit: QuantumCircuit, register: QuantumRegister, target: Qubit, index: int) -> None:
    """
    Apply an MCX gate on target controlled on the register being in |index>.

    Implements basis-state conditioning via X on 0-bits, then MCX, then uncompute.
    Suitable for small n (testing); not scalable to large n.
    """
    n = len(register)
    bits = [(index >> b) & 1 for b in range(n)]

    # Map |index> to |11..1> by flipping controls where bit=0
    for b, bit in enumerate(bits):
        if bit == 0:
            circuit.x(register[b])

    # Multi-controlled X (all controls = register qubits)
    circuit.mcx(list(register), target)  # relies on default decomposition

    # Uncompute flips
    for b, bit in enumerate(bits):
        if bit == 0:
            circuit.x(register[b])


def mark_threshold_states(
    circuit: QuantumCircuit,
    price_register: QuantumRegister,
    ancilla: Qubit,
    prices: np.ndarray,
    K: float,
) -> int:
    """
    Mark ancilla when price basis corresponds to S >= K.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to append oracle operations to.
    price_register : QuantumRegister
        Asset price register (n qubits).
    ancilla : Qubit
        Ancilla qubit to flip when S >= K.
    prices : np.ndarray
        Price grid aligned with basis ordering of the register.
    K : float
        Strike/threshold.

    Returns
    -------
    count_marked : int
        Number of basis states marked (size of set {i | prices[i] >= K}).
    """
    assert len(prices) == 2 ** len(price_register), "Prices length must match register dimension"

    indices = [i for i, s in enumerate(prices) if s >= K]
    for idx in indices:
        _mark_basis_state(circuit, price_register, ancilla, idx)

    return len(indices)


def ancilla_probability(circuit: QuantumCircuit, ancilla: Qubit) -> float:
    """
    Compute probability of ancilla=1 from statevector (qubit marginal).
    """
    sv = Statevector(circuit)
    amps = sv.data
    pos = circuit.qubits.index(ancilla)
    p1 = 0.0
    for i, amp in enumerate(amps):
        if ((i >> pos) & 1) == 1:
            p1 += float((amp.conjugate() * amp).real)
    return p1


def classical_threshold_probability(prices: np.ndarray, probabilities: np.ndarray, K: float) -> float:
    """
    Compute classical probability P(S >= K) for discretized distribution.
    """
    assert len(prices) == len(probabilities)
    probs = probabilities / np.sum(probabilities)
    mask = prices >= K
    return float(np.sum(probs[mask]))


def call_payoff(prices: np.ndarray, K: float) -> np.ndarray:
    """
    Vectorized European call payoff max(S-K,0).
    """
    return np.maximum(prices - K, 0.0)


def _apply_ctrl_ry(circuit: QuantumCircuit, theta: float, controls: List[Qubit], target: Qubit) -> None:
    gate = RYGate(theta).control(len(controls))
    circuit.append(gate, controls + [target])


def apply_call_payoff_rotation(
    circuit: QuantumCircuit,
    price_register: QuantumRegister,
    ancilla: Qubit,
    prices: np.ndarray,
    K: float,
    scale: float | None = None,
) -> float:
    """
    Encode call payoff into ancilla via controlled Ry rotations.

    For each basis |i> on price_register, apply Ry(2*arcsin(sqrt(payoff_i/scale)))
    controlled on register being |i>. The resulting ancilla probability equals
    E[payoff]/scale.

    Returns the scale used to recover expectation.
    """
    n = len(price_register)
    assert len(prices) == 2 ** n
    pay = call_payoff(prices, K)
    max_pay = float(np.max(pay))
    if scale is None:
        scale = max(1e-12, max_pay)  # avoid division by zero
    # Iterate all basis indices
    for idx in range(2 ** n):
        p = pay[idx] / scale
        if p <= 0:
            continue
        if p > 1:
            p = 1.0
        a = np.sqrt(p)
        theta = 2.0 * np.arcsin(a)
        # Condition on |idx>
        bits = [(idx >> b) & 1 for b in range(n)]
        # map to |11..1>
        flipped = []
        for b, bit in enumerate(bits):
            if bit == 0:
                circuit.x(price_register[b])
                flipped.append(b)
        _apply_ctrl_ry(circuit, theta, list(price_register), ancilla)
        # uncompute flips
        for b in flipped:
            circuit.x(price_register[b])
    return scale


def ancilla_scaled_expectation(circuit: QuantumCircuit, ancilla: Qubit, scale: float) -> float:
    """
    Return E[payoff] = scale * P(ancilla=1).
    """
    return scale * ancilla_probability(circuit, ancilla)


def direct_expected_call_from_statevector(
    circuit: QuantumCircuit,
    price_register: QuantumRegister,
    prices: np.ndarray,
    K: float,
) -> float:
    """
    Compute E[max(S-K,0)] directly from statevector by marginalizing
    the specified price register.
    """
    n = circuit.num_qubits
    reg_positions = [circuit.qubits.index(q) for q in price_register]
    # Precompute mapping for basis indices of register
    reg_dim = 2 ** len(price_register)
    probs_reg = np.zeros(reg_dim, dtype=float)
    sv = Statevector(circuit).data
    for i, amp in enumerate(sv):
        # extract register index
        ridx = 0
        for j, pos in enumerate(reg_positions):
            ridx |= (((i >> pos) & 1) << j)
        probs_reg[ridx] += float((amp.conjugate() * amp).real)
    payoff = call_payoff(prices, K)
    return float(np.dot(payoff, probs_reg))
