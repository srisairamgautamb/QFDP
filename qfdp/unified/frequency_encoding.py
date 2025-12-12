"""
Quantum State Preparation for Frequency-Domain Encoding

This module implements Step 8 of the FB-IQFT flowchart:
- Normalize the modified CF ψ(u_j) values
- Encode as quantum state |ψ_freq⟩ = Σ a_j|j⟩ via amplitude encoding
- Verify encoding fidelity (optional)

The key is that we encode M=16-32 complex amplitudes onto k=4-5 qubits,
which enables the subsequent shallow IQFT.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from typing import Tuple


def encode_frequency_state(
    psi_values: np.ndarray,
    num_qubits: int
) -> Tuple[QuantumCircuit, float]:
    """
    Step 8: Encode ψ(u_j) as quantum state |ψ_freq⟩.
    
    We normalize the modified characteristic function values and prepare
    an amplitude-encoded state:
    
        a_j = ψ(u_j) / √(Σ_k |ψ(u_k)|²)    (normalization)
        |ψ_freq⟩ = Σ_{j=0}^{M-1} a_j |j⟩   (quantum state)
    
    This state lives in the computational basis {|0⟩, |1⟩, ..., |M-1⟩}
    where M = 2^k and k is the number of qubits.
    
    Args:
        psi_values: Modified CF ψ(u_j) from Carr-Madan, shape (M,), complex
        num_qubits: Number of qubits k = ⌈log₂(M)⌉
    
    Returns:
        circuit: QuantumCircuit with StatePreparation gate applied
        norm_factor: √(Σ |ψ|²), saved for later price reconstruction
    
    Raises:
        AssertionError: If M ≠ 2^{num_qubits}
    
    Notes:
        - Qiskit StatePreparation handles complex amplitudes (≥v0.45)
        - If issues arise, fall back to magnitude/phase decomposition
        - Circuit depth: O(M) for arbitrary amplitude state
    
    Example:
        >>> psi = np.array([1.0, 0.5+0.5j, 0.3, 0.2-0.1j])
        >>> circuit, norm = encode_frequency_state(psi, num_qubits=2)
        >>> print(f"Circuit depth: {circuit.depth()}")
        >>> print(f"Normalization factor: {norm:.4f}")
    """
    M = len(psi_values)
    assert M == 2**num_qubits, \
        f"M={M} must equal 2^{num_qubits}={2**num_qubits}"
    
    # Compute normalization factor (save for post-processing)
    norm_factor = np.sqrt(np.sum(np.abs(psi_values)**2))
    
    # Normalized amplitudes (complex)
    amplitudes = psi_values / norm_factor
    
    # Create quantum circuit
    qc = QuantumCircuit(num_qubits)
    
    # StatePreparation gate (handles complex amplitudes in Qiskit ≥0.45)
    # NOTE: If Qiskit version has issues with complex amplitudes, use:
    #   magnitudes = np.abs(amplitudes)
    #   phases = np.angle(amplitudes)
    #   # Build RY-tree + phase gates manually
    state_prep = StatePreparation(amplitudes)
    qc.append(state_prep, range(num_qubits))
    
    return qc, norm_factor


def verify_encoding(
    circuit: QuantumCircuit,
    target_amplitudes: np.ndarray
) -> float:
    """
    Verify state preparation fidelity via statevector simulation.
    
    Computes the overlap |⟨ψ_target|ψ_actual⟩|² between the target
    amplitudes and the actual statevector produced by the circuit.
    
    Fidelity should be ≈1.0 (typically >0.9999) for correct encoding.
    
    Args:
        circuit: QuantumCircuit from encode_frequency_state()
        target_amplitudes: Expected normalized amplitudes, shape (M,)
    
    Returns:
        fidelity: |⟨ψ_target|ψ_actual⟩|², range [0, 1]
    
    Example:
        >>> psi = np.array([0.6, 0.8, 0.0, 0.0])  # Already normalized
        >>> circuit, _ = encode_frequency_state(psi, 2)
        >>> fidelity = verify_encoding(circuit, psi)
        >>> assert fidelity > 0.999, "State preparation failed"
    
    Notes:
        This is primarily for debugging and validation during development.
        Remove from production code to avoid statevector overhead.
    """
    from qiskit.quantum_info import Statevector
    
    # Get actual statevector from circuit
    sv = Statevector(circuit)
    actual = sv.data
    
    # Compute overlap: |⟨target|actual⟩|²
    overlap = np.vdot(target_amplitudes, actual)
    fidelity = np.abs(overlap)**2
    
    return fidelity
