"""
Tensor QFT/IQFT Utilities (Phase 4)
====================================

Implements parallel QFT/IQFT across multiple asset registers.

- build_qft: Construct QFT/IQFT for a single register
- apply_tensor_qft: Apply QFT/IQFT to a list of registers in-place
- estimate_qft_resources: Provide O(n^2) resource estimates

Design goals:
- Compatible with Phase 2/3 register layout (asset_i, factor_k)
- No reliance on Aer; uses statevector for tests
- Production-ready for QSP/IQFT-based payoffs in later phases

Author: QFDP Multi-Asset Research Team
Date: November 2025
"""

from typing import List, Dict, Any
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector
import numpy as np


def build_qft(n_qubits: int, inverse: bool = False, do_swaps: bool = True) -> QuantumCircuit:
    """
    Build a QFT or IQFT circuit for a single register.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the register.
    inverse : bool, default=False
        If True, returns the inverse QFT (IQFT).
    do_swaps : bool, default=True
        If True, include final qubit swaps (bit-reversal correction).

    Returns
    -------
    QuantumCircuit
        QFT or IQFT circuit on n_qubits.
    """
    qft = QFT(num_qubits=n_qubits, inverse=inverse, do_swaps=do_swaps, approximation_degree=0)
    return qft.to_instruction().definition  # as QuantumCircuit


def apply_qft(circuit: QuantumCircuit, register: QuantumRegister, inverse: bool = False, do_swaps: bool = True) -> None:
    """
    Apply QFT/IQFT to a specific register within an existing circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to modify in-place.
    register : QuantumRegister
        Target register to transform.
    inverse : bool, default=False
        If True, apply IQFT.
    do_swaps : bool, default=True
        Include final swaps.
    """
    qft_gate = QFT(len(register), inverse=inverse, do_swaps=do_swaps, approximation_degree=0).to_gate(label=("IQFT" if inverse else "QFT"))
    circuit.append(qft_gate, list(register))


def apply_tensor_qft(circuit: QuantumCircuit, registers: List[QuantumRegister], inverse: bool = False, do_swaps: bool = True) -> None:
    """
    Apply QFT/IQFT in parallel across multiple registers (sequentially appended).

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to modify in-place.
    registers : List[QuantumRegister]
        List of registers to transform (e.g., all asset_i registers).
    inverse : bool, default=False
        Apply IQFT if True.
    do_swaps : bool, default=True
        Include swaps for each register.
    """
    for reg in registers:
        apply_qft(circuit, reg, inverse=inverse, do_swaps=do_swaps)


def estimate_qft_resources(n_qubits: int) -> Dict[str, Any]:
    """
    Estimate resource usage for QFT/IQFT on n_qubits.

    Returns
    -------
    dict with keys:
    - h_gates: n
    - phase_gates: n(n-1)/2
    - swaps: n//2 (if do_swaps)
    - depth_estimate: ~n^2/2 (heuristic)
    """
    n = n_qubits
    return {
        'h_gates': n,
        'phase_gates': n * (n - 1) // 2,
        'swaps': n // 2,
        'depth_estimate': int(0.5 * n * n)
    }


def estimate_tensor_qft_resources(n_registers: int, n_qubits: int) -> Dict[str, Any]:
    """
    Estimate resources for applying QFT to multiple registers of equal size.
    """
    single = estimate_qft_resources(n_qubits)
    return {
        'registers': n_registers,
        'total_h': n_registers * single['h_gates'],
        'total_phase': n_registers * single['phase_gates'],
        'total_swaps': n_registers * single['swaps'],
        'total_depth_estimate': n_registers * single['depth_estimate']
    }


def global_phase_invariant_fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """
    Compute fidelity between two statevectors up to a global phase.
    F = |<a|b>|^2
    """
    inner = np.vdot(state_a, state_b)
    return float(np.abs(inner) ** 2)
