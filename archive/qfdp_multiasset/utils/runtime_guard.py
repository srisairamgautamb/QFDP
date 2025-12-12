"""
Runtime Guard Utilities
=======================

Helpers to prevent pathological memory blow-ups when constructing statevectors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from qiskit import QuantumCircuit


BYTES_PER_COMPLEX128 = 16  # numpy complex128


@dataclass
class StatevectorBudget:
    max_qubits: int
    max_bytes: int


DEFAULT_BUDGET = StatevectorBudget(max_qubits=22, max_bytes=512 * 1024 * 1024)  # 512 MB


def estimate_statevector_bytes(num_qubits: int) -> int:
    """
    Estimate memory (bytes) for a complex128 statevector with `num_qubits` qubits.
    """
    amps = 1 << num_qubits  # 2**n
    return amps * BYTES_PER_COMPLEX128


def ensure_statevector_ok(
    circuit: QuantumCircuit,
    budget: StatevectorBudget = DEFAULT_BUDGET
) -> None:
    """
    Raise MemoryError if a statevector for `circuit` would exceed budget.
    """
    n = circuit.num_qubits
    if n > budget.max_qubits:
        raise MemoryError(
            f"Circuit has {n} qubits; exceeds safety cap of {budget.max_qubits}. "
            "Use fewer qubits or a sampler backend."
        )
    bytes_needed = estimate_statevector_bytes(n)
    if bytes_needed > budget.max_bytes:
        raise MemoryError(
            f"Statevector would need {bytes_needed/1024/1024:.1f} MB (> {budget.max_bytes/1024/1024:.1f} MB)."
        )
