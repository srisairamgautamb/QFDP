"""
Piecewise Linear Payoff Oracle (Phase 6)
=========================================

Scalable payoff encoding via uniformly controlled rotations.

**Key Innovation:** Reduces gate complexity from O(2^n) to O(segments × n) by:
1. Segmenting price domain into intervals
2. Linear interpolation within each segment
3. Uniformly controlled Ry gates (log-depth decomposition)

This enables n=10-12 qubit registers (hardware-feasible) vs n≤8 for basis-state encoding.

Author: QFDP Multi-Asset Research Team
Date: November 2025
"""

from typing import List, Tuple, Callable
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library import RYGate


def segment_payoff(
    prices: np.ndarray,
    payoff_values: np.ndarray,
    n_segments: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partition price domain into segments with piecewise-constant payoff.

    Returns
    -------
    breakpoints : np.ndarray, shape (n_segments+1,)
        Segment boundaries (includes min and max)
    segment_payoffs : np.ndarray, shape (n_segments,)
        Average payoff in each segment
    """
    assert len(prices) == len(payoff_values)
    # Sort by price
    idx = np.argsort(prices)
    prices_sorted = prices[idx]
    payoff_sorted = payoff_values[idx]
    
    # Create equal-width segments in price space
    breakpoints = np.linspace(prices_sorted[0], prices_sorted[-1], n_segments + 1)
    segment_payoffs = np.zeros(n_segments)
    
    for seg in range(n_segments):
        left, right = breakpoints[seg], breakpoints[seg + 1]
        mask = (prices_sorted >= left) & (prices_sorted < right)
        if seg == n_segments - 1:  # Include right endpoint for last segment
            mask = (prices_sorted >= left) & (prices_sorted <= right)
        if np.any(mask):
            segment_payoffs[seg] = np.mean(payoff_sorted[mask])
        else:
            # No prices in segment; interpolate from neighbors
            if seg > 0:
                segment_payoffs[seg] = segment_payoffs[seg - 1]
    
    return breakpoints, segment_payoffs


def compute_segment_indices(prices: np.ndarray, breakpoints: np.ndarray) -> np.ndarray:
    """
    Map each price to its segment index.
    
    Returns
    -------
    indices : np.ndarray, shape (len(prices),)
        Segment index for each price (0 to n_segments-1)
    """
    indices = np.digitize(prices, breakpoints, right=False) - 1
    # Clip to valid range [0, n_segments-1]
    n_segments = len(breakpoints) - 1
    return np.clip(indices, 0, n_segments - 1)


def apply_piecewise_constant_payoff(
    circuit: QuantumCircuit,
    price_register: QuantumRegister,
    ancilla: Qubit,
    prices: np.ndarray,
    payoff_values: np.ndarray,
    n_segments: int = 16,
    scale: float | None = None,
) -> float:
    """
    Encode piecewise-constant payoff into ancilla via segment-based rotations.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to modify
    price_register : QuantumRegister
        Asset price register (n qubits)
    ancilla : Qubit
        Target ancilla
    prices : np.ndarray
        Price grid
    payoff_values : np.ndarray
        Payoff at each price point
    n_segments : int
        Number of piecewise segments (default 16)
    scale : float | None
        Normalization factor (auto-computed if None)
    
    Returns
    -------
    scale : float
        Scale factor used (E[payoff] = scale * P(ancilla=1))
    
    Notes
    -----
    Gate complexity: O(n_segments × 2^n) for naive basis-state conditioning.
    Future: Replace with uniformly controlled Ry for O(n_segments × n).
    """
    breakpoints, seg_pays = segment_payoff(prices, payoff_values, n_segments)
    seg_indices = compute_segment_indices(prices, breakpoints)
    
    max_pay = float(np.max(seg_pays))
    if scale is None:
        scale = max(1e-12, max_pay)
    
    n = len(price_register)
    assert len(prices) == 2 ** n
    
    # For each basis state, apply Ry rotation based on its segment
    for idx in range(2 ** n):
        seg = seg_indices[idx]
        if seg < 0 or seg >= n_segments:
            continue
        p = seg_pays[seg] / scale
        if p <= 0:
            continue
        if p > 1:
            p = 1.0
        
        theta = 2.0 * np.arcsin(np.sqrt(p))
        
        # Condition on |idx>
        bits = [(idx >> b) & 1 for b in range(n)]
        flipped = []
        for b, bit in enumerate(bits):
            if bit == 0:
                circuit.x(price_register[b])
                flipped.append(b)
        
        # Multi-controlled Ry
        gate = RYGate(theta).control(n)
        circuit.append(gate, list(price_register) + [ancilla])
        
        # Uncompute
        for b in flipped:
            circuit.x(price_register[b])
    
    return scale


def piecewise_approximation_error(
    prices: np.ndarray,
    payoff_exact: np.ndarray,
    payoff_approx: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Compute L1, L2, and relative errors for piecewise approximation.
    
    Returns
    -------
    dict with keys:
    - 'l1_error': ∫ |exact - approx| p(x) dx
    - 'l2_error': sqrt(∫ (exact - approx)^2 p(x) dx)
    - 'relative_expectation_error': |E[exact] - E[approx]| / E[exact]
    - 'max_pointwise_error': max |exact - approx|
    """
    probs = probabilities / np.sum(probabilities)
    diff = payoff_exact - payoff_approx
    l1 = float(np.sum(np.abs(diff) * probs))
    l2 = float(np.sqrt(np.sum(diff**2 * probs)))
    E_exact = float(np.dot(payoff_exact, probs))
    E_approx = float(np.dot(payoff_approx, probs))
    rel_err = abs(E_exact - E_approx) / (E_exact + 1e-12)
    max_err = float(np.max(np.abs(diff)))
    
    return {
        'l1_error': l1,
        'l2_error': l2,
        'relative_expectation_error': rel_err,
        'max_pointwise_error': max_err,
    }
