"""
Enhanced IQFT Module - Unified from FB-QDP + qfdp_multiasset
=============================================================

Combines:
1. FB-QDP explicit IQFT (custom gate-by-gate construction)
2. qfdp_multiasset tensor IQFT (parallel multi-register)
3. New optimizations: approximate QFT, resource-aware selection

Features:
- Exact IQFT: Full precision O(n²) depth
- Approximate IQFT: Controlled precision with O(n log n) depth
- Tensor IQFT: Parallel transforms on multiple registers
- Resource estimation: Gate counts, depth, T-gates
- Automatic selection based on qubit count and precision needs

Author: QFDP Unified Research Team
Date: November 30, 2025
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT


@dataclass
class IQFTConfig:
    """Configuration for IQFT construction."""
    approximation_degree: int = 0  # 0 = exact, >0 = drop small rotations
    do_swaps: bool = True           # Include final bit reversal
    optimization_level: int = 0     # 0=none, 1=basic, 2=aggressive
    insert_barriers: bool = False   # For visualization/debugging


@dataclass
class IQFTResources:
    """Resource usage estimates for IQFT."""
    n_qubits: int
    h_gates: int
    phase_gates: int
    swap_gates: int
    total_gates: int
    depth_exact: int
    depth_approx: int
    t_count_estimate: int


# ============================================================================
# Core IQFT Implementations
# ============================================================================

def build_iqft_explicit(n_qubits: int, config: Optional[IQFTConfig] = None) -> QuantumCircuit:
    """
    Build explicit inverse QFT circuit (FB-QDP style).
    
    Constructs IQFT gate-by-gate with manual swaps and controlled phases.
    Most transparent implementation for understanding the algorithm.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits
    config : IQFTConfig, optional
        Configuration parameters
        
    Returns
    -------
    QuantumCircuit
        Explicit IQFT circuit
        
    Notes
    -----
    Algorithm:
    1. Swap qubits for bit reversal
    2. For each qubit j:
       - Apply Hadamard
       - Apply controlled phases from subsequent qubits
       - Phase angle: -2π / 2^(k-j+1)
    """
    if config is None:
        config = IQFTConfig()
    
    qc = QuantumCircuit(n_qubits, name="IQFT_explicit")
    
    # Step 1: Swap qubits for bit reversal (if enabled)
    if config.do_swaps:
        for i in range(n_qubits // 2):
            qc.swap(i, n_qubits - i - 1)
        if config.insert_barriers:
            qc.barrier()
    
    # Step 2: Inverse QFT rotations
    for j in range(n_qubits):
        # Hadamard on qubit j
        qc.h(j)
        
        # Controlled phases from subsequent qubits
        for k in range(j + 1, n_qubits):
            # Rotation angle decreases with distance
            power = k - j + 1
            angle = -2 * math.pi / (2 ** power)
            
            # Skip small rotations if approximating
            if config.approximation_degree > 0:
                if power > config.approximation_degree + 2:
                    continue
            
            qc.cp(angle, k, j)
        
        if config.insert_barriers and j < n_qubits - 1:
            qc.barrier()
    
    return qc


def build_iqft_library(n_qubits: int, config: Optional[IQFTConfig] = None) -> QuantumCircuit:
    """
    Build IQFT using Qiskit's optimized QFT library (qfdp_multiasset style).
    
    Uses Qiskit's built-in QFT with inverse=True. This is the most
    optimized implementation with hardware-aware transpilation.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits
    config : IQFTConfig, optional
        Configuration parameters
        
    Returns
    -------
    QuantumCircuit
        Library IQFT as QuantumCircuit definition
    """
    if config is None:
        config = IQFTConfig()
    
    qft = QFT(
        num_qubits=n_qubits,
        inverse=True,
        do_swaps=config.do_swaps,
        approximation_degree=config.approximation_degree
    )
    
    return qft.to_instruction().definition


def build_iqft_auto(n_qubits: int, config: Optional[IQFTConfig] = None) -> QuantumCircuit:
    """
    Automatically select best IQFT implementation based on size.
    
    Selection logic:
    - n ≤ 8: Explicit (easier to understand, visualize)
    - n > 8: Library (better optimization)
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits
    config : IQFTConfig, optional
        Configuration parameters
        
    Returns
    -------
    QuantumCircuit
        Best IQFT circuit for given size
    """
    if n_qubits <= 8:
        return build_iqft_explicit(n_qubits, config)
    else:
        return build_iqft_library(n_qubits, config)


# ============================================================================
# Tensor IQFT (Multi-Register)
# ============================================================================

def apply_iqft_to_register(
    circuit: QuantumCircuit,
    register: QuantumRegister,
    config: Optional[IQFTConfig] = None
) -> None:
    """
    Apply IQFT to a specific register in an existing circuit (in-place).
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to modify
    register : QuantumRegister
        Target register for IQFT
    config : IQFTConfig, optional
        Configuration parameters
    """
    if config is None:
        config = IQFTConfig()
    
    n = len(register)
    qft_gate = QFT(
        n,
        inverse=True,
        do_swaps=config.do_swaps,
        approximation_degree=config.approximation_degree
    ).to_gate(label="IQFT")
    
    circuit.append(qft_gate, list(register))


def apply_tensor_iqft(
    circuit: QuantumCircuit,
    registers: List[QuantumRegister],
    config: Optional[IQFTConfig] = None,
    parallel: bool = False
) -> None:
    """
    Apply IQFT to multiple registers (tensor product).
    
    For multi-asset pricing, applies IQFT independently to each asset
    register. Registers are independent so can be parallelized in
    hardware.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to modify in-place
    registers : List[QuantumRegister]
        List of registers to transform
    config : IQFTConfig, optional
        Configuration for each IQFT
    parallel : bool
        If True, insert barriers between IQFTs for parallel scheduling hint
        
    Examples
    --------
    >>> asset_regs = [QuantumRegister(8, f'asset_{i}') for i in range(5)]
    >>> circuit = QuantumCircuit(*asset_regs)
    >>> apply_tensor_iqft(circuit, asset_regs)
    """
    if config is None:
        config = IQFTConfig()
    
    for i, reg in enumerate(registers):
        apply_iqft_to_register(circuit, reg, config)
        
        # Insert barrier for parallel scheduling hint
        if parallel and i < len(registers) - 1:
            circuit.barrier()


# ============================================================================
# Resource Estimation
# ============================================================================

def estimate_iqft_resources(n_qubits: int, approximation_degree: int = 0) -> IQFTResources:
    """
    Estimate resource usage for IQFT on n qubits.
    
    Counts:
    - H gates: n
    - Phase gates (exact): n(n-1)/2
    - Phase gates (approx): reduced based on approximation_degree
    - Swap gates: n/2
    - Depth (exact): O(n²)
    - Depth (approx): O(n log n) to O(n)
    - T gates (fault-tolerant): ≈ phase gates × 40 (Clifford+T decomposition)
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits
    approximation_degree : int
        Approximation level (0 = exact)
        
    Returns
    -------
    IQFTResources
        Detailed resource breakdown
    """
    n = n_qubits
    
    # Hadamard gates: one per qubit
    h_gates = n
    
    # Controlled phase gates (exact QFT)
    phase_gates_exact = n * (n - 1) // 2
    
    # With approximation: drop rotations beyond approximation_degree
    if approximation_degree == 0:
        phase_gates = phase_gates_exact
    else:
        # Count only phases where power ≤ approximation_degree + 2
        phase_gates = 0
        for j in range(n):
            for k in range(j + 1, n):
                power = k - j + 1
                if power <= approximation_degree + 2:
                    phase_gates += 1
    
    # Swap gates for bit reversal
    swap_gates = n // 2
    
    # Total gates
    total_gates = h_gates + phase_gates + swap_gates * 3  # swap = 3 CNOTs
    
    # Depth estimates
    depth_exact = int(0.5 * n * n + 2 * swap_gates)  # Sequential phases + swaps
    
    if approximation_degree == 0:
        depth_approx = depth_exact
    else:
        # Approximation reduces depth roughly by factor of (log n / n)
        depth_approx = int(n * math.log2(n) + 2 * swap_gates) if n > 1 else depth_exact
    
    # T-count estimate (fault-tolerant compilation)
    # Each CP gate → ~40 T gates in Clifford+T basis
    t_count_estimate = phase_gates * 40
    
    return IQFTResources(
        n_qubits=n,
        h_gates=h_gates,
        phase_gates=phase_gates,
        swap_gates=swap_gates,
        total_gates=total_gates,
        depth_exact=depth_exact,
        depth_approx=depth_approx,
        t_count_estimate=t_count_estimate
    )


def estimate_tensor_iqft_resources(
    n_registers: int,
    qubits_per_register: int,
    approximation_degree: int = 0
) -> Dict[str, Any]:
    """
    Estimate resources for tensor IQFT on multiple registers.
    
    For multi-asset applications with N assets, each requiring n qubits.
    
    Parameters
    ----------
    n_registers : int
        Number of registers (e.g., number of assets)
    qubits_per_register : int
        Qubits per register
    approximation_degree : int
        Approximation level
        
    Returns
    -------
    dict
        Resource breakdown including:
        - Single register resources
        - Total resources (sum over all registers)
        - Parallel depth (if executed in parallel)
    """
    single = estimate_iqft_resources(qubits_per_register, approximation_degree)
    
    return {
        'n_registers': n_registers,
        'qubits_per_register': qubits_per_register,
        'single_register': single,
        'total_h_gates': n_registers * single.h_gates,
        'total_phase_gates': n_registers * single.phase_gates,
        'total_swap_gates': n_registers * single.swap_gates,
        'total_gates': n_registers * single.total_gates,
        'sequential_depth': n_registers * single.depth_exact,
        'parallel_depth': single.depth_exact,  # Registers are independent
        'total_t_count': n_registers * single.t_count_estimate,
    }


# ============================================================================
# Utility Functions
# ============================================================================

def compare_iqft_implementations(n_qubits: int) -> Dict[str, Any]:
    """
    Compare different IQFT implementations for given size.
    
    Builds and analyzes:
    - Explicit IQFT
    - Library IQFT
    - Approximate IQFT (degree=1)
    
    Returns comparison metrics.
    """
    results = {}
    
    # Explicit
    qc_explicit = build_iqft_explicit(n_qubits)
    results['explicit'] = {
        'gates': qc_explicit.size(),
        'depth': qc_explicit.depth(),
        'name': 'Explicit (FB-QDP style)'
    }
    
    # Library
    qc_library = build_iqft_library(n_qubits)
    results['library'] = {
        'gates': qc_library.size(),
        'depth': qc_library.depth(),
        'name': 'Library (qfdp_multiasset style)'
    }
    
    # Approximate
    config_approx = IQFTConfig(approximation_degree=1)
    qc_approx = build_iqft_library(n_qubits, config_approx)
    results['approximate'] = {
        'gates': qc_approx.size(),
        'depth': qc_approx.depth(),
        'name': 'Approximate (degree=1)'
    }
    
    # Resources
    resources = estimate_iqft_resources(n_qubits)
    results['resources'] = {
        'h_gates': resources.h_gates,
        'phase_gates': resources.phase_gates,
        'swap_gates': resources.swap_gates,
        'depth_estimate': resources.depth_exact,
        't_count': resources.t_count_estimate
    }
    
    return results


def global_phase_invariant_fidelity(
    state_a: np.ndarray,
    state_b: np.ndarray
) -> float:
    """
    Compute fidelity between two statevectors up to global phase.
    
    F = |⟨ψ_a|ψ_b⟩|²
    
    Useful for validating IQFT correctness.
    """
    inner = np.vdot(state_a, state_b)
    return float(np.abs(inner) ** 2)
