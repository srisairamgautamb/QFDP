"""
Inverse Quantum Fourier Transform and Measurement

This module implements Steps 9-10 of the FB-IQFT flowchart:
- Apply IQFT to transform frequency → strike basis
- Measure qubits to extract probability distribution P(m) ≈ |g_m|²

The IQFT is the key quantum operation that converts the frequency-encoded
state into strike-space amplitudes.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from typing import Dict, Union


def apply_iqft(
    circuit: QuantumCircuit,
    num_qubits: int
) -> QuantumCircuit:
    """
    Step 9: Apply IQFT to transform |ψ_freq⟩ → |ψ_strike⟩.
    
    The IQFT performs the unitary transform:
    
        IQFT|j⟩ = (1/√M) Σ_{m=0}^{M-1} e^(-i2πjm/M) |m⟩
    
    Applied to our state |ψ_freq⟩ = Σ_j a_j|j⟩, we get:
    
        |ψ_strike⟩ = Σ_m g_m |m⟩
        
        where g_m = (1/√M) Σ_j a_j e^(-i2πjm/M)
    
    NOTE: g_m are Fourier-inverted coefficients, NOT option prices directly.
    Prices are recovered via calibration in Step 11-12.
    
    Args:
        circuit: QuantumCircuit with |ψ_freq⟩ prepared (from Step 8)
        num_qubits: Number of qubits k = ⌈log₂(M)⌉
    
    Returns:
        circuit: Modified circuit with IQFT applied
    
    Notes:
        - Qiskit's QFT implements IQFT via .inverse() method
        - Circuit depth: O(k²) = O(log²(M))
        - For k=4-5 qubits: depth ≈ 16-25 gates
        - SWAP gates at end can be omitted if measurement order adjusted
    
    Example:
        >>> qc = QuantumCircuit(4)
        >>> # ... state preparation ...
        >>> qc = apply_iqft(qc, num_qubits=4)
        >>> print(f"IQFT depth: {qc.depth()}")
    """
    # Create inverse QFT gate
    # QFT.inverse() gives us IQFT: exp(-i2πjm/M) convention
    iqft_gate = QFT(num_qubits, inverse=True)
    
    # Append to circuit
    circuit.append(iqft_gate, range(num_qubits))
    
    return circuit


def extract_strike_amplitudes(
    circuit: QuantumCircuit,
    num_shots: int = 8192,
    backend: Union[str, object] = 'simulator'
) -> Dict[int, float]:
    """
    Step 10: Measure qubits to extract P(m) ≈ |g_m|².
    
    After IQFT, the state is |ψ_strike⟩ = Σ_m g_m |m⟩.
    Measurement gives us the probability distribution:
    
        P(m) = |g_m|²
    
    These probabilities are then calibrated to option prices in Step 11-12.
    
    Args:
        circuit: QuantumCircuit with IQFT applied (from Step 9)
        num_shots: Number of measurement shots (typical: 4096-16384)
        backend: Execution backend
            - 'simulator': AerSimulator (ideal, noiseless)
            - 'ibm_torino' or Backend object: Real hardware
    
    Returns:
        probabilities: Dictionary {m: P(m)} for m ∈ {0, 1, ..., M-1}
            - Keys are integers (basis state indices)
            - Values are normalized probabilities (sum to 1.0)
    
    Raises:
        ValueError: If backend type is invalid
    
    Notes:
        - Bit ordering: Qiskit uses little-endian (qubit 0 = LSB)
        - On hardware: noise reduces fidelity, calibration corrects
        - Typical hardware error: 15-25% vs simulator
    
    Example:
        >>> qc = QuantumCircuit(4)
        >>> # ... IQFT applied ...
        >>> probs = extract_strike_amplitudes(qc, num_shots=8192)
        >>> print(f"Max probability: {max(probs.values()):.4f}")
    """
    from qiskit import transpile
    
    # Try to import AerSimulator, fallback to Statevector if not available
    try:
        from qiskit_aer import AerSimulator
        USE_AER = True
    except ImportError:
        from qiskit.quantum_info import Statevector
        USE_AER = False
    
    # Add measurements to a copy of the circuit
    qc = circuit.copy()
    num_qubits = circuit.num_qubits
    
    # Select backend and execution method
    use_primitives = False
    use_statevector = False
    
    if isinstance(backend, str):
        if backend == 'simulator':
            if USE_AER:
                backend_obj = AerSimulator()
            else:
                # Use Statevector simulation (no shots needed)
                use_statevector = True
                backend_obj = None
        elif backend == 'ibm_torino' or backend.startswith('ibm_'):
            # Use IBM Quantum Primitives for hardware
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
            service = QiskitRuntimeService()
            if backend == 'ibm_torino':
                backend_obj = service.backend('ibm_torino')
            else:
                backend_obj = service.backend(backend)
            use_primitives = True
        else:
            raise ValueError(f"Unknown backend string: {backend}")
    else:
        # Assume backend is already a Backend object
        backend_obj = backend
        # Check if it's an IBM backend (use primitives)
        try:
            from qiskit_ibm_runtime import IBMBackend
            if isinstance(backend_obj, IBMBackend):
                use_primitives = True
        except ImportError:
            pass
    
    # Execute based on backend type
    if use_statevector:
        # Use Statevector for exact simulation without AerSimulator
        sv = Statevector.from_instruction(qc)
        probabilities_array = sv.probabilities()
        
        # Convert to dictionary format
        probabilities = {}
        for m, prob in enumerate(probabilities_array):
            probabilities[m] = float(prob)
        
        return probabilities
    
    # Add measurements for shot-based simulation
    qc.measure_all()
    
    # Transpile for backend
    qc_transpiled = transpile(qc, backend=backend_obj)
    
    # Execute based on backend type
    if use_primitives:
        # Use Sampler primitive for IBM hardware
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        sampler = Sampler(mode=backend_obj)
        job = sampler.run([qc_transpiled], shots=num_shots)
        result = job.result()
        # SamplerV2 returns a PubResult
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
    else:
        # Use legacy run() for simulators
        job = backend_obj.run(qc_transpiled, shots=num_shots)
        result = job.result()
        counts = result.get_counts()
    
    # Convert counts to probabilities
    # counts is dict like {'0101': 420, '1001': 380, ...}
    # We need {5: 0.0512, 9: 0.0463, ...} where keys are decimal indices
    probabilities = {}
    for bitstring, count in counts.items():
        # Convert bitstring to integer (little-endian: rightmost = qubit 0)
        m = int(bitstring, 2)
        probabilities[m] = count / num_shots
    
    # Fill in zero probabilities for unmeasured states
    M = 2**num_qubits
    for m in range(M):
        if m not in probabilities:
            probabilities[m] = 0.0
    
    return probabilities
