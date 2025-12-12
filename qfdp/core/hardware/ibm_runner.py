"""
IBM Quantum Hardware Runner
============================

Run qfdp_multiasset quantum circuits on real IBM Quantum hardware or simulators.

Features:
- Automatic backend selection (real hardware or simulator)
- Shot-based sampling for MLQAE
- Transpilation with optimization
- Result processing and error handling

Author: QFDP Research Team
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from qiskit import QuantumCircuit, transpile

# Try to import Aer components, fall back to StatevectorSampler
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import Sampler
    HAS_AER = True
except ImportError:
    from qiskit.primitives import StatevectorSampler as Sampler
    from qiskit.providers.basic_provider import BasicSimulator as AerSimulator
    HAS_AER = False
    # Suppress warning for cleaner output

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    HAS_IBM_RUNTIME = True
except ImportError:
    HAS_IBM_RUNTIME = False
    print("⚠️  qiskit-ibm-runtime not available, hardware execution disabled")


@dataclass
class HardwareResult:
    """Results from IBM Quantum hardware execution."""
    counts: Dict[str, int]
    shots: int
    backend_name: str
    transpiled_depth: int
    transpiled_gates: int
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class IBMQuantumRunner:
    """
    Runner for executing quantum circuits on IBM Quantum hardware.
    
    Examples
    --------
    >>> runner = IBMQuantumRunner(backend_name='ibm_fez')
    >>> result = runner.run(circuit, shots=1024)
    >>> print(f"Counts: {result.counts}")
    
    >>> # Or use simulator
    >>> runner = IBMQuantumRunner(use_simulator=True)
    >>> result = runner.run(circuit, shots=1024)
    """
    
    def __init__(
        self,
        backend_name: Optional[str] = None,
        use_simulator: bool = False,
        optimization_level: int = 3,
        instance: Optional[str] = None
    ):
        """
        Initialize IBM Quantum runner.
        
        Parameters
        ----------
        backend_name : str, optional
            Name of IBM backend (e.g. 'ibm_fez', 'ibm_torino')
            If None, automatically selects least busy backend
        use_simulator : bool
            If True, uses AerSimulator instead of real hardware
        optimization_level : int
            Transpiler optimization level (0-3)
        instance : str, optional
            IBM Quantum instance (e.g. 'ibm-q/open/main')
        """
        self.use_simulator = use_simulator
        self.optimization_level = optimization_level
        
        if use_simulator:
            self.backend = AerSimulator()
            self.backend_name = 'aer_simulator'
            self.service = None
        else:
            # Initialize IBM Quantum service
            try:
                if instance:
                    self.service = QiskitRuntimeService(instance=instance)
                else:
                    self.service = QiskitRuntimeService()
                
                # Select backend
                if backend_name:
                    self.backend = self.service.backend(backend_name)
                    self.backend_name = backend_name
                else:
                    # Auto-select least busy backend (excluding ibm_fez)
                    available_backends = self.service.backends(
                        operational=True,
                        simulator=False
                    )
                    # Filter out ibm_fez
                    available_backends = [b for b in available_backends if b.name != 'ibm_fez']
                    
                    if available_backends:
                        # Get least busy from filtered list
                        self.backend = min(available_backends, key=lambda b: b.status().pending_jobs)
                        self.backend_name = self.backend.name
                    else:
                        raise RuntimeError("No available backends (all filtered out)")
                    
                print(f"✅ Connected to IBM Quantum: {self.backend_name}")
                print(f"   Qubits: {self.backend.num_qubits}")
                
            except Exception as e:
                print(f"⚠️  Failed to connect to IBM Quantum: {e}")
                print("   Falling back to AerSimulator")
                self.backend = AerSimulator()
                self.backend_name = 'aer_simulator'
                self.use_simulator = True
    
    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        optimize: bool = True
    ) -> HardwareResult:
        """
        Execute quantum circuit on IBM hardware or simulator.
        
        Parameters
        ----------
        circuit : QuantumCircuit
            Quantum circuit to execute
        shots : int
            Number of measurement shots
        optimize : bool
            Whether to apply transpiler optimization
            
        Returns
        -------
        HardwareResult
            Execution results with counts and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Add measurements if not present
            if not circuit.num_clbits:
                circuit.measure_all()
            
            # Transpile circuit
            if optimize:
                transpiled = transpile(
                    circuit,
                    backend=self.backend,
                    optimization_level=self.optimization_level,
                    seed_transpiler=42
                )
            else:
                transpiled = circuit
            
            # Execute
            if self.use_simulator:
                # Use local Sampler for simulator
                sampler = Sampler()
                if HAS_AER:
                    job = sampler.run(transpiled, shots=shots)
                    result = job.result()
                    counts = result.quasi_dists[0].binary_probabilities()
                    # Convert to integer counts
                    counts = {k: int(v * shots) for k, v in counts.items()}
                else:
                    # StatevectorSampler API
                    job = sampler.run([transpiled], shots=shots)
                    result = job.result()
                    counts = result[0].data.meas.get_counts()
            else:
                # Use IBM Runtime SamplerV2 for real hardware
                sampler = SamplerV2(mode=self.backend)
                job = sampler.run([transpiled], shots=shots)
                result = job.result()
                counts = result[0].data.meas.get_counts()
            
            execution_time = time.time() - start_time
            
            return HardwareResult(
                counts=counts,
                shots=shots,
                backend_name=self.backend_name,
                transpiled_depth=transpiled.depth(),
                transpiled_gates=transpiled.size(),
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return HardwareResult(
                counts={},
                shots=shots,
                backend_name=self.backend_name,
                transpiled_depth=0,
                transpiled_gates=0,
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def estimate_ancilla_probability(
        self,
        circuit: QuantumCircuit,
        ancilla_index: int,
        shots: int = 1024
    ) -> Tuple[float, float]:
        """
        Estimate probability of ancilla qubit being |1⟩.
        
        This is the key measurement for MLQAE pricing.
        
        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit with ancilla encoding payoff
        ancilla_index : int
            Index of ancilla qubit to measure
        shots : int
            Number of measurement shots
            
        Returns
        -------
        prob : float
            Estimated probability P(ancilla=1)
        std_error : float
            Standard error of estimate
        """
        result = self.run(circuit, shots=shots)
        
        if not result.success:
            raise RuntimeError(f"Execution failed: {result.error_message}")
        
        # Count shots where ancilla is |1⟩
        ancilla_ones = 0
        for bitstring, count in result.counts.items():
            # Check ancilla bit (reverse indexing in Qiskit)
            if len(bitstring) > ancilla_index and bitstring[-(ancilla_index+1)] == '1':
                ancilla_ones += count
        
        prob = ancilla_ones / result.shots
        std_error = np.sqrt(prob * (1 - prob) / result.shots)
        
        return prob, std_error
    
    def available_backends(self) -> List[str]:
        """Get list of available IBM Quantum backends."""
        if self.service is None:
            return ['aer_simulator']
        
        backends = self.service.backends(simulator=False, operational=True)
        return [b.name for b in backends]
    
    def backend_info(self) -> Dict:
        """Get information about current backend."""
        info = {
            'name': self.backend_name,
            'is_simulator': self.use_simulator,
        }
        
        if not self.use_simulator and hasattr(self.backend, 'num_qubits'):
            info.update({
                'num_qubits': self.backend.num_qubits,
                'basis_gates': getattr(self.backend, 'basis_gates', []),
                'coupling_map': bool(getattr(self.backend, 'coupling_map', None)),
            })
        
        return info


def run_on_hardware(
    circuit: QuantumCircuit,
    backend_name: Optional[str] = None,
    shots: int = 1024,
    use_simulator: bool = False
) -> HardwareResult:
    """
    Convenience function to run circuit on IBM hardware.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to execute
    backend_name : str, optional
        IBM backend name
    shots : int
        Number of shots
    use_simulator : bool
        Use simulator instead of real hardware
        
    Returns
    -------
    HardwareResult
        Execution results
        
    Examples
    --------
    >>> result = run_on_hardware(circuit, 'ibm_fez', shots=2048)
    >>> print(f"Executed on {result.backend_name} in {result.execution_time:.2f}s")
    """
    runner = IBMQuantumRunner(backend_name=backend_name, use_simulator=use_simulator)
    return runner.run(circuit, shots=shots)
