"""
Quantum Temporal Convolution (QTC) - Main Implementation
=========================================================

Quantum convolutional network for extracting temporal patterns from price history.

Architecture:
    1. Sliding Window Kernels: 4 kernels, each processing 3 consecutive prices
    2. Quantum Feature Extraction: Each kernel runs a parameterized quantum circuit
    3. Global Pooling: Aggregate kernel outputs
    4. Deep Processing: Additional quantum layer for final pattern features

Input: Price history [S(t-5), S(t-4), S(t-3), S(t-2), S(t-1), S(t)]
Output: Pattern features capturing momentum, volatility, trends

Kernel Layout:
    Kernel 1: [t-5, t-4, t-3] - Early pattern
    Kernel 2: [t-4, t-3, t-2] - Middle pattern
    Kernel 3: [t-3, t-2, t-1] - Recent pattern
    Kernel 4: [t-2, t-1, t]   - Current pattern
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

try:
    from qiskit_aer import AerSimulator
    USE_AER = True
except ImportError:
    USE_AER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QTCResult:
    """Result from QTC forward pass."""
    patterns: np.ndarray         # Extracted pattern features
    kernel_outputs: List[np.ndarray]  # Individual kernel outputs
    circuit_depth: int           # Average circuit depth


class QuantumTemporalConvolution:
    """
    Quantum Temporal Convolution for price pattern extraction.
    
    Uses parameterized quantum circuits as convolutional kernels to
    extract temporal patterns from asset price history.
    
    Pipeline Position:
        Price History → [QTC] → Pattern Features → [Fusion] → [FB-IQFT]
    
    Attributes:
        kernel_size: Size of each convolutional window (default 3)
        n_kernels: Number of sliding kernels (default 4)
        n_qubits: Qubits per kernel (default 4)
        n_layers: Deep layers per kernel (default 3)
    
    Example:
        >>> qtc = QuantumTemporalConvolution()
        >>> prices = np.array([100.0, 100.5, 99.8, 100.2, 101.0, 100.8])
        >>> result = qtc.forward(prices)
        >>> print(f"Patterns: {result.patterns}")
    """
    
    def __init__(
        self,
        kernel_size: int = 3,
        n_kernels: int = 4,
        n_qubits: int = 4,
        n_layers: int = 3,
        n_shots: int = 100
    ):
        """
        Initialize QTC with random kernel parameters.
        
        Args:
            kernel_size: Prices per kernel window
            n_kernels: Number of sliding kernels
            n_qubits: Qubits per kernel circuit
            n_layers: Deep layers per kernel
            n_shots: Measurement shots
        """
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_shots = n_shots
        
        # Trainable parameters for each kernel
        # Structure: {kernel_idx: {theta: array, phi: array}}
        self.kernel_params = {}
        for k in range(n_kernels):
            self.kernel_params[k] = {
                'theta': np.random.randn(n_layers, n_qubits) * 0.1,
                'phi': np.random.randn(n_layers, n_qubits) * 0.1
            }
        
        # Global pooling layer parameters
        self.pool_weights = np.random.randn(n_kernels) * 0.1
        
        logger.info(
            f"QTC initialized: {n_kernels} kernels, size {kernel_size}, "
            f"{n_qubits} qubits, {n_layers} layers"
        )
    
    def forward(self, price_history: np.ndarray) -> QTCResult:
        """
        Extract temporal patterns from price history.
        
        Args:
            price_history: Array of prices [S(t-n), ..., S(t)]
                          Minimum length: kernel_size + n_kernels - 1
        
        Returns:
            QTCResult with extracted pattern features
        """
        # Normalize prices
        if len(price_history) < self.kernel_size:
            logger.warning(f"Price history too short: {len(price_history)}")
            return QTCResult(
                patterns=np.ones(self.n_qubits) / self.n_qubits,
                kernel_outputs=[],
                circuit_depth=0
            )
        
        # Z-score normalization
        mean_price = np.mean(price_history)
        std_price = np.std(price_history)
        if std_price < 1e-8:
            std_price = 1.0
        normalized = (price_history - mean_price) / std_price
        
        # Extract features from each sliding window
        kernel_outputs = []
        total_depth = 0
        
        for kernel_idx in range(self.n_kernels):
            window_start = kernel_idx
            window_end = kernel_idx + self.kernel_size
            
            if window_end <= len(normalized):
                window = normalized[window_start:window_end]
                feature, depth = self._process_kernel(window, kernel_idx)
                kernel_outputs.append(feature)
                total_depth += depth
        
        # Global pooling: weighted average of kernel outputs
        if kernel_outputs:
            # Stack and apply learned weights
            stacked = np.array(kernel_outputs)
            weights = self._softmax(self.pool_weights[:len(kernel_outputs)])
            pooled = np.sum(stacked * weights[:, np.newaxis], axis=0)
            
            # Deep processing layer
            patterns = self._deep_processing(pooled)
        else:
            patterns = np.ones(self.n_qubits) / self.n_qubits
        
        avg_depth = total_depth // max(len(kernel_outputs), 1)
        
        return QTCResult(
            patterns=patterns,
            kernel_outputs=kernel_outputs,
            circuit_depth=avg_depth
        )
    
    def build_circuits(self, price_history: np.ndarray) -> List[QuantumCircuit]:
        """
        Build all kernel circuits WITHOUT executing them.
        
        This method is for HARDWARE execution - returns circuits that can
        be transpiled and run on IBM Quantum or any backend.
        
        Args:
            price_history: Array of prices [S(t-n), ..., S(t)]
        
        Returns:
            List of QuantumCircuit objects (one per kernel)
        """
        circuits = []
        
        # Normalize prices
        if len(price_history) < self.kernel_size:
            return circuits
        
        mean_price = np.mean(price_history)
        std_price = np.std(price_history)
        if std_price < 1e-8:
            std_price = 1.0
        normalized = (price_history - mean_price) / std_price
        
        for kernel_idx in range(self.n_kernels):
            window_start = kernel_idx
            window_end = kernel_idx + self.kernel_size
            
            if window_end <= len(normalized):
                window = normalized[window_start:window_end]
                qc = self._build_kernel_circuit(window, kernel_idx)
                circuits.append(qc)
        
        # Store for later use
        self._last_circuits = circuits
        return circuits
    
    def _build_kernel_circuit(self, window: np.ndarray, kernel_idx: int) -> QuantumCircuit:
        """
        Build a single kernel circuit WITHOUT measurement execution.
        
        Args:
            window: Normalized price window
            kernel_idx: Kernel index
            
        Returns:
            QuantumCircuit ready for hardware execution
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Initialize superposition
        for i in range(self.n_qubits):
            qc.h(i)
        qc.barrier()
        
        # Encode window prices as rotations
        for i, price in enumerate(window):
            if i < self.n_qubits:
                price_clipped = np.clip(price, -1, 1)
                angle = np.arcsin(price_clipped) + np.pi
                qc.ry(angle, i)
        qc.barrier()
        
        # Apply learnable layers
        params = self.kernel_params[kernel_idx]
        for layer_idx in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.ry(params['theta'][layer_idx, q], q)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            for q in range(self.n_qubits):
                qc.rz(params['phi'][layer_idx, q], q)
            qc.barrier()
        
        # Add measurements
        qc.measure_all()
        
        return qc
    
    def forward_with_counts(self, kernel_counts: List[Dict[str, int]]) -> QTCResult:
        """
        Process using externally-obtained measurement counts.
        
        This is for HARDWARE execution - run circuits on IBM Quantum,
        get counts, then call this method.
        
        Args:
            kernel_counts: List of count dictionaries (one per kernel)
            
        Returns:
            QTCResult with patterns extracted from hardware results
        """
        kernel_outputs = []
        
        for counts in kernel_counts:
            feature = self._counts_to_feature(counts)
            kernel_outputs.append(feature)
        
        # Global pooling
        if kernel_outputs:
            stacked = np.array(kernel_outputs)
            weights = self._softmax(self.pool_weights[:len(kernel_outputs)])
            pooled = np.sum(stacked * weights[:, np.newaxis], axis=0)
            patterns = self._deep_processing(pooled)
        else:
            patterns = np.ones(self.n_qubits) / self.n_qubits
        
        # Estimate depth from last built circuits
        avg_depth = 0
        if hasattr(self, '_last_circuits') and self._last_circuits:
            avg_depth = sum(c.depth() for c in self._last_circuits) // len(self._last_circuits)
        
        return QTCResult(
            patterns=patterns,
            kernel_outputs=kernel_outputs,
            circuit_depth=avg_depth
        )
    
    def _process_kernel(
        self,
        window: np.ndarray,
        kernel_idx: int
    ) -> Tuple[np.ndarray, int]:
        """
        Process single kernel window with quantum circuit.
        
        Args:
            window: Normalized price window [p1, p2, p3]
            kernel_idx: Index of kernel (0-3)
        
        Returns:
            feature: Extracted feature vector
            depth: Circuit depth
        """
        # Create quantum circuit
        qc = QuantumCircuit(self.n_qubits)
        
        # Initialize superposition
        for i in range(self.n_qubits):
            qc.h(i)
        qc.barrier()
        
        # Encode window prices as rotations
        for i, price in enumerate(window):
            if i < self.n_qubits:
                # Clip to [-1, 1] for arcsin
                price_clipped = np.clip(price, -1, 1)
                angle = np.arcsin(price_clipped) + np.pi
                qc.ry(angle, i)
        qc.barrier()
        
        # Apply learnable layers (same structure as QRC)
        params = self.kernel_params[kernel_idx]
        for layer_idx in range(self.n_layers):
            # RY rotations
            for q in range(self.n_qubits):
                qc.ry(params['theta'][layer_idx, q], q)
            
            # Entanglement
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            
            # RZ phases
            for q in range(self.n_qubits):
                qc.rz(params['phi'][layer_idx, q], q)
            
            qc.barrier()
        
        # Measure
        qc.measure_all()
        
        # Execute simulation
        if USE_AER:
            simulator = AerSimulator()
            result = simulator.run(qc, shots=self.n_shots).result()
            counts = result.get_counts()
        else:
            # Fallback: Use Statevector sampling
            qc_no_meas = QuantumCircuit(self.n_qubits)
            for instruction in qc.data:
                if instruction.operation.name != 'measure':
                    qc_no_meas.append(instruction.operation, instruction.qubits, instruction.clbits)
            sv = Statevector(qc_no_meas)
            counts = sv.sample_counts(shots=self.n_shots)
        
        # Extract feature from probabilities
        feature = self._counts_to_feature(counts)
        
        return feature, qc.depth()
    
    def _counts_to_feature(self, counts: Dict[str, int]) -> np.ndarray:
        """Convert measurement counts to feature vector."""
        total = sum(counts.values())
        feature = np.zeros(self.n_qubits)
        
        for bitstring, count in counts.items():
            # Use Hamming weight distribution
            n_ones = bitstring.count('1')
            feature[min(n_ones, self.n_qubits - 1)] += count / total
        
        # Normalize
        if feature.sum() > 0:
            feature = feature / feature.sum()
        else:
            feature = np.ones(self.n_qubits) / self.n_qubits
        
        return feature
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax for weighted pooling."""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / exp_x.sum()
    
    def _deep_processing(self, features: np.ndarray) -> np.ndarray:
        """
        Additional non-linear processing of pooled features.
        
        Simple activation for now; can be replaced with another
        quantum circuit for full quantum processing.
        """
        # Tanh activation for non-linearity
        processed = np.tanh(features * 2)
        
        # Normalize to probabilities
        processed = np.abs(processed)
        if processed.sum() > 0:
            processed = processed / processed.sum()
        
        return processed
    
    def get_parameters(self) -> Dict:
        """Get all trainable kernel parameters."""
        return {
            'kernel_params': {k: {
                'theta': v['theta'].copy(),
                'phi': v['phi'].copy()
            } for k, v in self.kernel_params.items()},
            'pool_weights': self.pool_weights.copy()
        }
    
    def set_parameters(self, params: Dict) -> None:
        """Set trainable parameters."""
        if 'kernel_params' in params:
            for k, v in params['kernel_params'].items():
                self.kernel_params[k] = {
                    'theta': v['theta'].copy(),
                    'phi': v['phi'].copy()
                }
        if 'pool_weights' in params:
            self.pool_weights = params['pool_weights'].copy()
    
    def __repr__(self) -> str:
        return (
            f"QuantumTemporalConvolution(n_kernels={self.n_kernels}, "
            f"kernel_size={self.kernel_size}, n_qubits={self.n_qubits})"
        )
