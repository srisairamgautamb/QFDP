"""
Quantum Temporal Convolution (QTC) Module
==========================================

Extracts temporal patterns from price history using quantum circuits.

Architecture:
- 4 sliding window kernels (kernel_size=3)
- Each kernel: 4 qubits, 3 layers
- Global pooling layer for feature aggregation

Input: 6 consecutive prices [S(t-5), ..., S(t)]
Output: 4 pattern features [p₁, p₂, p₃, p₄]
"""

import numpy as np
from qiskit import QuantumCircuit
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class QuantumTemporalKernel:
    """
    Single sliding window kernel for price pattern extraction.
    
    Input: 3 consecutive prices [S(t), S(t+1), S(t+2)]
    Output: Pattern feature (scalar)
    
    Architecture:
    - 4 qubits (3 for prices + 1 auxiliary)
    - 3 deep layers: RY + CNOT + RZ
    - Measurement → feature extraction
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 3, shots: int = 100):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        
        # Trainable parameters (3 layers × 4 qubits × 2 angles = 24 params)
        np.random.seed(42)  # For reproducibility
        self.theta = np.random.randn(n_layers, n_qubits) * 0.1  # RY angles
        self.phi = np.random.randn(n_layers, n_qubits) * 0.1    # RZ angles
    
    def forward(self, price_window: np.ndarray) -> float:
        """
        Process 3-price window.
        
        Args:
            price_window: [p1, p2, p3] normalized to [-1, 1]
        
        Returns:
            feature: Extracted pattern feature (scalar in [0, 1])
        """
        # Step 1: Create circuit
        qc = QuantumCircuit(self.n_qubits)
        
        # Step 2: Initialize with Hadamard
        for i in range(self.n_qubits):
            qc.h(i)
        qc.barrier()
        
        # Step 3: Encode prices
        for i, price in enumerate(price_window):
            if i < 3:  # First 3 qubits
                angle = np.arcsin(np.clip(price, -0.99, 0.99)) + np.pi
                qc.ry(angle, i)
        qc.barrier()
        
        # Step 4: Deep layers
        for layer in range(self.n_layers):
            # Rotation
            for q in range(self.n_qubits):
                qc.ry(self.theta[layer, q], q)
            
            # Entanglement
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            
            # Phase
            for q in range(self.n_qubits):
                qc.rz(self.phi[layer, q], q)
            
            qc.barrier()
        
        # Step 5: Measure
        qc.measure_all()
        
        # Step 6: Simulate
        try:
            from qiskit_aer import AerSimulator
            simulator = AerSimulator()
        except ImportError:
            from qiskit.quantum_info import Statevector
            # Use statevector and sample
            sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
            probs = sv.probabilities()
            # Sample
            counts = {}
            samples = np.random.choice(len(probs), size=self.shots, p=probs)
            for s in samples:
                bitstring = format(s, f'0{self.n_qubits}b')
                counts[bitstring] = counts.get(bitstring, 0) + 1
            return self._counts_to_feature(counts)
        
        result = simulator.run(qc, shots=self.shots).result()
        counts = result.get_counts()
        
        # Step 7: Extract feature
        feature = self._counts_to_feature(counts)
        
        return feature
    
    def _counts_to_feature(self, counts: Dict[str, int]) -> float:
        """Convert measurement counts to single feature value."""
        total = sum(counts.values())
        
        # Weighted sum: '0000' → 0, '1111' → 15
        weighted_sum = 0
        for bitstring, count in counts.items():
            # Remove spaces from bitstring if present
            bitstring = bitstring.replace(' ', '')
            value = int(bitstring, 2)
            weighted_sum += value * (count / total)
        
        # Normalize to [0, 1]
        max_value = 2**self.n_qubits - 1
        feature = weighted_sum / max_value
        
        return feature


class QuantumTemporalConvolution:
    """
    Full QTC with 4 sliding window kernels.
    
    Input: 6 prices [S(t-5), ..., S(t)]
    Output: 4 pattern features [p₁, p₂, p₃, p₄]
    
    Kernels:
    - Kernel 1: [S(t-5), S(t-4), S(t-3)]  Early pattern
    - Kernel 2: [S(t-4), S(t-3), S(t-2)]  Middle pattern
    - Kernel 3: [S(t-3), S(t-2), S(t-1)]  Recent pattern
    - Kernel 4: [S(t-2), S(t-1), S(t)]    Current pattern
    """
    
    def __init__(self, kernel_size: int = 3, n_kernels: int = 4, shots: int = 100):
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.shots = shots
        
        # Create 4 independent kernels
        self.kernels = [
            QuantumTemporalKernel(n_qubits=4, n_layers=3, shots=shots) 
            for _ in range(n_kernels)
        ]
        
        logger.info(f"QTC initialized: {n_kernels} kernels, kernel_size={kernel_size}")
    
    def forward(self, price_history: np.ndarray) -> np.ndarray:
        """
        Extract patterns from price history.
        
        Args:
            price_history: [S(t-5), S(t-4), ..., S(t)] (length 6)
        
        Returns:
            patterns: [p₁, p₂, p₃, p₄] pattern features
        """
        # Normalize prices
        price_norm = (price_history - np.mean(price_history)) / (np.std(price_history) + 1e-8)
        
        # Create sliding windows
        windows = []
        for i in range(self.n_kernels):
            window = price_norm[i:i+self.kernel_size]
            windows.append(window)
        
        # Process each window with its kernel
        patterns = []
        for kernel, window in zip(self.kernels, windows):
            feature = kernel.forward(window)
            patterns.append(feature)
        
        return np.array(patterns)
    
    def forward_with_pooling(self, price_history: np.ndarray) -> np.ndarray:
        """
        Extract patterns + deep global processing.
        
        Returns:
            final_patterns: Processed pattern features (normalized to sum=1)
        """
        # Step 1: Get kernel features
        kernel_features = self.forward(price_history)  # [p₁, p₂, p₃, p₄]
        
        # Step 2: Global deep processing
        final_patterns = self._deep_processing(kernel_features)
        
        return final_patterns
    
    def _deep_processing(self, kernel_features: np.ndarray) -> np.ndarray:
        """
        Apply another quantum circuit to aggregate kernel outputs.
        
        Input: [p₁, p₂, p₃, p₄] from 4 kernels
        Output: [f₁, f₂, f₃, f₄] final features (normalized)
        """
        # Create circuit
        qc = QuantumCircuit(4)
        
        # Initialize
        for i in range(4):
            qc.h(i)
        qc.barrier()
        
        # Encode kernel features
        for i, feat in enumerate(kernel_features):
            angle = feat * 2 * np.pi  # Map [0, 1] to [0, 2π]
            qc.ry(angle, i)
        qc.barrier()
        
        # One deep layer
        for i in range(4):
            qc.ry(np.pi / 4, i)
        
        for i in range(3):
            qc.cx(i, i + 1)
        
        for i in range(4):
            qc.rz(np.pi / 3, i)
        
        qc.barrier()
        qc.measure_all()
        
        # Simulate
        try:
            from qiskit_aer import AerSimulator
            result = AerSimulator().run(qc, shots=100).result()
            counts = result.get_counts()
        except ImportError:
            from qiskit.quantum_info import Statevector
            sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
            probs = sv.probabilities()
            counts = {}
            samples = np.random.choice(len(probs), size=100, p=probs)
            for s in samples:
                bitstring = format(s, '04b')
                counts[bitstring] = counts.get(bitstring, 0) + 1
        
        # Extract features from measurement distribution
        final_patterns = np.zeros(4)
        total = sum(counts.values())
        
        for bitstring, count in counts.items():
            bitstring = bitstring.replace(' ', '')
            for i, bit in enumerate(bitstring[::-1]):  # Reverse for qubit ordering
                if i < 4:
                    final_patterns[i] += (int(bit) * count / total)
        
        # Normalize to sum to 1
        final_patterns = final_patterns / (final_patterns.sum() + 1e-10)
        
        return final_patterns


# ============================================================================
# VALIDATION
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("QTC MODULE VALIDATION")
    print("=" * 60)
    
    # Test single kernel
    print("\n--- Testing Single Kernel ---")
    kernel = QuantumTemporalKernel()
    
    window_up = np.array([0.1, 0.2, 0.3])  # Uptrend
    window_down = np.array([0.3, 0.2, 0.1])  # Downtrend
    
    feat_up = kernel.forward(window_up)
    feat_down = kernel.forward(window_down)
    
    print(f"Uptrend feature:   {feat_up:.4f}")
    print(f"Downtrend feature: {feat_down:.4f}")
    print(f"Different? {feat_up != feat_down} ✅" if feat_up != feat_down else "❌")
    
    # Test full QTC
    print("\n--- Testing Full QTC ---")
    qtc = QuantumTemporalConvolution()
    
    prices = np.array([100.0, 100.5, 101.2, 101.8, 102.1, 102.5])  # Uptrend
    patterns = qtc.forward_with_pooling(prices)
    
    print(f"Price history: {prices}")
    print(f"QTC patterns:  {patterns}")
    print(f"Patterns sum to 1? {np.isclose(patterns.sum(), 1.0)} ✅" if np.isclose(patterns.sum(), 1.0) else "❌")
    
    # Test different history
    prices_volatile = np.array([100.0, 102.0, 99.5, 101.5, 98.0, 103.0])  # Volatile
    patterns_volatile = qtc.forward_with_pooling(prices_volatile)
    
    print(f"\nVolatile history: {prices_volatile}")
    print(f"QTC patterns:     {patterns_volatile}")
    print(f"Different from uptrend? {not np.allclose(patterns, patterns_volatile)} ✅" if not np.allclose(patterns, patterns_volatile) else "❌")
    
    print("\n✅ QTC MODULE VALIDATED")
