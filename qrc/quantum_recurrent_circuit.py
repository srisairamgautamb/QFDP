"""
Quantum Recurrent Circuit (QRC) - Main Implementation
======================================================

A quantum recurrent neural network for adaptive factor extraction in derivative pricing.

Architecture:
    1. Input Encoding: Market data → RY rotation angles
    2. Recurrent Gate: CNOT ladder + RZ phases for temporal memory
    3. Deep Layers: 3 layers of (RY rotations → CX entanglement → RZ phases)
    4. Measurement: Extract adaptive factors from probability distribution

Key Features:
    - 8 qubits (expandable)
    - 3 deep quantum layers
    - Hidden state persistence across timesteps
    - Learnable parameters (θ, φ angles)
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
class QRCResult:
    """Result from QRC forward pass."""
    factors: np.ndarray           # Extracted factors [F1, F2, F3, F4]
    hidden_state: np.ndarray      # Classical hidden state for next step
    circuit_depth: int            # Circuit depth (gates)
    n_shots: int                  # Shots used for measurement


class QuantumRecurrentCircuit:
    """
    Quantum Recurrent Circuit for adaptive factor extraction.
    
    Replaces static PCA in the FB-IQFT pipeline with a dynamic,
    regime-aware factor extraction mechanism.
    
    Pipeline Position:
        Market Data → [QRC] → Adaptive Factors → [FB-IQFT] → Price
    
    Attributes:
        n_factors: Number of factors to extract (default 4)
        n_qubits: Number of qubits (default 8)
        n_deep_layers: Number of deep quantum layers (default 3)
        theta: Learnable RY rotation parameters
        phi: Learnable RZ phase parameters
        h_classical: Classical hidden state from previous timestep
    
    Example:
        >>> qrc = QuantumRecurrentCircuit(n_factors=4, n_qubits=8)
        >>> market_data = {'prices': 100, 'volatility': 0.2, 'stress': 0.1}
        >>> result = qrc.forward(market_data)
        >>> print(f"Factors: {result.factors}")
    """
    
    def __init__(
        self,
        n_factors: int = 4,
        n_qubits: int = 8,
        n_deep_layers: int = 3,
        n_shots: int = 100
    ):
        """
        Initialize QRC with random parameters.
        
        Args:
            n_factors: Number of factors to extract
            n_qubits: Number of qubits in the circuit
            n_deep_layers: Number of deep processing layers
            n_shots: Measurement shots for factor extraction
        """
        self.n_factors = n_factors
        self.n_qubits = n_qubits
        self.n_deep_layers = n_deep_layers
        self.n_shots = n_shots
        
        # Trainable parameters: small random initialization for stability
        # Shape: [layer_idx, qubit_idx]
        self.theta = np.random.randn(n_deep_layers, n_qubits) * 0.1
        self.phi = np.random.randn(n_deep_layers, n_qubits) * 0.1
        
        # Recurrent gate parameters
        self.recurrent_weights = np.random.randn(n_qubits) * 0.1
        
        # Hidden state storage (classical representation)
        self.h_classical = np.ones(n_factors) / n_factors  # Uniform init
        self.h_quantum = None  # Store circuit for debugging
        
        logger.info(
            f"QRC initialized: {n_qubits} qubits, {n_deep_layers} layers, "
            f"{n_factors} factors, {self._count_parameters()} trainable params"
        )
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return (
            self.n_deep_layers * self.n_qubits * 2 +  # theta + phi
            self.n_qubits  # recurrent weights
        )
    
    def forward(self, market_data: Dict) -> QRCResult:
        """
        Process market data and return adaptive factors.
        
        This is the main entry point for the QRC. Takes current market
        conditions, mixes with previous hidden state, and extracts
        regime-aware factors for pricing.
        
        Args:
            market_data: Dictionary with market observations
                Required keys:
                    - 'prices': Asset price(s) (float or array)
                Optional keys:
                    - 'volatility': Current volatility (float, 0-1)
                    - 'vol_trend': Volatility change rate (float)
                    - 'corr_change': Correlation matrix change (float)
                    - 'stress': Market stress indicator (float, 0-1)
        
        Returns:
            QRCResult containing extracted factors and metadata
        """
        # Create quantum circuit
        qc = QuantumCircuit(self.n_qubits)
        
        # Step 1: Initialize superposition (blank canvas)
        for i in range(self.n_qubits):
            qc.h(i)
        qc.barrier()
        
        # Step 2: Encode market data as rotation angles
        self._encode_input(qc, market_data)
        qc.barrier()
        
        # Step 3: Recurrent interaction (mix with previous hidden state)
        self._recurrent_gate(qc)
        qc.barrier()
        
        # Step 4: Deep quantum layers (learnable processing)
        self._deep_layers(qc)
        qc.barrier()
        
        # Step 5: Measure to extract factors
        qc.measure_all()
        
        # Execute simulation
        if USE_AER:
            # Use AerSimulator if available
            simulator = AerSimulator()
            result = simulator.run(qc, shots=self.n_shots).result()
            counts = result.get_counts()
        else:
            # Fallback: Use Statevector and sample
            qc_no_meas = QuantumCircuit(self.n_qubits)
            # Copy gates without measurement
            for instruction in qc.data:
                if instruction.operation.name != 'measure':
                    qc_no_meas.append(instruction.operation, instruction.qubits, instruction.clbits)
            sv = Statevector(qc_no_meas)
            counts = sv.sample_counts(shots=self.n_shots)
        
        # Step 6: Extract factors from measurement statistics
        factors = self._extract_factors(counts)
        
        # Step 7: Update hidden state for next timestep
        self.h_classical = factors
        self.h_quantum = qc
        
        return QRCResult(
            factors=factors,
            hidden_state=self.h_classical.copy(),
            circuit_depth=qc.depth(),
            n_shots=self.n_shots
        )
    
    def _normalize_feature(self, x: float, method: str = 'tanh') -> float:
        """
        Normalize feature to [-1, 1] for quantum encoding.
        
        Args:
            x: Raw feature value
            method: 'tanh' (smooth) or 'clip' (hard)
        
        Returns:
            Normalized value in [-1, 1]
        """
        if method == 'tanh':
            return float(np.tanh(x))
        elif method == 'clip':
            return float(np.clip(x, -1, 1))
        else:
            raise ValueError(f"Unknown normalization: {method}")
    
    def _encode_input(self, qc: QuantumCircuit, market_data: Dict) -> None:
        """
        Encode market observation as quantum rotation angles.
        
        Maps features to RY rotation angles:
            - Larger values → larger rotations
            - Range: [-π/2, π/2] after arcsin transformation
        
        Qubit assignment:
            q0: Price level (normalized)
            q1: Volatility
            q2: Volatility trend
            q3: Correlation change
            q4: Market stress
            q5-q7: Auxiliary (fixed angles)
        """
        # Extract features with defaults
        prices = market_data.get('prices', 0)
        if isinstance(prices, np.ndarray):
            prices = float(np.mean(prices))  # Average if multiple assets
        
        volatility = market_data.get('volatility', 0.2)
        vol_trend = market_data.get('vol_trend', 0)
        corr_change = market_data.get('corr_change', 0)
        stress = market_data.get('stress', 0)
        
        # Normalize to [-1, 1]
        features = [
            self._normalize_feature(prices / 100 - 1),  # Centered around 100
            self._normalize_feature(volatility * 2 - 1),  # 0.5 → 0
            self._normalize_feature(vol_trend * 5),  # Scale small values
            self._normalize_feature(corr_change * 10),  # Scale small values
            self._normalize_feature(stress * 2 - 1),  # 0.5 → 0
        ]
        
        # Encode as RY rotations
        for i, feat in enumerate(features):
            if i < self.n_qubits:
                # arcsin maps [-1, 1] to [-π/2, π/2]
                # Add π to shift to [π/2, 3π/2] for full representation
                angle = np.arcsin(feat) + np.pi
                qc.ry(angle, i)
        
        # Auxiliary qubits: fixed angles for entanglement structure
        for i in range(len(features), self.n_qubits):
            qc.ry(np.pi / 4, i)
    
    def _recurrent_gate(self, qc: QuantumCircuit) -> None:
        """
        Recurrent gate: Mix current state with previous hidden state.
        
        Creates temporal continuity via:
            1. CNOT ladder: Entangles all qubits (quantum correlation)
            2. RZ phases: Modulates based on previous factors
        
        Effect: Information from previous timesteps influences current output
        """
        if self.h_quantum is None:
            # First call: no previous state to mix
            return
        
        # CNOT ladder: create entanglement between qubits
        # This spreads information across the register
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        
        # Phase modulation based on previous hidden state
        # Maps previous factor values to phase rotations
        for i in range(self.n_qubits):
            factor_idx = i % self.n_factors
            # Previous factor value determines phase
            phase_angle = self.h_classical[factor_idx] * 2 * np.pi
            # Additional learnable weight
            phase_angle *= (1 + self.recurrent_weights[i])
            qc.rz(phase_angle, i)
    
    def _deep_layers(self, qc: QuantumCircuit) -> None:
        """
        Apply 3 deep quantum layers for feature extraction.
        
        Each layer consists of:
            1. Rotation block: Learnable RY rotations (angles θ)
            2. Entanglement block: CNOT ladder for information mixing
            3. Phase block: Learnable RZ phases (angles φ)
        
        Why 3 layers?
            - 1 layer: Insufficient feature extraction
            - 2 layers: Limited expressivity
            - 3 layers: Good depth/noise tradeoff on NISQ
            - 4+ layers: Too much decoherence
        """
        for layer_idx in range(self.n_deep_layers):
            # Rotation block (learnable RY gates)
            for qubit_idx in range(self.n_qubits):
                angle = self.theta[layer_idx, qubit_idx]
                qc.ry(angle, qubit_idx)
            
            # Entanglement block (CNOT ladder)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            
            # Phase block (learnable RZ gates)
            for qubit_idx in range(self.n_qubits):
                angle = self.phi[layer_idx, qubit_idx]
                qc.rz(angle, qubit_idx)
            
            qc.barrier()
    
    def _extract_factors(self, counts: Dict[str, int]) -> np.ndarray:
        """
        Convert measurement counts to factors.
        
        Strategy:
            1. Normalize counts to probabilities
            2. Group bitstrings by first 2 bits → factor index
            3. Sum probabilities → factor values
            4. Normalize to sum = 1
        
        Args:
            counts: Measurement outcomes {bitstring: count}
        
        Returns:
            factors: Normalized factor array [F1, F2, F3, F4]
        """
        total_shots = sum(counts.values())
        
        # Initialize factors
        factors = np.zeros(self.n_factors)
        
        # Assign each bitstring to a factor based on first 2 bits
        for bitstring, count in counts.items():
            # Reverse bitstring (Qiskit convention: LSB first)
            bitstring_rev = bitstring[::-1]
            # First 2 bits determine factor index
            if len(bitstring_rev) >= 2:
                factor_idx = int(bitstring_rev[:2], 2) % self.n_factors
            else:
                factor_idx = 0
            factors[factor_idx] += count
        
        # Normalize to probabilities
        factors = factors / total_shots
        
        # Ensure sum = 1 (numerical stability)
        factor_sum = factors.sum()
        if factor_sum > 1e-10:
            factors = factors / factor_sum
        else:
            factors = np.ones(self.n_factors) / self.n_factors
        
        return factors
    
    def reset_hidden_state(self) -> None:
        """Reset hidden state for new sequence."""
        self.h_classical = np.ones(self.n_factors) / self.n_factors
        self.h_quantum = None
        logger.debug("QRC hidden state reset")
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all trainable parameters."""
        return {
            'theta': self.theta.copy(),
            'phi': self.phi.copy(),
            'recurrent_weights': self.recurrent_weights.copy()
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set trainable parameters (for training)."""
        if 'theta' in params:
            self.theta = params['theta'].copy()
        if 'phi' in params:
            self.phi = params['phi'].copy()
        if 'recurrent_weights' in params:
            self.recurrent_weights = params['recurrent_weights'].copy()
    
    def __repr__(self) -> str:
        return (
            f"QuantumRecurrentCircuit(n_factors={self.n_factors}, "
            f"n_qubits={self.n_qubits}, n_deep_layers={self.n_deep_layers})"
        )
