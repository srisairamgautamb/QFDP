"""
Quantum Recurrent Circuit (QRC) for Adaptive Factor Extraction
===============================================================

Replaces static PCA with a quantum recurrent network that:
1. Adapts to changing market conditions in real-time
2. Maintains temporal memory via quantum entanglement
3. Outputs regime-aware factors for FB-IQFT pricing

Key Innovation: Uses quantum entanglement as temporal "memory" 
              without collapsing quantum state between timesteps.
"""

from .quantum_recurrent_circuit import QuantumRecurrentCircuit, QRCResult
from .training import train_qrc, QRCTrainer

__all__ = [
    'QuantumRecurrentCircuit',
    'QRCResult',
    'train_qrc',
    'QRCTrainer',
]

__version__ = '1.0.0'
