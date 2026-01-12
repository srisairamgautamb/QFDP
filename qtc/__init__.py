"""
Quantum Temporal Convolution (QTC) for Price Pattern Extraction
================================================================

Extracts temporal patterns from price history using quantum circuits
with sliding window kernels.

Key Innovation: Quantum circuits as convolutional kernels that learn
               to recognize momentum, volatility clustering, and trends.
"""

from .quantum_temporal_conv import QuantumTemporalConvolution, QTCResult

__all__ = [
    'QuantumTemporalConvolution',
    'QTCResult',
]

__version__ = '1.0.0'
