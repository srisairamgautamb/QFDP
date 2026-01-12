"""
Quantum Deep Pricing System
============================

Integrated system combining:
    - QRC (Quantum Recurrent Circuit) for adaptive factors
    - QTC (Quantum Temporal Convolution) for pattern features
    - Feature Fusion for combining both
    - FB-IQFT for final pricing

This creates a full quantum deep learning pipeline for derivative pricing.
"""

from .feature_fusion import FeatureFusion, FusionResult
from .integrated_pricer import QuantumDeepPricer, DeepPricingResult

__all__ = [
    'FeatureFusion',
    'FusionResult',
    'QuantumDeepPricer',
    'DeepPricingResult',
]

__version__ = '1.0.0'
