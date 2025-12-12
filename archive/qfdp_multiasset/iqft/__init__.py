"""
Tensor QFT/IQFT Module
======================

Provides utilities to apply QFT/IQFT across asset registers.

Functions:
- build_qft
- apply_qft
- apply_tensor_qft
- estimate_qft_resources
- estimate_tensor_qft_resources
"""

from .tensor_iqft import (
    build_qft,
    apply_qft,
    apply_tensor_qft,
    estimate_qft_resources,
    estimate_tensor_qft_resources,
)

__all__ = [
    'build_qft',
    'apply_qft',
    'apply_tensor_qft',
    'estimate_qft_resources',
    'estimate_tensor_qft_resources',
]