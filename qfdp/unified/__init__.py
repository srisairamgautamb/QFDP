"""
FB-IQFT: Factor-Based Inverse Quantum Fourier Transform for Derivative Pricing

This package implements the complete QFDP (Quantum Fourier Derivative Pricing)
pipeline using factor decomposition to enable shallow IQFT circuits suitable for
NISQ hardware.

Modules:
    - carr_madan_factor: Carr-Madan Fourier pricing setup (Steps 5-7)
    - frequency_encoding: Quantum state preparation (Step 8)
    - iqft_application: IQFT and measurement (Steps 9-10)
    - calibration: Normalization and price reconstruction (Steps 11-12)
    - fb_iqft_pricing: Main pricing pipeline (Steps 1-12)

Key Innovation:
    Factor decomposition enables Gaussian characteristic function, which requires
    only M=16-32 frequency points (vs M=256-1024 for general multi-asset CF).
    This yields shallow IQFT depth O(k²) ≈ 16-25 (vs 64-100 for standard QFDP).

Example:
    >>> from qfdp.unified import FBIQFTPricing
    >>> pricer = FBIQFTPricing(M=16, alpha=1.0, num_shots=8192)
    >>> result = pricer.price_option(
    ...     asset_prices=prices,
    ...     asset_volatilities=vols,
    ...     correlation_matrix=corr,
    ...     portfolio_weights=weights,
    ...     K=110.0,
    ...     T=1.0,
    ...     r=0.05,
    ...     backend='simulator'
    ... )
    >>> print(f"Option price: ${result['price_quantum']:.2f}")
    >>> print(f"Error: {result['error_percent']:.2f}%")
"""

from .carr_madan_gaussian import (
    compute_characteristic_function,
    apply_carr_madan_transform,
    setup_fourier_grid,
    classical_fft_baseline,
)

from .frequency_encoding import (
    encode_frequency_state,
    verify_encoding,
)

from .iqft_application import (
    apply_iqft,
    extract_strike_amplitudes,
)

from .calibration import (
    calibrate_quantum_to_classical,
    reconstruct_option_prices,
    validate_prices,
)

from .fb_iqft_pricing import FBIQFTPricing

__all__ = [
    # Carr-Madan module
    'compute_characteristic_function',
    'apply_carr_madan_transform',
    'setup_fourier_grid',
    'classical_fft_baseline',
    # Frequency encoding module
    'encode_frequency_state',
    'verify_encoding',
    # IQFT application module
    'apply_iqft',
    'extract_strike_amplitudes',
    # Calibration module
    'calibrate_quantum_to_classical',
    'reconstruct_option_prices',
    'validate_prices',
    # Main pricing class
    'FBIQFTPricing',
]

__version__ = '1.0.0'
__author__ = 'QFDP Team'
__description__ = 'Factor-Based IQFT for NISQ-Ready Quantum Derivative Pricing'
