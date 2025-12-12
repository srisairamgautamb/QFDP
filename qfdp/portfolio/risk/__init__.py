"""
Risk Management Module
=======================

Real VaR/CVaR calculation via Monte Carlo simulation.

NO shortcuts. NO approximations. ONLY real simulated paths.
"""

from .monte_carlo_var import (
    compute_var_cvar_mc,
    analytical_var_single_asset,
    VaRCVaRResult
)

__all__ = [
    'compute_var_cvar_mc',
    'analytical_var_single_asset',
    'VaRCVaRResult'
]
