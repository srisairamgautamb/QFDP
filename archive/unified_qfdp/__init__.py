"""
Unified QFDP: Combined FB-QDP + Multi-Asset
============================================

Consolidates:
- FB-QDP base model (factor-based pricing, O(NK) gates)
- qfdp_multiasset (sparse copula, MLQAE k>0, hardware)

Unified modules:
- state_prep: Best state preparation from both
- iqft: Enhanced IQFT (combining tensor IQFT + base IQFT)
- sparse_copula: Adaptive factor decomposition
- mlqae: k>0 amplitude estimation
- hardware: IBM Quantum integration
- portfolio: Multi-asset pricing
- risk: VaR/CVaR computation
"""

# Re-export from qfdp_multiasset (primary)
from qfdp_multiasset.sparse_copula import FactorDecomposer
from qfdp_multiasset.state_prep import prepare_lognormal_asset
from qfdp_multiasset.oracles import apply_call_payoff_rotation
from qfdp_multiasset.mlqae import run_mlqae
from qfdp_multiasset.risk import compute_var_cvar_mc
from qfdp_multiasset.hardware import IBMQuantumRunner, run_on_hardware

__version__ = "1.0.0-unified"
__all__ = [
    'FactorDecomposer',
    'prepare_lognormal_asset',
    'apply_call_payoff_rotation',
    'run_mlqae',
    'compute_var_cvar_mc',
    'IBMQuantumRunner',
    'run_on_hardware',
]
