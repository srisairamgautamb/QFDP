"""
IBM Quantum Hardware Integration
=================================

Enables running qfdp_multiasset circuits on real IBM Quantum hardware.
"""

from .ibm_runner import IBMQuantumRunner, run_on_hardware

__all__ = ['IBMQuantumRunner', 'run_on_hardware']
