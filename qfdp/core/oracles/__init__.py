"""
Oracle Module
=============

Provides payoff and characteristic-function oracles.

Phase 5 initial:
- Digital payoff (threshold) oracle for P(S >= K)
"""

from .payoff_oracle import (
    mark_threshold_states,
    ancilla_probability,
    classical_threshold_probability,
    call_payoff,
    apply_call_payoff_rotation,
    ancilla_scaled_expectation,
    direct_expected_call_from_statevector,
)

from .piecewise_payoff import (
    segment_payoff,
    compute_segment_indices,
    apply_piecewise_constant_payoff,
    piecewise_approximation_error,
)

__all__ = [
    'mark_threshold_states',
    'ancilla_probability',
    'classical_threshold_probability',
    'call_payoff',
    'apply_call_payoff_rotation',
    'ancilla_scaled_expectation',
    'direct_expected_call_from_statevector',
    'segment_payoff',
    'compute_segment_indices',
    'apply_piecewise_constant_payoff',
    'piecewise_approximation_error',
]
