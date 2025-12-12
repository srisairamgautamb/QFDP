# QFDP Research Paper Readiness - Implementation Roadmap
**Date**: November 19, 2025  
**Status**: Core validated, enhancements needed for publication

---

## Executive Summary

**What's Production-Ready NOW**:
- ✅ Real VaR/CVaR via Monte Carlo (100% validated, 35 tests passing)
- ✅ Market data integration
- ✅ Classical portfolio analytics
- ✅ Quantum circuit construction
- ✅ Single-asset option pricing

**What Needs Research-Grade Enhancement** (10-14 hours):
1. ⚠️ Copula reconstruction error (0.88 → target <0.3)
2. ⚠️ MLQAE k>0 for quantum speedup
3. ⚠️ True basket pricing (joint distribution)
4. ⚠️ N≥10 demonstrations

---

## Issue 1: Copula Reconstruction Error

### Current State
- **Error**: 0.8751 Frobenius norm
- **Variance**: 87.5% explained
- **Problem**: K=3 for N=5 is suboptimal (loses 12.5% variance)

### Root Cause
Fixed K selection doesn't adapt to correlation structure. N=5 with K=3 ratio (0.6) is too aggressive.

### Research-Grade Fix

**File**: `qfdp_multiasset/sparse_copula/factor_model.py`

**Add Method**:
```python
def auto_select_K(self, corr_matrix: np.ndarray, 
                  variance_threshold: float = 0.95,
                  error_threshold: float = 0.3) -> int:
    """
    Automatically select K to meet quality thresholds.
    
    Uses scree plot analysis: finds elbow where marginal variance
    explained drops below threshold.
    
    Parameters
    ----------
    corr_matrix : Correlation matrix
    variance_threshold : Min cumulative variance (default 0.95)
    error_threshold : Max Frobenius error (default 0.3)
    
    Returns
    -------
    K : Optimal number of factors
    
    Algorithm
    ---------
    1. Compute all eigenvalues
    2. Find K where cumvar ≥ threshold
    3. Verify Frobenius error < threshold
    4. If not, increment K and retry
    """
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    N = len(eigenvalues)
    cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    
    # Find minimum K for variance threshold
    K_min = np.argmax(cumvar >= variance_threshold) + 1
    
    # Verify reconstruction quality
    for K in range(K_min, N+1):
        L, D, metrics = self.fit(corr_matrix, K, validate=False)
        if metrics.frobenius_error < error_threshold:
            return K
    
    # Fallback: use all factors
    return N
```

**Update `fit()` Method**:
```python
def fit(self, corr_matrix, K=None, variance_threshold=0.95, 
        error_threshold=0.3):
    """If K is None, auto-select to meet thresholds."""
    if K is None:
        K = self.auto_select_K(corr_matrix, variance_threshold, error_threshold)
        print(f"Auto-selected K={K} (var={variance_threshold}, err<{error_threshold})")
    
    # Rest of existing code...
```

**Expected Results**:
- N=5: K=4 (variance >95%, error <0.3)
- N=10: K=5-6
- N=20: K=8-10

**Implementation Time**: 1-2 hours  
**Testing Time**: 30 minutes  
**Research Impact**: HIGH - directly addresses reviewer concern

---

## Issue 2: MLQAE k>0 (Quantum Speedup)

### Current State
- **k values**: [0] only
- **Speedup**: None (equivalent to classical MC)
- **Root cause**: `initialize()` not invertible

### Why This Matters
This is THE quantum advantage. Without k>0, it's not quantum computing, just quantum sampling.

### Research-Grade Fix

**New File**: `qfdp_multiasset/state_prep/invertible_prep.py`

```python
"""
Invertible Quantum State Preparation
====================================

Prepares log-normal distribution using ONLY RY rotations.
Enables proper Grover operator: Q = -AS₀A†Sχ

Key: No initialize() - only parametrized gates that can be inverted.

References:
- Grover & Rudolph (2002): Creating superpositions
- Tanaka et al. (2023): Invertible amplitude encoding
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from typing import Tuple, List
from scipy import stats

def compute_rotation_tree(
    target_probs: np.ndarray
) -> List[List[float]]:
    """
    Compute RY rotation angles for binary tree.
    
    Uses Grover-Rudolph decomposition:
    Level 0: P(left subtree) for root
    Level 1: P(left | parent) for each node at level 1
    ...
    
    Returns angles in tree structure for controlled-RY gates.
    """
    n = int(np.log2(len(target_probs)))
    angles = []
    
    for level in range(n):
        level_angles = []
        nodes_at_level = 2**level
        
        for k in range(nodes_at_level):
            # Indices for left/right children
            left_start = k * 2**(n-level)
            left_end = left_start + 2**(n-level-1)
            right_end = left_end + 2**(n-level-1)
            
            # Probabilities
            p_left = np.sum(target_probs[left_start:left_end])
            p_total = np.sum(target_probs[left_start:right_end])
            
            if p_total > 1e-10:
                # Angle for P(left | parent)
                theta = 2 * np.arcsin(np.sqrt(p_left / p_total))
            else:
                theta = 0.0
            
            level_angles.append(theta)
        
        angles.append(level_angles)
    
    return angles

def prepare_lognormal_invertible(
    S0: float, r: float, sigma: float, T: float,
    n_qubits: int
) -> Tuple[QuantumCircuit, np.ndarray]:
    """
    Prepare log-normal distribution with invertible circuit.
    
    Returns
    -------
    circuit : QuantumCircuit with ONLY RY/CRY gates
    prices : Price grid
    """
    # Target distribution
    mu = np.log(S0) + (r - 0.5*sigma**2)*T
    sig = sigma * np.sqrt(T)
    
    # Price grid
    M = 2**n_qubits
    x = np.linspace(mu - 4*sig, mu + 4*sig, M)
    prices = np.exp(x)
    
    # Target probabilities (log-normal CDF)
    cdf = stats.norm.cdf(x, loc=mu, scale=sig)
    probs = np.diff(cdf, prepend=0)
    probs = probs / np.sum(probs)  # Normalize
    
    # Compute rotation angles
    angle_tree = compute_rotation_tree(probs)
    
    # Build circuit
    qr = QuantumRegister(n_qubits, 'state')
    circuit = QuantumCircuit(qr)
    
    # Apply rotations level by level
    for level, angles_at_level in enumerate(angle_tree):
        for k, angle in enumerate(angles_at_level):
            if level == 0:
                # Root: unconditional RY
                circuit.ry(angle, qr[0])
            else:
                # Controlled-RY based on parent qubits
                control_qubits = list(range(level))
                target_qubit = level
                
                # Binary encoding of k determines control values
                control_state = format(k, f'0{level}b')
                
                # Multi-controlled-RY
                if level == 1:
                    if control_state == '0':
                        circuit.x(qr[0])
                        circuit.cry(angle, qr[0], qr[1])
                        circuit.x(qr[0])
                    else:
                        circuit.cry(angle, qr[0], qr[1])
                else:
                    # For level>1, use mcrx decomposition
                    # This is complex but invertible
                    apply_mcrx(circuit, control_qubits, target_qubit, 
                              angle, control_state)
    
    return circuit, prices

def build_grover_operator(
    A_circuit: QuantumCircuit,
    payoff_circuit: QuantumCircuit,
    ancilla_idx: int
) -> QuantumCircuit:
    """
    Build TRUE Grover operator: Q = -AS₀A†Sχ
    
    Requirements:
    - A must be invertible (RY gates only, no initialize)
    - Payoff must be uncomputed before A†
    
    Returns amplification operator for MLQAE.
    """
    n_qubits = A_circuit.num_qubits
    Q = QuantumCircuit(n_qubits + 1)  # +1 for ancilla
    
    # Sχ: Mark ancilla |1⟩
    Q.z(ancilla_idx)
    
    # Uncompute payoff (if payoff_circuit.inverse() exists)
    Q.compose(payoff_circuit.inverse(), inplace=True)
    
    # A†: Invert state preparation
    Q.compose(A_circuit.inverse(), inplace=True)
    
    # S₀: Mark |0...0⟩ state
    # Z gate on all qubits if all zero
    Q.x(range(n_qubits))
    Q.h(n_qubits - 1)
    Q.mcx(list(range(n_qubits-1)), n_qubits-1)
    Q.h(n_qubits - 1)
    Q.x(range(n_qubits))
    
    # A: Reapply state prep
    Q.compose(A_circuit, inplace=True)
    
    # Recompute payoff
    Q.compose(payoff_circuit, inplace=True)
    
    # Global phase
    Q.global_phase = np.pi
    
    return Q
```

**Update MLQAE**:
```python
def run_mlqae_with_amplification(
    a_circuit: QuantumCircuit,
    payoff_circuit: QuantumCircuit,
    ancilla: int,
    scale: float,
    grover_powers: List[int] = [0, 1, 2, 4, 8],
    shots_per_power: int = 100,
    seed: int = 42
):
    """
    REAL MLQAE with k>0.
    
    Key difference: Applies Q^k for k>0, not just Q^0.
    """
    measurements = []
    
    Q = build_grover_operator(a_circuit, payoff_circuit, ancilla)
    
    for k in grover_powers:
        circ = a_circuit.copy()
        circ.compose(payoff_circuit, inplace=True)
        
        # Apply Q^k
        for _ in range(k):
            circ.compose(Q, inplace=True)
        
        # Measure
        h_k = simulate_measurement_outcomes(circ, ancilla, shots_per_power, seed)
        measurements.append((k, shots_per_power, h_k))
    
    # MLE estimation (existing code)
    ...
```

**Expected Results**:
- k=0: Baseline (100 shots)
- k=1: Slight amplification
- k=4: Significant amplification (~√16 = 4× improvement)
- k=8: Near-optimal for most problems

**Implementation Time**: 3-4 hours (complex)  
**Testing Time**: 1-2 hours  
**Research Impact**: CRITICAL - this IS the quantum advantage

---

## Issue 3: True Basket Pricing

### Current State
Uses E[payoff | S₁] (marginal), not E[payoff | S₁,S₂,...,Sₙ] (joint).

### Why This Matters
Correlation impact on basket options is THE multi-asset effect.

### Research-Grade Fix

**New File**: `qfdp_multiasset/portfolio/basket_pricing_joint.py`

```python
def price_basket_joint(
    asset_params: List[Tuple],  # [(S0, r, σ, T), ...]
    correlation_matrix: np.ndarray,
    payoff: PortfolioPayoff,
    n_qubits_per_asset: int = 4,
    n_factors: int = 3
) -> MLQAEResult:
    """
    Price basket option with JOINT distribution.
    
    Key: Encodes payoff on |S₁⟩⊗|S₂⟩⊗...⊗|Sₙ⟩, not just |S₁⟩.
    
    Warning: Exponential in N. Practical limit: N≤4 with n=4 qubits.
    """
    N = len(asset_params)
    n_total = N * n_qubits_per_asset
    
    # Prepare N asset states (independently)
    asset_circuits = []
    price_grids = []
    for S0, r, sigma, T in asset_params:
        circ, prices = prepare_lognormal_invertible(
            S0, r, sigma, T, n_qubits_per_asset
        )
        asset_circuits.append(circ)
        price_grids.append(prices)
    
    # Encode correlation (sparse copula)
    decomposer = FactorDecomposer()
    L, D, metrics = decomposer.fit(correlation_matrix, K=n_factors)
    
    corr_circuit = encode_sparse_copula_with_decomposition(
        asset_circuits, L, D, n_qubits_factor=2
    )
    
    # Encode JOINT basket payoff
    M = 2**n_qubits_per_asset
    all_states = itertools.product(range(M), repeat=N)
    
    payoffs = np.zeros(M**N)
    for i, state_indices in enumerate(all_states):
        basket_value = sum(
            payoff.weights[j] * price_grids[j][state_indices[j]]
            for j in range(N)
        )
        payoffs[i] = max(basket_value - payoff.strike, 0)
    
    scale = payoffs.max()
    
    # Multi-controlled payoff encoding
    anc = QuantumRegister(1, 'anc')
    corr_circuit.add_register(anc)
    
    for i, p in enumerate(payoffs):
        if p > 0:
            angle = 2 * np.arcsin(np.sqrt(p / scale))
            # Encode: if state is |i⟩, rotate ancilla by angle
            binary_controls = format(i, f'0{n_total}b')
            apply_multicontrolled_ry(
                corr_circuit, binary_controls, anc[0], angle
            )
    
    # MLQAE
    return run_mlqae(corr_circuit, anc[0], scale, ...)
```

**Validation Test**:
```python
def test_correlation_impact():
    """Basket price MUST differ for ρ=0 vs ρ=0.9."""
    params = [(100, 0.05, 0.2, 1.0)] * 2
    weights = [0.5, 0.5]
    strike = 100
    
    # Uncorrelated
    price_uncorr = price_basket_joint(
        params, np.eye(2), 
        PortfolioPayoff('basket', weights, strike)
    ).price_estimate
    
    # Highly correlated
    corr_high = np.array([[1.0, 0.9], [0.9, 1.0]])
    price_corr = price_basket_joint(
        params, corr_high,
        PortfolioPayoff('basket', weights, strike)
    ).price_estimate
    
    # MUST be different
    assert abs(price_uncorr - price_corr) / price_uncorr > 0.05
```

**Implementation Time**: 2-3 hours  
**Testing Time**: 1 hour  
**Research Impact**: HIGH - shows multi-asset correlation modeling

---

## Issue 4: N≥10 Demonstrations

### Quick Win
Create demo scripts showing clear gate advantage.

**File**: `demo_10_asset_sparse_advantage.py`
```python
"""
10-Asset Portfolio: Sparse Copula Advantage
===========================================

Demonstrates O(N×K) advantage over O(N²).

Portfolio: 10 tech/finance stocks
K=3 factors → 30 gates vs 45 gates (1.5× reduction)
"""

tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'JPM', 'BAC', 'GS', 'WFC', 'C'
]

# Build portfolio, show:
# 1. Correlation structure
# 2. Factor decomposition (K=3)
# 3. Gate count: 30 vs 45 ✅
# 4. Variance explained: >90%
# 5. VaR/CVaR for full portfolio
```

**File**: `demo_20_asset_sparse_advantage.py`
```python
"""
20-Asset Portfolio: Maximum Sparse Advantage
============================================

K=5 factors → 100 gates vs 190 gates (1.9× reduction)
"""
# Similar structure, larger N
```

**Implementation Time**: 1 hour (straightforward)  
**Research Impact**: MEDIUM - proves scaling claims

---

## IBM Quantum Integration (For 10-Minute Validation)

### When You Have IBM API

**File**: `ibm_quantum_validation.py`
```python
"""
IBM Quantum Hardware Validation
================================

Uses 10 minutes to validate:
1. Circuit compilation works
2. Real hardware matches simulation
3. Quantum advantage exists (if k>0 implemented)
"""

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# Setup
service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_TOKEN")
backend = service.least_busy(operational=True, simulator=False)

# Test 1: Single-asset option (small circuit)
circuit, prices = prepare_lognormal_invertible(100, 0.05, 0.2, 1.0, n_qubits=6)
# ... encode payoff ...

# Simulate vs Real
simulator_result = run_on_simulator(circuit)
hardware_result = run_on_hardware(circuit, backend)

print(f"Sim: ${simulator_result:.2f}")
print(f"IBM: ${hardware_result:.2f}")
print(f"Error: {abs(simulator_result - hardware_result)/simulator_result*100:.1f}%")

# Test 2: MLQAE with k>0 (if implemented)
if k_greater_than_zero_available:
    mlqae_test()
```

**What to Report to Reviewers**:
1. Circuit executed successfully on IBM hardware ✅
2. Results match simulation within error bounds ✅
3. (If k>0 done) Quantum speedup observed ✅

---

## Implementation Priority

### For Research Paper Submission

**Critical (Must Have)**:
1. ✅ VaR/CVaR (DONE - 100% validated)
2. ⚠️ Copula error <0.3 (1-2 hours)
3. ⚠️ MLQAE k>0 (3-4 hours) - **THIS IS THE QUANTUM ADVANTAGE**
4. ⚠️ N=10,20 demos (1 hour)

**Important (Should Have)**:
5. ⚠️ True basket pricing (2-3 hours)

**Timeline**: 7-10 hours for critical items

### Recommendation

Focus on items 2, 3, 4 first. Item 3 (MLQAE k>0) is THE differentiator between "quantum sampling" and "quantum computing".

---

## Current System Strengths (For Paper)

### What You CAN Claim:
1. ✅ **Real VaR/CVaR via Monte Carlo** (35 tests, all passing)
2. ✅ **Sparse copula mathematics** (correct, just needs K optimization)
3. ✅ **Market data integration** (Alpha Vantage, caching)
4. ✅ **Quantum state preparation** (working, needs invertibility)
5. ✅ **Complete integration** (all components work together)

### What You CANNOT Claim (Yet):
1. ❌ Quantum speedup (k=0 limitation)
2. ❌ Optimal copula reconstruction (error 0.88 > target 0.3)
3. ❌ True multi-asset basket pricing (marginal approximation)

---

## Conclusion

**System Status**: Production-ready for portfolio management, needs enhancements for research publication.

**Critical Path**: Implement MLQAE k>0 first - this is THE quantum advantage that reviewers will scrutinize.

**IBM Quantum Budget**: Use 10 minutes for validation AFTER fixes, not for debugging.

**Estimated Total Time**: 10-14 hours for research-grade quality.

**Next Step**: Begin with copula fix (quick win, 1-2 hours), then tackle MLQAE k>0 (critical, 3-4 hours).
