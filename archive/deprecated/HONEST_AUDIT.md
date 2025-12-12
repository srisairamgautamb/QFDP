# HONEST AUDIT: What Actually Works
**Date**: 2025-11-19  
**Auditor**: Critical Review

## 🟢 REAL & WORKING

### 1. Single-Asset Option Pricing
**Status**: ✅ REAL, VALIDATED  
**Evidence**: 
- Live AAPL data: $267.44 (Alpha Vantage)
- Quantum MLQAE: $22.50
- Classical exact: $21.52
- Error: 4.6% ✅

**What's actually happening**:
- Qiskit `initialize()` creates a statevector with log-normal distribution
- Controlled-RY gates encode call payoff on ancilla
- Binomial sampling (2000 shots) estimates ancilla=1 probability
- Scale back by max payoff to get price

**Test**: `demo_realtime_pricing.py AAPL` - runs in 30s, uses real market data

### 2. Historical Volatility Estimation
**Status**: ✅ REAL
**Evidence**: AAPL σ=32.2% from 252-day returns
**Verification**: `np.std(returns) * np.sqrt(252)` - standard formula, real data

### 3. Sparse Copula Mathematics
**Status**: ✅ REAL (but advantage overstated for small N)
**Evidence**: 
- Factor decomposition via eigenvalues: REAL
- Variance explained: 83.7% for 5-asset portfolio: REAL
- Gate count claim: **MISLEADING for N=5**

**Honest truth**:
- N=5, K=3: 15 gates (sparse) vs 10 gates (full) = **1.5× WORSE**
- Advantage only appears at N≥10:
  - N=10, K=3: 30 gates vs 45 = 1.5× better ✅
  - N=20, K=5: 100 gates vs 190 = 1.9× better ✅

**Fix**: Don't claim advantage for N<10. Our demo uses N=5 which is WORSE.

---

## 🟡 PARTIALLY REAL

### 4. MLQAE "Amplitude Estimation"
**Status**: ⚠️ REAL but NOT quantum advantage
**What we claim**: "MLQAE pricing"
**What's real**: We run k=0 (no Grover iterations) = **direct sampling only**
**Why**: Qiskit's `initialize()` gate cannot be inverted (complex parameters)

**Honest comparison**:
- Classical Monte Carlo: Sample 2000 paths → estimate E[payoff]
- Our "MLQAE" (k=0): Prepare quantum state → sample 2000 shots → estimate E[payoff]
- **Result**: SAME complexity, NO speedup ❌

**What's fake**: Calling this "amplitude estimation" when we don't amplify anything

**To make it real**:
1. Replace `initialize` with explicit rotation gates (invertible)
2. Implement proper Grover operator: A†S₀AS_χ
3. Run k>0 iterations for actual amplification
4. Then we get √speedup (100 shots quantum = 10,000 classical)

### 5. Multi-Asset Basket Pricing
**Status**: ⚠️ RUNS but uses MARGINAL approximation
**What we do**: Encode marginal payoff on first asset only
**Code**: `marginal_payoff = payoff.reshape([2**n]*N).mean(axis=tuple(range(1,N)))`

**Problem**: This is NOT the joint distribution pricing!
- Real basket: E[max(w₁S₁ + w₂S₂ - K, 0)] with CORRELATION
- Our code: E[max(S₁ - K, 0)] marginalized (loses correlation structure)

**Impact**: Basket prices are APPROXIMATE and likely underpriced for diversified portfolios

**To make it real**: Implement full N-register controlled payoff oracle (hard)

---

## 🔴 BROKEN / MISLEADING

### 6. "Quantum Advantage" Claims
**Current state**: We claim 0.67× advantage for N=5 portfolio
**Reality**: 15 gates > 10 gates = **DISADVANTAGE** ❌

**Where this shows up**:
```
Gate advantage: 0.67× over full correlation
```
This is BACKWARDS. Should say "1.5× MORE gates"

**Fix**: Only claim advantage for N≥10, stop showing metrics for N=5

### 7. Portfolio VaR/CVaR Estimates
**Status**: 🔴 FABRICATED
**Code**: 
```python
var_95 = portfolio_value * portfolio_vol * 1.645
cvar_95 = var_95 * 1.2  # Approximation
```

**Problems**:
1. VaR formula is for 1-day, but we use annualized vol (wrong time scale)
2. CVaR = VaR × 1.2 is NOT how CVaR works (completely made up)
3. No correlation impact (uses sqrt(w^T Σ w) but then ignores it for VaR)

**Real VaR/CVaR** requires:
- Monte Carlo simulation of correlated returns
- Tail quantiles (95th percentile)
- CVaR = mean of losses beyond VaR

**Current numbers are MEANINGLESS** ❌

### 8. Sharpe Ratio
**Status**: 🟡 FORMULA IS CORRECT, but data is suspect
**Code**:
```python
sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
```

**Issues**:
- Expected return = mean(historical returns) × 252: assumes past = future ❌
- For 5 tech stocks, getting Sharpe = 0.75 is reasonable but not predictive
- Not "fake" but misleading if used for forward-looking decisions

---

---

## ✅ FIXES COMPLETED (2025-11-19)

### Fix 1: Gate Comparison Now Honest ✅
**Status**: FIXED  
**File**: `qfdp_multiasset/portfolio/quantum_portfolio_manager.py:176-197`  
**Change**: 
- N=5, K=3 now shows "Overhead: 1.50× MORE gates ⚠️"
- Warning added: "(sparse helps at N≥10)"
- Only claims advantage when sparse_gates < full_gates

**Verification**: `verify_honest_fixes.py` Test 3 ✅

### Fix 2: VaR/CVaR Fabrication Removed ✅
**Status**: FIXED  
**File**: `qfdp_multiasset/portfolio/quantum_portfolio_manager.py:214-237`  
**Change**:
```python
# BEFORE (FAKE):
var_95 = portfolio_value * vol * 1.645
cvar_95 = var_95 * 1.2

# AFTER (HONEST):
var_95 = None  # Requires Monte Carlo simulation
cvar_95 = None  # Not implemented
```
**Verification**: `verify_honest_fixes.py` Test 1 ✅

### Fix 3: MLQAE k=0 Limitation Documented ✅
**Status**: FIXED  
**File**: `qfdp_multiasset/mlqae/mlqae_pricing.py:160-178`  
**Change**:
- Added docstring warning: "⚠️ LIMITATION: Only k=0 implemented (no quantum speedup)"
- Changed default: `grover_powers = [0]` (was `[0,1,2,4,8]`)
- Explicitly states "NO QUANTUM SPEEDUP vs classical MC"

**Verification**: `verify_honest_fixes.py` Test 2 ✅

### Fix 4: Basket Marginal Approximation Warning ✅
**Status**: FIXED  
**File**: `qfdp_multiasset/portfolio/basket_pricing.py:210-218`  
**Change**:
```python
# Phase 6: Encode payoff with piecewise oracle
# ⚠️ WARNING: MARGINAL APPROXIMATION ⚠️
# We encode the MARGINAL payoff on first asset only, averaging over others.
# This LOSES correlation structure and is NOT true basket pricing.
```
**Verification**: `verify_honest_fixes.py` Test 4 ✅

---

## 📊 VERIFICATION RESULTS

Run `python verify_honest_fixes.py` to check all fixes:

```
✅ ALL FIXES VERIFIED - OUTPUT IS NOW HONEST

Key changes:
  1. VaR/CVaR: Removed fake calculations, now None
  2. MLQAE: Default k=0 only (no quantum speedup vs MC)
  3. Gate comparison: Shows overhead warning for N<10
  4. Basket pricing: Warns about marginal approximation
```

**All 4/4 tests passing** ✅

---

## 🎯 REMAINING LIMITATIONS (HONEST)

These are NOT bugs to fix - they are honest statements of current capabilities:

1. **MLQAE k=0**: No quantum advantage until we implement invertible state prep
2. **Basket pricing**: Marginal approximation only, not joint distribution
3. **Memory limit**: 14 qubits max due to Qiskit initialize bug (128 TiB allocation)
4. **Sparse copula**: Only advantageous for N≥10 assets
5. **VaR/CVaR**: Not implemented (would need MC simulation)

**Everything else is REAL and VALIDATED** ✅
