# Sparse Copula Theory: Factor-Model Encoding for Quantum Multi-Asset Pricing

Version: 0.1 (Phase 1)

## Overview

We formalize the mathematical foundations of the sparse copula approach: low-rank factor decompositions of correlation matrices for efficient quantum encoding of multi-asset distributions.

Let Σ ∈ R^{N×N} be a correlation matrix (symmetric, positive-definite, unit diagonal). For rank-K approximation (K ≪ N), define:

- Eigenvalue decomposition: Σ = V Λ V^T, Λ = diag(λ₁ ≥ … ≥ λ_N > 0)
- Truncation: V_K = [v₁, …, v_K], Λ_K = diag(λ₁, …, λ_K)
- Loading matrix: L = V_K Λ_K^{1/2} ∈ R^{N×K}
- Idiosyncratic component: D = diag(Σ − L L^T) ≥ 0

Our approximation is Σ_K := L L^T + D.

---

## Theorem A (Rank-K fidelity bound)

Statement (Gaussian-like encoding):

Assume the target joint distribution is multivariate normal with correlation Σ and the quantum-encoded approximation corresponds to Σ_K = L L^T + D with matched marginals. Then the (Uhlmann) fidelity between the corresponding (discretized) quantum states |ψ(Σ)⟩ and |ψ(Σ_K)⟩ admits the lower bound

F(|ψ(Σ)⟩, |ψ(Σ_K)⟩) ≥ exp(−α · ||Σ − Σ_K||_F^2),

where α = α(n, m, Δ) depends on discretization resolution (asset qubits n, factor qubits m) and bin width Δ. For fixed resolution, α is O(1). The bound holds up to additive discretization error O(Δ²).

Sketch of proof:
- For multivariate Gaussians, classical Hellinger/Bhattacharyya affinity bounds can be related to Frobenius perturbations of covariance/correlation.
- Quantum state fidelity reduces to classical overlap of amplitude distributions (statevector encoding of √pdfs).
- A second-order perturbation argument on Cholesky factors yields exponential lower bound in ||Σ − Σ_K||_F² with constant folded into α.
- Discretization induces an additional approximation error upper-bounded by c·Δ².

References:
- Uhlmann (1976), Jozsa (1994) on state fidelity
- Bounds between Gaussian measures via covariance perturbations

---

## Lemma B (Portfolio variance error bound)

Statement:

For any weight vector w ∈ R^N,

| w^T (Σ − Σ_K) w | ≤ ||w||_2^2 · ||Σ − Σ_K||_F.

In particular, for equal-weight portfolio w = (1/N)·1, we get

|σ²_true − σ²_approx| ≤ (1/N) · ||Σ − Σ_K||_F.

Proof:

By Cauchy–Schwarz in Frobenius inner product,

|w^T (Σ − Σ_K) w| = |⟨Σ − Σ_K, w w^T⟩_F| ≤ ||Σ − Σ_K||_F · ||w w^T||_F.

But ||w w^T||_F = ||w||_2^2. This yields the result. The equal-weight corollary follows from ||w||_2^2 = 1/N.

---

## Practical guidance for K selection

- Use scree plots and variance-explained curves to select minimal K such that variance explained ≥ 70–85%.
- Empirically, K=3–5 suffices for many equity universes (S&P-like).
- Portfolio error tolerance 5% suggests Frobenius error threshold ≈ 0.05·N for equal-weight portfolios.

---

## Implications for quantum encoding

- Gate reduction: O(N²) → O(NK) controlled rotations.
- Error propagation: Portfolio metrics are robust to moderate Frobenius error.
- Calibration: Angle mapping can be tuned (β parameter) to match empirical correlations.

---

## Next steps (Phase 1 → Phase 3)

- Validate Theorem A numerically via statevector fidelity for N=5, K=3 (GATE 1 setup).
- Implement angle calibration routine to minimize Frobenius error under circuit constraints.
- Integrate with quantum encoder in Phase 3 and run 25-seed validation.
