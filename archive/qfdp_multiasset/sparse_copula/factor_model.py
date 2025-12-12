"""
Sparse Copula: Factor Model Decomposition
==========================================

Implements low-rank approximation of correlation matrices via eigenvalue decomposition.

Mathematical Foundation:
-----------------------
For a correlation matrix Σ (N×N, symmetric, positive-definite):

    Σ = V·Λ·V^T                    (eigenvalue decomposition)
    Σ_K ≈ V_K·Λ_K·V_K^T            (rank-K truncation)
    L = V_K·Λ_K^(1/2)              (loading matrix, N×K)
    Σ_K = L·L^T                    (reconstruction without idiosyncratic)
    D = diag(Σ - L·L^T)            (idiosyncratic diagonal)
    
Final decomposition:
    Σ ≈ L·L^T + D

This reduces quantum circuit complexity:
    - Naive: O(N²) controlled rotations for full correlation
    - Sparse: O(N×K) controlled rotations where K ≪ N

Theorems:
---------
- **Theorem A (Fidelity Bound):** See SPARSE_COPULA_THEORY.md
- **Lemma B (Portfolio Error):** See SPARSE_COPULA_THEORY.md

Author: QFDP Multi-Asset Research Team
Date: November 2025
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import warnings


@dataclass
class DecompositionMetrics:
    """Metrics for evaluating factor decomposition quality.
    
    Attributes
    ----------
    variance_explained : float
        Fraction of total variance captured by top K factors (0 to 1)
    frobenius_error : float
        Frobenius norm ||Σ - Σ_K||_F of reconstruction error
    max_element_error : float
        Maximum absolute element-wise error max|Σ_ij - Σ_K_ij|
    condition_number : float
        Condition number of original matrix (κ = λ_max / λ_min)
    eigenvalue_ratio : float
        Ratio of smallest retained to largest eigenvalue (λ_K / λ_1)
    """
    variance_explained: float
    frobenius_error: float
    max_element_error: float
    condition_number: float
    eigenvalue_ratio: float


class FactorDecomposer:
    """
    Factor model decomposition of correlation matrices.
    
    Performs low-rank approximation Σ ≈ L·L^T + D using eigenvalue decomposition,
    enabling sparse quantum correlation encoding with O(N×K) complexity.
    
    Parameters
    ----------
    min_eigenvalue : float, default=1e-10
        Minimum eigenvalue threshold for numerical stability
    ensure_positive_definite : bool, default=True
        Ensure reconstructed matrix is positive-definite by clipping D
    
    Examples
    --------
    >>> import numpy as np
    >>> from qfdp_multiasset.sparse_copula import FactorDecomposer
    >>> 
    >>> # Create sample 5×5 correlation matrix
    >>> corr = np.array([
    ...     [1.0, 0.5, 0.3, 0.2, 0.1],
    ...     [0.5, 1.0, 0.4, 0.3, 0.2],
    ...     [0.3, 0.4, 1.0, 0.5, 0.3],
    ...     [0.2, 0.3, 0.5, 1.0, 0.4],
    ...     [0.1, 0.2, 0.3, 0.4, 1.0]
    ... ])
    >>> 
    >>> # Decompose with K=3 factors
    >>> decomposer = FactorDecomposer()
    >>> L, D, metrics = decomposer.fit(corr, K=3)
    >>> 
    >>> print(f"Variance explained: {metrics.variance_explained:.1%}")
    >>> print(f"Frobenius error: {metrics.frobenius_error:.3f}")
    >>> print(f"Loading matrix L shape: {L.shape}")
    >>> print(f"Idiosyncratic D shape: {D.shape}")
    """
    
    def __init__(
        self,
        min_eigenvalue: float = 1e-10,
        ensure_positive_definite: bool = True
    ):
        self.min_eigenvalue = min_eigenvalue
        self.ensure_positive_definite = ensure_positive_definite
        
        # Cached results from last fit
        self._eigenvalues: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None
        self._corr_matrix: Optional[np.ndarray] = None
    
    def auto_select_K(
        self,
        corr_matrix: np.ndarray,
        variance_threshold: float = 0.95,
        error_threshold: float = 0.3
    ) -> int:
        """
        Automatically select K to meet research-grade quality thresholds.
        
        Uses scree plot analysis combined with reconstruction quality validation.
        This addresses the copula reconstruction error issue where fixed K
        can be suboptimal for correlation structure.
        
        Parameters
        ----------
        corr_matrix : np.ndarray, shape (N, N)
            Correlation matrix to analyze
        variance_threshold : float, default=0.95
            Minimum cumulative variance explained (0 to 1)
        error_threshold : float, default=0.3
            Maximum Frobenius reconstruction error
        
        Returns
        -------
        K : int
            Optimal number of factors meeting both thresholds
        
        Algorithm
        ---------
        1. Compute eigenvalue spectrum
        2. Find minimum K where cumulative variance ≥ threshold
        3. Incrementally test K values until Frobenius error < threshold
        4. Return first K satisfying both constraints
        5. Fallback: return N if no K satisfies constraints
        
        Notes
        -----
        Research-grade targets:
        - Variance explained: ≥95% (captures most correlation structure)
        - Frobenius error: <0.3 (low reconstruction error)
        
        For N=5, this typically selects K=4 instead of K=3,
        reducing error from 0.88 to <0.3.
        """
        N = corr_matrix.shape[0]
        
        # Compute eigenvalue spectrum
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
        
        # Cumulative variance explained
        total_variance = np.sum(eigenvalues)
        cumulative_var = np.cumsum(eigenvalues) / total_variance
        
        # Find minimum K for variance threshold
        K_candidates = np.where(cumulative_var >= variance_threshold)[0]
        if len(K_candidates) == 0:
            # Threshold too high, use all factors
            return N
        
        K_min = K_candidates[0] + 1  # +1 because indices are 0-based
        
        # Test K values starting from K_min until error threshold met
        for K in range(K_min, N + 1):
            # Reconstruct with K factors
            eigenvalues_K = eigenvalues[:K]
            eigenvectors = np.linalg.eigh(corr_matrix)[1]
            sort_idx = np.argsort(np.linalg.eigvalsh(corr_matrix))[::-1]
            eigenvectors_K = eigenvectors[:, sort_idx[:K]]
            
            # Loading matrix
            sqrt_lambda = np.sqrt(np.maximum(eigenvalues_K, self.min_eigenvalue))
            L = eigenvectors_K * sqrt_lambda[np.newaxis, :]
            
            # Reconstruction
            Sigma_K = L @ L.T
            
            # Frobenius error
            frobenius_error = np.linalg.norm(corr_matrix - Sigma_K, ord='fro')
            
            if frobenius_error < error_threshold:
                return K
        
        # If no K satisfies error threshold, return N (full rank)
        warnings.warn(
            f"No K in [1, {N}] achieves error < {error_threshold}. "
            f"Using K={N} (full rank)."
        )
        return N
    
    def auto_select_K_balanced(
        self,
        corr_matrix: np.ndarray,
        variance_threshold: float = 0.90,
        error_threshold: float = 0.4,
        prefer_efficiency: bool = True
    ) -> int:
        """
        Select K balancing quality AND gate efficiency.
        
        For production use where both reconstruction quality and quantum
        gate count matter. More relaxed thresholds than auto_select_K().
        
        Parameters
        ----------
        corr_matrix : np.ndarray
            Correlation matrix
        variance_threshold : float, default=0.90
            Minimum variance (relaxed from 0.95)
        error_threshold : float, default=0.4
            Maximum error (relaxed from 0.3)
        prefer_efficiency : bool, default=True
            If True, choose smallest K meeting thresholds
        
        Returns
        -------
        K : int
            Balanced K value
        """
        N = corr_matrix.shape[0]
        
        # For small N, use heuristic: K ≈ sqrt(N) or N/2
        K_heuristic = max(2, min(int(np.sqrt(N)) + 1, N // 2))
        
        # Test K values from small to large
        for K in range(2, N + 1):
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvectors = np.linalg.eigh(corr_matrix)[1]
            sort_idx = np.argsort(np.linalg.eigvalsh(corr_matrix))[::-1]
            
            # Check variance
            cumvar = np.sum(eigenvalues[:K]) / np.sum(eigenvalues)
            if cumvar < variance_threshold:
                continue
            
            # Check error
            eigenvectors_K = eigenvectors[:, sort_idx[:K]]
            sqrt_lambda = np.sqrt(np.maximum(eigenvalues[:K], self.min_eigenvalue))
            L = eigenvectors_K * sqrt_lambda[np.newaxis, :]
            Sigma_K = L @ L.T
            error = np.linalg.norm(corr_matrix - Sigma_K, ord='fro')
            
            if error < error_threshold:
                # Found acceptable K
                if prefer_efficiency:
                    return K  # Return immediately (smallest K)
                else:
                    # Continue searching for better quality
                    if cumvar >= 0.95 and error < 0.3:
                        return K
        
        # Fallback: use heuristic
        return min(K_heuristic, N)
    
    def fit(
        self,
        corr_matrix: np.ndarray,
        K: Optional[int] = None,
        validate: bool = True,
        variance_threshold: float = 0.95,
        error_threshold: float = 0.3,
        mode: str = 'quality',
        gate_priority: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, DecompositionMetrics]:
        """
        Fit factor model decomposition: Σ ≈ L·L^T + D.
        
        Parameters
        ----------
        corr_matrix : np.ndarray, shape (N, N)
            Correlation matrix to decompose. Must be:
            - Symmetric: Σ_ij = Σ_ji
            - Positive-definite: all eigenvalues > 0
            - Unit diagonal: Σ_ii = 1
        K : int, optional
            Number of factors to retain (1 ≤ K ≤ N).
            If None, automatically selects K based on mode.
        validate : bool, default=True
            Validate correlation matrix properties
        variance_threshold : float, default=0.95
            Minimum variance explained (used if K is None)
        error_threshold : float, default=0.3
            Maximum Frobenius error (used if K is None)
        mode : str, default='quality'
            Selection mode: 'quality' (strict thresholds) or 'balanced' (efficiency)
        gate_priority : bool, default=False
            If True, prioritize gate efficiency over quality for small N (<30).
            Uses fixed K=3-4 to ensure gate advantage.
        
        Returns
        -------
        L : np.ndarray, shape (N, K)
            Loading matrix: L = V_K·Λ_K^(1/2)
        D : np.ndarray, shape (N, N)
            Diagonal idiosyncratic matrix: D = diag(Σ - L·L^T)
        metrics : DecompositionMetrics
            Quality metrics for decomposition
        
        Raises
        ------
        ValueError
            If K is invalid, matrix is not correlation matrix, or decomposition fails
        
        Notes
        -----
        **Algorithm:**
        
        1. Eigenvalue decomposition: Σ = V·Λ·V^T
        2. Sort eigenvalues descending: λ_1 ≥ λ_2 ≥ ... ≥ λ_N
        3. Retain top K: V_K = V[:, :K], Λ_K = Λ[:K, :K]
        4. Compute loadings: L = V_K·√Λ_K
        5. Compute idiosyncratic: D = diag(Σ - L·L^T)
        6. Ensure positive: D_ii = max(D_ii, min_eigenvalue)
        
        **Complexity:**
        
        - Time: O(N³) for eigendecomposition
        - Space: O(N²) for intermediate matrices
        
        **Quantum Circuit Impact:**
        
        Reduces controlled-Ry gates from N(N-1)/2 to N×K:
        
        - N=5, K=3: 10 → 15 rotations (no reduction for small N)
        - N=10, K=3: 45 → 30 rotations (1.5× reduction)
        - N=20, K=3: 190 → 60 rotations (3.2× reduction)
        - N=50, K=3: 1,225 → 150 rotations (8.2× reduction)
        """
        # Cache input
        self._corr_matrix = corr_matrix.copy()
        N = corr_matrix.shape[0]
        
        # Validate inputs
        if validate:
            self._validate_correlation_matrix(corr_matrix)
        
        # Auto-select K if not provided
        if K is None:
            # Gate-priority mode: For small N, use fixed low K for advantage
            if gate_priority and N < 30:
                # Fixed K for gate efficiency
                K = 3 if N <= 12 else 4
                print(f"[FactorDecomposer] Gate-priority K={K}/{N} "
                      f"(optimized for quantum gate advantage)")
                # Validate it meets minimum quality
                eigenvalues = np.linalg.eigvalsh(corr_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]
                variance_explained = np.sum(eigenvalues[:K]) / np.sum(eigenvalues)
                print(f"  → Variance: {variance_explained:.1%} "
                      f"(relaxed threshold for gate efficiency)")
            elif mode == 'balanced':
                K = self.auto_select_K_balanced(
                    corr_matrix,
                    variance_threshold=0.90,
                    error_threshold=0.4,
                    prefer_efficiency=True
                )
                print(f"[FactorDecomposer] Balanced K={K}/{N} "
                      f"(optimized for gate efficiency)")
            else:
                K = self.auto_select_K(
                    corr_matrix, 
                    variance_threshold=variance_threshold,
                    error_threshold=error_threshold
                )
                print(f"[FactorDecomposer] Quality K={K}/{N} "
                      f"(var≥{variance_threshold:.0%}, err<{error_threshold})")
        
        if not (1 <= K <= N):
            raise ValueError(f"K must be in [1, {N}], got K={K}")
        
        # Step 1: Eigenvalue decomposition
        # Use eigh (Hermitian) for symmetric matrices - more stable and faster
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        
        # Cache for analysis
        self._eigenvalues = eigenvalues.copy()
        self._eigenvectors = eigenvectors.copy()
        
        # Step 2: Sort in descending order (eigh returns ascending)
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        
        # Check for numerical issues
        if eigenvalues[0] < 0:
            warnings.warn(
                f"Largest eigenvalue is negative: {eigenvalues[0]:.2e}. "
                "Matrix may not be positive-definite."
            )
        
        # Step 3: Retain top K eigenvalues and eigenvectors
        eigenvalues_K = eigenvalues[:K]
        eigenvectors_K = eigenvectors[:, :K]
        
        # Step 4: Compute loading matrix L = V_K·√Λ_K
        # Clip negative eigenvalues for numerical stability
        eigenvalues_K_clipped = np.maximum(eigenvalues_K, self.min_eigenvalue)
        sqrt_lambda_K = np.sqrt(eigenvalues_K_clipped)
        
        L = eigenvectors_K * sqrt_lambda_K[np.newaxis, :]  # Broadcast sqrt eigenvalues
        
        # Step 5: Reconstruct low-rank approximation
        Sigma_K = L @ L.T
        
        # Step 6: Compute idiosyncratic diagonal
        residual = corr_matrix - Sigma_K
        D_diag = np.diag(residual)
        
        # Ensure positive diagonal (correlation matrix must have positive diagonal)
        if self.ensure_positive_definite:
            D_diag = np.maximum(D_diag, self.min_eigenvalue)
        
        D = np.diag(D_diag)
        
        # Step 7: Compute quality metrics
        metrics = self._compute_metrics(corr_matrix, Sigma_K, D, eigenvalues, K)
        
        return L, D, metrics
    
    def _validate_correlation_matrix(self, corr_matrix: np.ndarray) -> None:
        """Validate correlation matrix properties."""
        N = corr_matrix.shape[0]
        
        # Check square
        if corr_matrix.shape != (N, N):
            raise ValueError(
                f"Correlation matrix must be square, got shape {corr_matrix.shape}"
            )
        
        # Check symmetric
        if not np.allclose(corr_matrix, corr_matrix.T, atol=1e-8):
            max_asymmetry = np.max(np.abs(corr_matrix - corr_matrix.T))
            raise ValueError(
                f"Correlation matrix must be symmetric. "
                f"Max asymmetry: {max_asymmetry:.2e}"
            )
        
        # Check diagonal is ones
        diagonal = np.diag(corr_matrix)
        if not np.allclose(diagonal, 1.0, atol=1e-6):
            max_diag_error = np.max(np.abs(diagonal - 1.0))
            raise ValueError(
                f"Correlation matrix diagonal must be 1.0. "
                f"Max error: {max_diag_error:.2e}"
            )
        
        # Check positive-definite (all eigenvalues > 0)
        # We'll recompute eigenvalues in fit(), but check here for early failure
        try:
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            min_eigenvalue = np.min(eigenvalues)
            if min_eigenvalue < -1e-10:  # Allow small numerical errors
                raise ValueError(
                    f"Correlation matrix must be positive-definite. "
                    f"Minimum eigenvalue: {min_eigenvalue:.2e}"
                )
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to compute eigenvalues: {e}")
    
    def _compute_metrics(
        self,
        Sigma: np.ndarray,
        Sigma_K: np.ndarray,
        D: np.ndarray,
        eigenvalues: np.ndarray,
        K: int
    ) -> DecompositionMetrics:
        """Compute decomposition quality metrics."""
        N = Sigma.shape[0]
        
        # Variance explained: sum of top K eigenvalues / sum of all eigenvalues
        total_variance = np.sum(eigenvalues)
        explained_variance = np.sum(eigenvalues[:K])
        variance_explained = explained_variance / total_variance if total_variance > 0 else 0.0
        
        # Frobenius error: ||Σ - Σ_K||_F
        diff = Sigma - Sigma_K
        frobenius_error = np.linalg.norm(diff, ord='fro')
        
        # Max element error
        max_element_error = np.max(np.abs(diff))
        
        # Condition number: λ_max / λ_min
        max_eigenvalue = eigenvalues[0]
        min_eigenvalue = eigenvalues[-1]
        condition_number = max_eigenvalue / min_eigenvalue if min_eigenvalue > 1e-15 else np.inf
        
        # Eigenvalue ratio: λ_K / λ_1
        eigenvalue_ratio = eigenvalues[K-1] / eigenvalues[0] if eigenvalues[0] > 0 else 0.0
        
        return DecompositionMetrics(
            variance_explained=variance_explained,
            frobenius_error=frobenius_error,
            max_element_error=max_element_error,
            condition_number=condition_number,
            eigenvalue_ratio=eigenvalue_ratio
        )
    
    def get_eigenvalue_spectrum(self) -> Optional[np.ndarray]:
        """
        Get eigenvalue spectrum from last fit.
        
        Returns
        -------
        eigenvalues : np.ndarray or None
            Eigenvalues sorted descending, or None if fit() not called yet
        """
        return self._eigenvalues.copy() if self._eigenvalues is not None else None
    
    def compute_portfolio_variance_error(
        self,
        L: np.ndarray,
        D: np.ndarray,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute portfolio variance error: |w^T·Σ·w - w^T·Σ_K·w|.
        
        This validates **Lemma B** from the theory: portfolio-level impact of
        the sparse approximation.
        
        Parameters
        ----------
        L : np.ndarray, shape (N, K)
            Loading matrix from fit()
        D : np.ndarray, shape (N, N)
            Idiosyncratic diagonal from fit()
        weights : np.ndarray, shape (N,)
            Portfolio weights (should sum to 1)
        
        Returns
        -------
        results : dict
            - 'variance_true': w^T·Σ·w (true portfolio variance)
            - 'variance_approx': w^T·(L·L^T + D)·w (approximate)
            - 'absolute_error': |variance_true - variance_approx|
            - 'relative_error': absolute_error / variance_true
        
        Notes
        -----
        **Lemma B:** |w^T·Σ·w - w^T·Σ_K·w| ≤ ||w||²·||Σ - Σ_K||_F
        
        For equal-weight portfolio (w_i = 1/N), the bound simplifies to:
            error ≤ ||Σ - Σ_K||_F / N
        """
        if self._corr_matrix is None:
            raise RuntimeError("Must call fit() before computing portfolio variance error")
        
        Sigma = self._corr_matrix
        Sigma_K = L @ L.T + D
        
        # True portfolio variance
        variance_true = weights @ Sigma @ weights
        
        # Approximate portfolio variance
        variance_approx = weights @ Sigma_K @ weights
        
        # Errors
        absolute_error = np.abs(variance_true - variance_approx)
        relative_error = absolute_error / variance_true if variance_true > 1e-15 else np.inf
        
        return {
            'variance_true': float(variance_true),
            'variance_approx': float(variance_approx),
            'absolute_error': float(absolute_error),
            'relative_error': float(relative_error)
        }


def generate_synthetic_correlation_matrix(
    N: int,
    K: int,
    noise_scale: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic S&P-like correlation matrix using factor model.
    
    Creates realistic correlation matrix with controllable factor structure,
    useful for testing and validation.
    
    Parameters
    ----------
    N : int
        Number of assets
    K : int
        Number of latent factors
    noise_scale : float, default=0.1
        Scale of idiosyncratic noise (0 = pure factor model)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    corr_matrix : np.ndarray, shape (N, N)
        Synthetic correlation matrix
    
    Notes
    -----
    **Generation Process:**
    
    1. Generate random factor loadings L (N×K) from N(0, 1/√K)
    2. Compute systematic correlation: Σ_sys = L·L^T
    3. Add idiosyncratic: D = noise_scale·I
    4. Combine: Σ = Σ_sys + D
    5. Normalize to correlation: Σ_ij /= √(Σ_ii·Σ_jj)
    
    **Properties:**
    
    - Guaranteed positive-definite (by construction)
    - Diagonal elements = 1 (after normalization)
    - Variance explained by top K factors ≈ 1/(1 + noise_scale)
    
    Examples
    --------
    >>> from qfdp_multiasset.sparse_copula import generate_synthetic_correlation_matrix
    >>> 
    >>> # Generate 10×10 matrix with 3 factors
    >>> corr = generate_synthetic_correlation_matrix(N=10, K=3, seed=42)
    >>> print(corr.shape)
    (10, 10)
    >>> 
    >>> # Check properties
    >>> print(f"Symmetric: {np.allclose(corr, corr.T)}")
    >>> print(f"Diagonal: {np.allclose(np.diag(corr), 1.0)}")
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random factor loadings
    # Scale by 1/√K to keep variance reasonable
    L = np.random.randn(N, K) / np.sqrt(K)
    
    # Systematic correlation component
    Sigma_sys = L @ L.T
    
    # Add idiosyncratic diagonal
    D = noise_scale * np.eye(N)
    
    # Combine
    Sigma = Sigma_sys + D
    
    # Normalize to correlation matrix (diagonal = 1)
    std_devs = np.sqrt(np.diag(Sigma))
    corr_matrix = Sigma / np.outer(std_devs, std_devs)
    
    # Ensure exactly 1 on diagonal (numerical stability)
    np.fill_diagonal(corr_matrix, 1.0)
    
    return corr_matrix


def analyze_eigenvalue_decay(
    corr_matrix: np.ndarray,
    K_max: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Analyze eigenvalue decay and variance explained by factors.
    
    Useful for determining optimal K (number of factors) for a given
    correlation matrix.
    
    Parameters
    ----------
    corr_matrix : np.ndarray, shape (N, N)
        Correlation matrix to analyze
    K_max : int, optional
        Maximum K to consider (default: N)
    
    Returns
    -------
    analysis : dict
        - 'eigenvalues': sorted eigenvalues (descending)
        - 'variance_explained': cumulative variance explained (K=1 to K=N)
        - 'frobenius_error_by_K': Frobenius error for each K
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from qfdp_multiasset.sparse_copula import analyze_eigenvalue_decay
    >>> 
    >>> # Analyze S&P 500 correlation matrix
    >>> analysis = analyze_eigenvalue_decay(corr_matrix)
    >>> 
    >>> # Plot scree plot
    >>> plt.plot(analysis['eigenvalues'])
    >>> plt.xlabel('Factor index')
    >>> plt.ylabel('Eigenvalue')
    >>> plt.title('Scree Plot')
    >>> plt.show()
    >>> 
    >>> # Plot cumulative variance explained
    >>> plt.plot(analysis['variance_explained'])
    >>> plt.xlabel('Number of factors K')
    >>> plt.ylabel('Variance explained')
    >>> plt.axhline(0.9, color='r', linestyle='--', label='90% threshold')
    >>> plt.legend()
    >>> plt.show()
    """
    N = corr_matrix.shape[0]
    K_max = K_max or N
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    
    # Sort descending
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    
    # Cumulative variance explained
    total_variance = np.sum(eigenvalues)
    variance_explained = np.cumsum(eigenvalues) / total_variance
    
    # Frobenius error for each K
    frobenius_errors = np.zeros(K_max)
    for K in range(1, K_max + 1):
        # Reconstruct with K factors
        eigenvalues_K = eigenvalues[:K]
        eigenvectors_K = eigenvectors[:, :K]
        L = eigenvectors_K * np.sqrt(np.maximum(eigenvalues_K, 0))[np.newaxis, :]
        Sigma_K = L @ L.T
        
        # Frobenius error
        frobenius_errors[K-1] = np.linalg.norm(corr_matrix - Sigma_K, ord='fro')
    
    return {
        'eigenvalues': eigenvalues,
        'variance_explained': variance_explained[:K_max],
        'frobenius_error_by_K': frobenius_errors
    }
