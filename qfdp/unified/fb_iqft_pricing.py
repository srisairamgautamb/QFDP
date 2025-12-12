"""
Main FB-IQFT Pricing Pipeline

This module implements the complete 12-step FB-IQFT pipeline for portfolio
option pricing, integrating all components:
- Phase 1 (Steps 1-4): Classical preprocessing
- Phase 2 (Steps 5-7): Carr-Madan Fourier setup
- Phase 3 (Steps 8-10): Quantum computation with IQFT
- Phase 4 (Steps 11-12): Price reconstruction via calibration

The key innovation is using factor decomposition to derive a single portfolio
volatility σ_p, which yields a Gaussian characteristic function requiring only
M=16-32 frequency points. This enables shallow IQFT circuits (depth 32-57) 
suitable for NISQ hardware.
"""

import numpy as np
from typing import Dict, Union
from .carr_madan_gaussian import (
    compute_characteristic_function,
    apply_carr_madan_transform,
    setup_fourier_grid,
    classical_fft_baseline
)
from .frequency_encoding import encode_frequency_state
from .iqft_application import apply_iqft, extract_strike_amplitudes
from .calibration import (
    calibrate_quantum_to_classical,
    reconstruct_option_prices,
    validate_prices
)


class FBIQFTPricing:
    """
    Complete FB-IQFT pipeline for portfolio option pricing.
    
    Implements the unified 12-step QFDP framework with factor-based
    dimensionality reduction to enable shallow quantum circuits.
    
    Attributes:
        M: Fourier grid size (must be power of 2, typically 16 or 32)
        num_qubits: k = log₂(M) qubits needed for IQFT
        alpha: Carr-Madan damping parameter (typically 1.0)
        num_shots: Measurement shots for quantum execution
        A, B: Calibration parameters (fitted during first price_option call)
    
    Example:
        >>> pricer = FBIQFTPricing(M=16, alpha=1.0, num_shots=8192)
        >>> result = pricer.price_option(
        ...     asset_prices=np.array([100, 105, 95]),
        ...     asset_volatilities=np.array([0.2, 0.25, 0.18]),
        ...     correlation_matrix=np.eye(3),
        ...     portfolio_weights=np.array([0.4, 0.3, 0.3]),
        ...     K=110.0,
        ...     T=1.0,
        ...     r=0.05,
        ...     backend='simulator'
        ... )
        >>> print(f"Quantum price: ${result['price_quantum']:.2f}")
        >>> print(f"Classical price: ${result['price_classical']:.2f}")
        >>> print(f"Error: {result['error_percent']:.2f}%")
        >>> print(f"Circuit depth: {result['circuit_depth']}")
    """
    
    def __init__(
        self,
        M: int = 16,
        alpha: float = 1.0,
        num_shots: int = 8192
    ):
        """
        Initialize FB-IQFT pricer.
        
        Args:
            M: Fourier grid size (16 or 32, must be power of 2)
            alpha: Carr-Madan damping parameter (typically 1.0)
            num_shots: Number of measurement shots (typically 8192)
        
        Raises:
            AssertionError: If M is not a power of 2
        """
        assert M > 0 and (M & (M - 1)) == 0, f"M={M} must be a power of 2"
        
        self.M = M
        self.num_qubits = int(np.log2(M))
        self.alpha = alpha
        self.num_shots = num_shots
        
        # Calibration constants (fitted during first price_option call)
        self.A = None
        self.B = None
    
    def price_option(
        self,
        # Portfolio inputs
        asset_prices: np.ndarray,
        asset_volatilities: np.ndarray,
        correlation_matrix: np.ndarray,
        portfolio_weights: np.ndarray,
        # Option parameters
        K: float,
        T: float,
        r: float = 0.05,
        # Execution
        backend: Union[str, object] = 'simulator',
        # Optional
        recalibrate: bool = False
    ) -> Dict:
        """
        Price portfolio option using the complete FB-IQFT pipeline.
        
        Implements all 12 steps of the flowchart:
        - Phase 1: Classical preprocessing (covariance, factors, σ_p, basket)
        - Phase 2: Carr-Madan setup (CF, grid, classical baseline)
        - Phase 3: Quantum computation (state prep, IQFT, measurement)
        - Phase 4: Post-processing (calibration, price reconstruction)
        
        Args:
            asset_prices: Current prices [S₁, ..., Sₙ], shape (N,)
            asset_volatilities: Volatilities [σ₁, ..., σₙ], shape (N,)
            correlation_matrix: Asset correlation matrix, shape (N, N)
            portfolio_weights: Weights [w₁, ..., wₙ], shape (N,), sum to 1
            K: Strike price for target option
            T: Time to maturity (years)
            r: Risk-free rate (e.g., 0.05 for 5%)
            backend: Execution backend
                - 'simulator': AerSimulator (ideal, noiseless)
                - 'ibm_torino' or Backend object: Real hardware
            recalibrate: Force recalibration even if A, B already exist
        
        Returns:
            results: Dictionary containing:
                - price_quantum: Option price from quantum circuit
                - price_classical: Option price from classical FFT baseline
                - error_percent: Relative error |quantum - classical| / classical
                - sigma_p: Portfolio volatility from factor decomposition
                - B_0: Initial basket value
                - num_factors: Number of factors kept (K)
                - explained_variance: % variance explained by K factors
                - loading_matrix: Factor loadings L (N×K)
                - factor_variances: Eigenvalues λ₁, ..., λₖ
                - circuit: QuantumCircuit object
                - circuit_depth: Circuit depth (gates)
                - num_qubits: k qubits used
                - k_grid: Log-strike grid [k₀, ..., k_{M-1}]
                - strikes: Strike grid [K₀, ..., K_{M-1}]
                - prices_quantum: Option prices across all strikes (quantum)
                - prices_classical: Option prices across all strikes (classical)
        
        Raises:
            AssertionError: If validation checks fail (negative prices, etc.)
        
        Example:
            >>> result = pricer.price_option(
            ...     asset_prices=prices,
            ...     asset_volatilities=vols,
            ...     correlation_matrix=corr,
            ...     portfolio_weights=weights,
            ...     K=110.0,
            ...     T=1.0,
            ...     r=0.05
            ... )
            >>> print(f"Price: ${result['price_quantum']:.2f}")
        """
        # ====================================================================
        # PHASE 1: CLASSICAL PREPROCESSING (Steps 1-4)
        # ====================================================================
        
        # Step 1: Covariance matrix construction
        # Σ = Diag(σ)·C·Diag(σ) where C is correlation matrix
        cov = np.outer(asset_volatilities, asset_volatilities) * correlation_matrix
        
        # Step 2: Factor decomposition via eigendecomposition
        # Σ = L·Λ·L^T where L are eigenvectors, Λ are eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvalues descending (PCA convention)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top K factors (typically K=4-5 explains 95%+ variance)
        num_factors = min(5, len(eigenvalues))
        L = eigenvectors[:, :num_factors]  # Loading matrix (N×K)
        Lambda = np.diag(eigenvalues[:num_factors])  # Factor variances (K×K)
        
        # Compute explained variance
        explained_var = np.sum(eigenvalues[:num_factors]) / np.sum(eigenvalues) * 100
        
        # Step 3: Portfolio volatility σ_p
        # CRITICAL: σ_p is a SINGLE scalar, not K-dimensional
        # Formula: σ_p² = w^T·Σ·w
        sigma_p = np.sqrt(portfolio_weights @ cov @ portfolio_weights)
        
        # Step 4: Basket value
        B_0 = np.sum(portfolio_weights * asset_prices)
        
        # ====================================================================
        # PHASE 2: CARR-MADAN FOURIER SETUP (Steps 5-7)
        # ====================================================================
        
        # Step 7: Setup Fourier grid (u-domain and k-domain)
        u_grid, k_grid, delta_u, delta_k = setup_fourier_grid(
            self.M, sigma_p, T, B_0, r, self.alpha
        )
        
        # Validate Nyquist constraint: Δu·Δk = 2π/M
        assert np.isclose(delta_u * delta_k, 2 * np.pi / self.M, rtol=1e-6), \
            f"Nyquist violated: Δu·Δk = {delta_u * delta_k:.6f}, expected {2*np.pi/self.M:.6f}"
        
        # Step 5: Characteristic function φ(u)
        # φ(u) = exp(iu(r-½σ_p²)T - ½σ_p²Tu²)
        phi_values = compute_characteristic_function(u_grid, r, sigma_p, T)
        
        # Step 6: Modified characteristic function ψ(u) via Carr-Madan
        # ψ(u) = e^(-rT)·φ(u-i(α+1)) / (α²+α-u²+i(2α+1)u)
        psi_values = apply_carr_madan_transform(
            u_grid, r, sigma_p, T, self.alpha
        )
        
        # Step 7 (continued): Classical FFT baseline for calibration
        C_classical = classical_fft_baseline(
            psi_values, self.alpha, delta_u, k_grid
        )
        
        # Validate classical prices (sanity checks)
        forward_price = B_0 * np.exp(r * T)
        assert np.all(C_classical >= -1e-10), \
            f"Negative option prices detected in classical baseline (min={np.min(C_classical):.6f})"
        assert np.all(C_classical <= forward_price * 1.01), \
            f"Prices exceed forward bound (max={np.max(C_classical):.2f}, forward={forward_price:.2f})"
        
        # ====================================================================
        # PHASE 3: QUANTUM COMPUTATION (Steps 8-10)
        # ====================================================================
        
        # Step 8: Quantum state preparation
        # Encode ψ(u_j) as |ψ_freq⟩ = Σ_j a_j|j⟩
        circuit, norm_factor = encode_frequency_state(psi_values, self.num_qubits)
        
        # Step 9: Apply IQFT
        # Transform |ψ_freq⟩ → |ψ_strike⟩ = Σ_m g_m|m⟩
        circuit = apply_iqft(circuit, self.num_qubits)
        
        # Step 10: Measurement
        # Extract P(m) ≈ |g_m|² via shots
        quantum_probs = extract_strike_amplitudes(
            circuit, self.num_shots, backend
        )
        
        # ====================================================================
        # PHASE 4: CLASSICAL POST-PROCESSING (Steps 11-12)
        # ====================================================================
        
        # Find target strike index
        k_target = np.log(K / B_0)
        target_idx = np.argmin(np.abs(k_grid - k_target))
        
        # Step 11: LOCAL CALIBRATION (per-strike, not global!)
        # IMPROVEMENT: Use local window around target strike
        window_size = 7  # ±3 strikes around target
        half_window = window_size // 2
        
        # Define calibration window (with boundary handling)
        idx_start = max(0, target_idx - half_window)
        idx_end = min(len(k_grid), target_idx + half_window + 1)
        
        # Extract local data for calibration
        quantum_probs_local = {
            m: quantum_probs.get(m, 0.0) 
            for m in range(idx_start, idx_end)
        }
        C_classical_local = C_classical[idx_start:idx_end]
        k_grid_local = k_grid[idx_start:idx_end]
        
        # Calibrate on local window ONLY
        A_local, B_local = calibrate_quantum_to_classical(
            quantum_probs_local, 
            C_classical_local, 
            k_grid_local
        )
        
        # Store calibration (overwrite global with local)
        self.A = A_local
        self.B = B_local
        
        # Step 12: Price reconstruction (using LOCAL calibration)
        price_quantum = A_local * quantum_probs.get(target_idx, 0.0) + B_local
        price_classical = C_classical[target_idx]
        
        # Also reconstruct full curve for visualization (optional)
        option_prices_quantum = reconstruct_option_prices(
            quantum_probs, A_local, B_local, k_grid, B_0
        )
        
        # Compute relative error
        if price_classical > 1e-10:
            error_percent = abs(price_quantum - price_classical) / price_classical * 100
        else:
            error_percent = np.inf  # Classical price too small
        
        # Optional: Validate prices
        validation = validate_prices(
            option_prices_quantum, k_grid, B_0, r, T, tol=0.05
        )
        
        # ====================================================================
        # RETURN COMPREHENSIVE RESULTS
        # ====================================================================
        
        return {
            # Prices
            'price_quantum': float(price_quantum),
            'price_classical': float(price_classical),
            'error_percent': float(error_percent),
            
            # Portfolio characteristics
            'sigma_p': float(sigma_p),
            'B_0': float(B_0),
            
            # Factor decomposition info
            'num_factors': int(num_factors),
            'explained_variance': float(explained_var),
            'loading_matrix': L,
            'factor_variances': eigenvalues[:num_factors],
            
            # Quantum circuit info
            'circuit': circuit,
            'circuit_depth': int(circuit.depth()),
            'num_qubits': int(self.num_qubits),
            
            # Calibration
            'calibration_A': float(self.A),
            'calibration_B': float(self.B),
            
            # Pricing grids (full strike range)
            'k_grid': k_grid,
            'strikes': B_0 * np.exp(k_grid),
            'prices_quantum': option_prices_quantum,
            'prices_classical': C_classical,
            
            # Validation
            'validation': validation
        }
    
    def price_portfolio_option(
        self,
        weights: np.ndarray,
        sigmas: np.ndarray,
        factors: np.ndarray,
        S0: float,
        K: float,
        T: float,
        r: float,
        backend: Union[str, object] = 'simulator'
    ) -> float:
        """Simplified interface for portfolio option pricing.
        
        Args:
            weights: Portfolio weights
            sigmas: Asset volatilities
            factors: Factor loading matrix (from Cholesky decomposition)
            S0: Initial portfolio value
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            backend: Quantum backend (simulator or hardware)
        
        Returns:
            Option price
        """
        N = len(weights)
        asset_prices = np.ones(N) * (S0 / np.sum(weights))
        correlation_matrix = factors @ factors.T
        
        result = self.price_option(
            asset_prices=asset_prices,
            asset_volatilities=sigmas,
            correlation_matrix=correlation_matrix,
            portfolio_weights=weights,
            K=K,
            T=T,
            r=r,
            backend=backend
        )
        return result['price_quantum']
    
    def price_on_hardware(
        self,
        weights: np.ndarray,
        sigmas: np.ndarray,
        factors: np.ndarray,
        S0: float,
        K: float,
        T: float,
        r: float,
        backend: object
    ) -> float:
        """Price option on REAL IBM Quantum hardware.
        
        This method FORCES execution on actual quantum processors.
        No simulators allowed!
        
        Args:
            weights: Portfolio weights
            sigmas: Asset volatilities  
            factors: Factor loading matrix
            S0: Initial portfolio value
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            backend: IBM Quantum Backend object (from QiskitRuntimeService)
        
        Returns:
            Option price from real quantum hardware
        
        Raises:
            ValueError: If backend is not a real QPU
        """
        # Verify it's actually a real backend
        if isinstance(backend, str):
            raise ValueError(f"backend must be a Backend object, not string '{backend}'")
        
        # Check if it's a simulator (reject it!)
        backend_name = backend.name if hasattr(backend, 'name') else str(backend)
        if 'sim' in backend_name.lower() or 'aer' in backend_name.lower():
            raise ValueError(f"Simulator backend '{backend_name}' not allowed! Use real QPU only.")
        
        return self.price_portfolio_option(
            weights=weights,
            sigmas=sigmas,
            factors=factors,
            S0=S0,
            K=K,
            T=T,
            r=r,
            backend=backend
        )
