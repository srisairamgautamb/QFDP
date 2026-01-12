"""
Enhanced FB-IQFT Pricing with QRC Factor Support
================================================

Extends base FB-IQFT to accept adaptive factors from QRC,
replacing the static PCA decomposition.

Integration Point: Phase 1, Step 2 (Factor Decomposition)
- Original: eigenvalues/eigenvectors from PCA
- Enhanced: QRC factors modulate the covariance structure

Key insight: QRC factors weight the importance of different variance 
components, affecting the effective portfolio volatility σ_p.
"""

import numpy as np
from typing import Dict, Union, Optional, Tuple
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.fb_iqft_pricing import FBIQFTPricing
from qrc import QuantumRecurrentCircuit


class FBIQFTPricingEnhanced(FBIQFTPricing):
    """
    Enhanced FB-IQFT that accepts custom factors from QRC.
    
    Backward compatible: Works with or without QRC factors.
    
    Usage:
        pricer = FBIQFTPricingEnhanced()
        
        # With QRC factors:
        pricer.set_qrc_factors(qrc_factors)
        result = pricer.price_option(..., use_qrc=True)
        
        # Without QRC (standard PCA):
        result = pricer.price_option(..., use_qrc=False)
    """
    
    def __init__(self, M: int = 16, alpha: float = 1.0, num_shots: int = 8192, beta: float = 0.5):
        super().__init__(M, alpha, num_shots)
        
        # QRC factor storage
        self._qrc_factors = None
        self._use_qrc = False
        
        # Mathematically rigorous modulation framework
        from qfdp.unified.qrc_modulation import QRCModulation
        self._modulator = QRCModulation(beta=beta)
        
    def set_qrc_factors(self, factors: np.ndarray):
        """
        Set custom factors from QRC.
        
        Args:
            factors: np.ndarray of shape (n_factors,)
                    Must sum to approximately 1.0
        """
        factors = np.array(factors)
        
        # Validation
        if not np.allclose(factors.sum(), 1.0, atol=0.05):
            raise ValueError(f"Factors must sum to ~1.0, got {factors.sum():.4f}")
        if np.any(factors < 0):
            raise ValueError("Factors must be non-negative")
        
        self._qrc_factors = factors
        
    def clear_qrc_factors(self):
        """Reset to default PCA factors."""
        self._qrc_factors = None
        
    def _compute_sigma_p_with_qrc(
        self,
        portfolio_weights: np.ndarray,
        asset_volatilities: np.ndarray,
        correlation_matrix: np.ndarray,
        qrc_factors: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Compute portfolio volatility with QRC factor modulation.
        
        Implements the rigorous mathematical framework:
        - Modulation: h(f, f̄) = 1 + β(f/f̄ - 1)
        - Adaptive eigenvalues: λ̃ᵢ = λᵢ · h(fᵢ, f̄)
        - Adaptive covariance: C_QRC = Q Λ̃ Q^T
        - Portfolio vol: σ_p = √(w^T C_QRC w)
        
        Args:
            portfolio_weights: Portfolio weights w
            asset_volatilities: Asset volatilities σ
            correlation_matrix: Correlation matrix Σ
            qrc_factors: QRC factors f(t)
        
        Returns:
            sigma_p: Adaptive portfolio volatility
            diagnostics: Dict with modulation details
        """
        n_assets = len(portfolio_weights)
        
        # Base covariance: C = diag(σ) Σ diag(σ)
        base_cov = np.outer(asset_volatilities, asset_volatilities) * correlation_matrix
        
        # PCA decomposition: C = Q Λ Q^T
        eigenvalues, eigenvectors = np.linalg.eigh(base_cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top K factors
        n_factors = min(len(qrc_factors), len(eigenvalues))
        eigenvectors_K = eigenvectors[:, :n_factors]
        eigenvalues_K = eigenvalues[:n_factors]
        
        # Use rigorous QRC modulation
        sigma_p, diagnostics = self._modulator.compute_adaptive_portfolio_variance(
            portfolio_weights,
            eigenvectors_K,
            eigenvalues_K,
            qrc_factors
        )
        
        return sigma_p, diagnostics
    
    def price_option_enhanced(
        self,
        asset_prices: np.ndarray,
        asset_volatilities: np.ndarray,
        correlation_matrix: np.ndarray,
        portfolio_weights: np.ndarray,
        K: float,
        T: float,
        r: float = 0.05,
        backend: Union[str, object] = 'simulator',
        use_qrc: bool = True,
        qrc_factors: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Price option with QRC factor enhancement.
        
        This PROPERLY uses QRC factors by modulating sigma_p BEFORE
        the Carr-Madan and IQFT steps.
        """
        from qfdp.unified.carr_madan_gaussian import (
            setup_fourier_grid,
            compute_characteristic_function,
            apply_carr_madan_transform,
            classical_fft_baseline
        )
        from qfdp.unified.frequency_encoding import encode_frequency_state
        from qfdp.unified.iqft_application import apply_iqft, extract_strike_amplitudes
        from qfdp.unified.calibration import (
            calibrate_quantum_to_classical,
            reconstruct_option_prices,
            validate_prices
        )
        
        # ================================================================
        # PHASE 1: CLASSICAL PREPROCESSING (with QRC modulation)
        # ================================================================
        
        # Step 1: Covariance matrix
        cov = np.outer(asset_volatilities, asset_volatilities) * correlation_matrix
        
        # Step 2: Factor decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        num_factors = min(5, len(eigenvalues))
        L = eigenvectors[:, :num_factors]
        Lambda = np.diag(eigenvalues[:num_factors])
        explained_var = np.sum(eigenvalues[:num_factors]) / np.sum(eigenvalues) * 100
        
        # Step 3: Portfolio volatility - QRC MODULATION HERE
        if use_qrc:
            factors = qrc_factors if qrc_factors is not None else self._qrc_factors
            if factors is None:
                raise ValueError("QRC factors not set")
            
            # Compute QRC-modulated sigma_p
            sigma_p, mod_eigenvalues = self._compute_sigma_p_with_qrc(
                portfolio_weights, asset_volatilities, correlation_matrix, factors
            )
            qrc_info = {
                'qrc_factors': factors,
                'qrc_enabled': True,
                'sigma_p_pca': np.sqrt(portfolio_weights @ cov @ portfolio_weights)
            }
        else:
            # Standard PCA sigma_p
            sigma_p = np.sqrt(portfolio_weights @ cov @ portfolio_weights)
            qrc_info = {'qrc_enabled': False}
        
        # Step 4: Basket value
        B_0 = np.sum(portfolio_weights * asset_prices)
        
        # ================================================================
        # PHASE 2: CARR-MADAN FOURIER SETUP
        # ================================================================
        u_grid, k_grid, delta_u, delta_k = setup_fourier_grid(
            self.M, sigma_p, T, B_0, r, self.alpha
        )
        
        phi_values = compute_characteristic_function(u_grid, r, sigma_p, T)
        psi_values = apply_carr_madan_transform(u_grid, r, sigma_p, T, self.alpha)
        C_classical = classical_fft_baseline(psi_values, self.alpha, delta_u, k_grid)
        
        # ================================================================
        # PHASE 3: QUANTUM COMPUTATION
        # ================================================================
        circuit, norm_factor = encode_frequency_state(psi_values, self.num_qubits)
        circuit = apply_iqft(circuit, self.num_qubits)
        quantum_probs = extract_strike_amplitudes(circuit, self.num_shots, backend)
        
        # ================================================================
        # PHASE 4: POST-PROCESSING
        # ================================================================
        k_target = np.log(K / B_0)
        target_idx = np.argmin(np.abs(k_grid - k_target))
        
        # Local calibration
        window_size = 7
        half_window = window_size // 2
        idx_start = max(0, target_idx - half_window)
        idx_end = min(len(k_grid), target_idx + half_window + 1)
        
        quantum_probs_local = {m: quantum_probs.get(m, 0.0) for m in range(idx_start, idx_end)}
        C_classical_local = C_classical[idx_start:idx_end]
        k_grid_local = k_grid[idx_start:idx_end]
        
        A_local, B_local = calibrate_quantum_to_classical(
            quantum_probs_local, C_classical_local, k_grid_local
        )
        
        self.A = A_local
        self.B = B_local
        
        # Price reconstruction
        price_quantum = A_local * quantum_probs.get(target_idx, 0.0) + B_local
        price_classical = C_classical[target_idx]
        
        option_prices_quantum = reconstruct_option_prices(
            quantum_probs, A_local, B_local, k_grid, B_0
        )
        
        if price_classical > 1e-10:
            error_percent = abs(price_quantum - price_classical) / price_classical * 100
        else:
            error_percent = np.inf
        
        validation = validate_prices(option_prices_quantum, k_grid, B_0, r, T, tol=0.05)
        
        # ================================================================
        # RETURN RESULTS
        # ================================================================
        result = {
            'price_quantum': float(price_quantum),
            'price_classical': float(price_classical),
            'error_percent': float(error_percent),
            'sigma_p': float(sigma_p),
            'B_0': float(B_0),
            'num_factors': int(num_factors),
            'explained_variance': float(explained_var),
            'loading_matrix': L,
            'factor_variances': eigenvalues[:num_factors],
            'circuit': circuit,
            'circuit_depth': int(circuit.depth()),
            'num_qubits': int(self.num_qubits),
            'calibration_A': float(self.A),
            'calibration_B': float(self.B),
            'k_grid': k_grid,
            'strikes': B_0 * np.exp(k_grid),
            'prices_quantum': option_prices_quantum,
            'prices_classical': C_classical,
            'validation': validation
        }
        
        result.update(qrc_info)
        return result


class QRCIntegratedPricer:
    """
    High-level wrapper combining QRC + FB-IQFT for end-to-end pricing.
    
    This is the main interface for QRC-enhanced derivative pricing.
    
    Usage:
        pricer = QRCIntegratedPricer()
        
        # Price with QRC adaptation
        result = pricer.price_with_qrc(
            market_data={'prices': 100, 'volatility': 0.25, ...},
            strike=100,
            correlation_matrix=corr,
            ...
        )
        
        # Price without QRC (baseline)
        result_baseline = pricer.price_with_pca(...)
    """
    
    def __init__(self, n_factors: int = 4, M: int = 16):
        self.qrc = QuantumRecurrentCircuit(n_factors=n_factors)
        self.fb_iqft = FBIQFTPricingEnhanced(M=M)
        self.n_factors = n_factors
        
    def price_with_qrc(
        self,
        market_data: Dict,
        asset_prices: np.ndarray,
        asset_volatilities: np.ndarray,
        correlation_matrix: np.ndarray,
        portfolio_weights: np.ndarray,
        strike: float,
        maturity: float = 1.0,
        r: float = 0.05
    ) -> Dict:
        """
        Price option with QRC-extracted factors.
        
        Args:
            market_data: Dict for QRC input with keys:
                - 'prices': Current spot price
                - 'volatility': Market volatility
                - 'corr_change': Correlation change from baseline
                - 'stress': Market stress indicator (0-1)
            asset_prices: Asset price array
            asset_volatilities: Asset volatility array
            correlation_matrix: Correlation matrix
            portfolio_weights: Portfolio weights
            strike: Option strike price
            maturity: Time to maturity
            r: Risk-free rate
            
        Returns:
            result: Dict with price and QRC info
        """
        
        # Generate QRC factors
        qrc_result = self.qrc.forward(market_data)
        qrc_factors = qrc_result.factors
        
        # Set factors in pricer
        self.fb_iqft.set_qrc_factors(qrc_factors)
        
        # Price with QRC
        result = self.fb_iqft.price_option_enhanced(
            asset_prices=asset_prices,
            asset_volatilities=asset_volatilities,
            correlation_matrix=correlation_matrix,
            portfolio_weights=portfolio_weights,
            K=strike,
            T=maturity,
            r=r,
            use_qrc=True
        )
        
        result['qrc_factors'] = qrc_factors
        result['method'] = 'QRC-Enhanced'
        
        return result
    
    def price_with_pca(
        self,
        asset_prices: np.ndarray,
        asset_volatilities: np.ndarray,
        correlation_matrix: np.ndarray,
        portfolio_weights: np.ndarray,
        strike: float,
        maturity: float = 1.0,
        r: float = 0.05
    ) -> Dict:
        """
        Price option with standard PCA factors (baseline).
        """
        
        # Clear any QRC factors
        self.fb_iqft.clear_qrc_factors()
        
        # Price with standard PCA
        result = self.fb_iqft.price_option(
            asset_prices=asset_prices,
            asset_volatilities=asset_volatilities,
            correlation_matrix=correlation_matrix,
            portfolio_weights=portfolio_weights,
            K=strike,
            T=maturity,
            r=r
        )
        
        result['method'] = 'Standard PCA'
        
        return result
