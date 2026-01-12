"""
QRC-Integrated Pricer - FIXED VERSION
======================================

CRITICAL FIX: Actually uses QRC sigma_p in pricing, not just computes it.

The previous version computed sigma_p_qrc but the base model ignored it.
This version directly injects sigma_p into the Carr-Madan/IQFT pipeline.
"""

import numpy as np
from typing import Dict, Optional
from datetime import datetime
import logging
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.adapter_layer import BaseModelAdapter
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
from qrc import QuantumRecurrentCircuit

logger = logging.getLogger(__name__)


class QRCIntegratedPricer:
    """
    Integrated pricer that ACTUALLY uses QRC sigma_p in pricing.
    
    Key difference from broken version:
    - Broken: computed sigma_p_qrc, then called base_pricer which ignored it
    - Fixed:  computes sigma_p_qrc, then calls pricing pipeline directly with it
    """
    
    def __init__(self, n_factors: int = 4, M: int = 16, beta: float = 0.1, num_shots: int = 8192):
        self.qrc = QuantumRecurrentCircuit(n_factors=n_factors)
        self.adapter = BaseModelAdapter(beta=beta)
        self.M = M
        self.num_qubits = int(np.log2(M))
        self.num_shots = num_shots
        self.alpha = 1.0  # Carr-Madan damping
        
        logger.info(f"QRCIntegratedPricer initialized: M={M}, beta={beta}")
    
    def _price_with_sigma_p(
        self,
        sigma_p: float,
        B_0: float,
        K: float,
        T: float,
        r: float = 0.05
    ) -> Dict:
        """
        Price option using a given sigma_p.
        
        This is the CORE fix: directly use sigma_p in the pricing pipeline.
        """
        # Phase 2: Carr-Madan setup with the given sigma_p
        u_grid, k_grid, delta_u, delta_k = setup_fourier_grid(
            self.M, sigma_p, T, B_0, r, self.alpha
        )
        
        phi_values = compute_characteristic_function(u_grid, r, sigma_p, T)
        psi_values = apply_carr_madan_transform(u_grid, r, sigma_p, T, self.alpha)
        C_classical = classical_fft_baseline(psi_values, self.alpha, delta_u, k_grid)
        
        # Phase 3: Quantum computation
        circuit, norm_factor = encode_frequency_state(psi_values, self.num_qubits)
        circuit = apply_iqft(circuit, self.num_qubits)
        quantum_probs = extract_strike_amplitudes(circuit, self.num_shots, 'simulator')
        
        # Phase 4: Post-processing
        k_target = np.log(K / B_0)
        target_idx = np.argmin(np.abs(k_grid - k_target))
        
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
        
        price_quantum = A_local * quantum_probs.get(target_idx, 0.0) + B_local
        price_classical = C_classical[target_idx]
        
        if price_classical > 1e-10:
            error_percent = abs(price_quantum - price_classical) / price_classical * 100
        else:
            error_percent = float('inf')
        
        return {
            'price_quantum': float(price_quantum),
            'price_classical': float(price_classical),
            'error_percent': float(error_percent),
            'sigma_p': float(sigma_p)
        }
    
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
        Price option using QRC-adapted sigma_p.
        
        FIXED: Actually uses sigma_p_qrc in pricing!
        """
        # Step 1: Generate QRC factors
        qrc_result = self.qrc.forward(market_data)
        qrc_factors = qrc_result.factors
        
        # Step 2: Compute QRC sigma_p via adapter
        adapter_result = self.adapter.prepare_for_qrc_pricing(
            spot_prices=asset_prices,
            volatilities=asset_volatilities,
            correlation_matrix=correlation_matrix,
            qrc_factors=qrc_factors,
            portfolio_weights=portfolio_weights
        )
        
        sigma_p_qrc = adapter_result['sigma_p_qrc']
        sigma_p_pca = adapter_result['sigma_p_pca']
        
        # Step 3: Basket value
        B_0 = np.sum(portfolio_weights * asset_prices)
        
        # Step 4: ACTUALLY price with sigma_p_qrc (THE FIX!)
        pricing_result = self._price_with_sigma_p(sigma_p_qrc, B_0, strike, maturity, r)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'mode': 'QRC',
            'price_quantum': pricing_result['price_quantum'],
            'price_classical': pricing_result['price_classical'],
            'error_percent': pricing_result['error_percent'],
            'sigma_p_qrc': sigma_p_qrc,
            'sigma_p_pca': sigma_p_pca,
            'sigma_p_change_pct': adapter_result['sigma_p_change_pct'],
            'qrc_factors': qrc_factors.tolist()
        }
    
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
        Price option using standard PCA sigma_p (baseline).
        """
        adapter_result = self.adapter.prepare_for_pca_pricing(
            spot_prices=asset_prices,
            volatilities=asset_volatilities,
            correlation_matrix=correlation_matrix,
            portfolio_weights=portfolio_weights
        )
        
        sigma_p_pca = adapter_result['sigma_p_pca']
        B_0 = np.sum(portfolio_weights * asset_prices)
        
        # Price with PCA sigma_p
        pricing_result = self._price_with_sigma_p(sigma_p_pca, B_0, strike, maturity, r)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'mode': 'PCA',
            'price_quantum': pricing_result['price_quantum'],
            'price_classical': pricing_result['price_classical'],
            'error_percent': pricing_result['error_percent'],
            'sigma_p_pca': sigma_p_pca
        }
    
    def compare_qrc_vs_pca(
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
        """Compare QRC vs PCA pricing."""
        result_qrc = self.price_with_qrc(
            market_data, asset_prices, asset_volatilities,
            correlation_matrix, portfolio_weights, strike, maturity, r
        )
        
        result_pca = self.price_with_pca(
            asset_prices, asset_volatilities,
            correlation_matrix, portfolio_weights, strike, maturity, r
        )
        
        return {
            'qrc': result_qrc,
            'pca': result_pca,
            'price_diff': abs(result_qrc['price_quantum'] - result_pca['price_quantum']),
            'price_diff_pct': abs(result_qrc['price_quantum'] - result_pca['price_quantum']) / result_pca['price_quantum'] * 100
        }


# ============================================================
# FIXED INTEGRATION TEST
# ============================================================

def test_fixed_integration():
    """Test that QRC and PCA now give DIFFERENT prices."""
    
    print("\n" + "=" * 80)
    print("FIXED INTEGRATION TEST: QRC vs PCA should give DIFFERENT prices!")
    print("=" * 80)
    
    pricer = QRCIntegratedPricer(n_factors=4, M=16, beta=0.1)
    
    n_assets = 5
    asset_prices = np.full(n_assets, 100.0)
    asset_vols = np.full(n_assets, 0.20)
    weights = np.ones(n_assets) / n_assets
    
    regimes = [
        {'name': 'Calm', 'corr': 0.3, 'stress': 0.2},
        {'name': 'Stressed', 'corr': 0.8, 'stress': 0.8}
    ]
    
    print(f"\n{'Regime':<12} {'QRC Price':<12} {'PCA Price':<12} {'Diff %':<10} {'σ_p Change':<12}")
    print("-" * 60)
    
    for regime in regimes:
        corr = np.eye(n_assets) + regime['corr'] * (1 - np.eye(n_assets))
        market_data = {
            'prices': 100.0,
            'volatility': 0.20,
            'corr_change': regime['stress'] * 0.5,
            'stress': regime['stress']
        }
        
        comparison = pricer.compare_qrc_vs_pca(
            market_data, asset_prices, asset_vols,
            corr, weights, strike=100.0
        )
        
        qrc_price = comparison['qrc']['price_quantum']
        pca_price = comparison['pca']['price_quantum']
        diff_pct = comparison['price_diff_pct']
        sigma_change = comparison['qrc']['sigma_p_change_pct']
        
        print(f"{regime['name']:<12} ${qrc_price:<11.4f} ${pca_price:<11.4f} {diff_pct:<9.2f}% {sigma_change:+.2f}%")
    
    print("\n" + "=" * 80)
    if comparison['price_diff'] > 0.001:
        print("✅ FIX CONFIRMED: QRC and PCA now give DIFFERENT prices!")
    else:
        print("❌ Still identical - debugging needed")
    print("=" * 80)


if __name__ == "__main__":
    test_fixed_integration()
