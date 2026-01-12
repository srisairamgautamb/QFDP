"""
QTC Integrated Pricer
=====================

Complete pipeline: QRC + QTC + Fusion + FB-IQFT

This combines:
- QRC: Correlation regime adaptation
- QTC: Temporal pattern extraction
- Fusion: Feature combination
- Carr-Madan: Option pricing
"""

import numpy as np
from typing import Dict, Optional
import logging
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.qtc.quantum_temporal_convolution import QuantumTemporalConvolution
from qfdp.fusion.feature_fusion import FeatureFusion
from qfdp.unified.adapter_layer import BaseModelAdapter
from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
from qrc import QuantumRecurrentCircuit

logger = logging.getLogger(__name__)


class QTCIntegratedPricer:
    """
    Complete pipeline: QRC + QTC + Fusion + FB-IQFT
    
    Key Methods:
    - price_with_qrc_only: Use only QRC (baseline)
    - price_with_qtc_only: Use only QTC
    - price_with_qrc_qtc: Use both QRC + QTC combined
    """
    
    def __init__(self, qrc_beta: float = 0.1, fusion_method: str = 'weighted'):
        """
        Args:
            qrc_beta: Beta parameter for QRC modulation
            fusion_method: 'concat' | 'weighted' | 'gating'
        """
        # QRC for correlation adaptation
        self.qrc = QuantumRecurrentCircuit(n_factors=4)
        self.adapter = BaseModelAdapter(beta=qrc_beta)
        
        # QTC for temporal patterns
        self.qtc = QuantumTemporalConvolution()
        
        # Fusion layer
        self.fusion = FeatureFusion(method=fusion_method)
        
        logger.info(f"QTCIntegratedPricer initialized: beta={qrc_beta}, fusion={fusion_method}")
    
    def price_with_qrc_only(
        self,
        market_data: Dict,
        strike: float,
        maturity: float = 1.0,
        r: float = 0.05
    ) -> Dict:
        """
        Price using QRC only (no temporal patterns).
        """
        asset_prices = market_data['spot_prices']
        asset_vols = market_data['volatilities']
        corr = market_data['correlation_matrix']
        weights = market_data.get('weights', np.ones(len(asset_prices)) / len(asset_prices))
        
        n = len(asset_prices)
        S = np.mean(asset_prices)
        
        # QRC processing
        self.qrc.reset_hidden_state()
        stress = np.mean(corr) - 0.3  # Simple stress measure
        qrc_input = {
            'prices': S,
            'volatility': np.mean(asset_vols),
            'corr_change': stress,
            'stress': max(0, min(1, stress * 2))
        }
        qrc_result = self.qrc.forward(qrc_input)
        qrc_factors = qrc_result.factors
        
        # Get σ_p from QRC
        qrc_adapted = self.adapter.prepare_for_qrc_pricing(
            asset_prices, asset_vols, corr, qrc_factors, weights
        )
        sigma_p_qrc = qrc_adapted['sigma_p_qrc']
        
        # Price
        price_result = price_call_option_corrected(S, strike, maturity, r, sigma_p_qrc)
        
        return {
            'price': price_result['price'],
            'sigma_p': sigma_p_qrc,
            'qrc_factors': qrc_factors,
            'method': 'QRC_only'
        }
    
    def price_with_qtc_only(
        self,
        price_history: np.ndarray,
        strike: float,
        sigma_base: float,
        S: float,
        maturity: float = 1.0,
        r: float = 0.05
    ) -> Dict:
        """
        Price using QTC only (no correlation adaptation).
        """
        # QTC processing
        qtc_patterns = self.qtc.forward_with_pooling(price_history)
        
        # Simple volatility adjustment based on patterns
        # High concentration in recent pattern → momentum → adjust vol
        pattern_concentration = np.max(qtc_patterns) - np.mean(qtc_patterns)
        adjustment = 1 + 0.1 * pattern_concentration
        sigma_p_qtc = sigma_base * adjustment
        
        # Price
        price_result = price_call_option_corrected(S, strike, maturity, r, sigma_p_qtc)
        
        return {
            'price': price_result['price'],
            'sigma_p': sigma_p_qtc,
            'qtc_patterns': qtc_patterns,
            'adjustment': adjustment,
            'method': 'QTC_only'
        }
    
    def price_with_qrc_qtc(
        self,
        market_data: Dict,
        price_history: np.ndarray,
        strike: float,
        maturity: float = 1.0,
        r: float = 0.05
    ) -> Dict:
        """
        Price using QRC + QTC combined (full pipeline).
        
        Args:
            market_data: Current market state (spots, vols, correlations)
            price_history: Last 6 prices for QTC
            strike: Strike price
            maturity: Time to maturity
            r: Risk-free rate
        
        Returns:
            Dict with price, volatilities, and features
        """
        asset_prices = market_data['spot_prices']
        asset_vols = market_data['volatilities']
        corr = market_data['correlation_matrix']
        weights = market_data.get('weights', np.ones(len(asset_prices)) / len(asset_prices))
        
        n = len(asset_prices)
        S = np.mean(asset_prices)
        
        # Step 1: QRC for correlation regime
        self.qrc.reset_hidden_state()
        stress = np.mean(corr) - 0.3
        qrc_input = {
            'prices': S,
            'volatility': np.mean(asset_vols),
            'corr_change': stress,
            'stress': max(0, min(1, stress * 2))
        }
        qrc_result = self.qrc.forward(qrc_input)
        qrc_factors = qrc_result.factors
        
        # Step 2: QTC for temporal patterns
        qtc_patterns = self.qtc.forward_with_pooling(price_history)
        
        # Step 3: Get base σ_p from QRC
        qrc_adapted = self.adapter.prepare_for_qrc_pricing(
            asset_prices, asset_vols, corr, qrc_factors, weights
        )
        sigma_p_qrc = qrc_adapted['sigma_p_qrc']
        
        # Step 4: Fuse features and adjust volatility
        sigma_p_combined = self.fusion.compute_volatility_adjustment(
            qrc_factors, qtc_patterns, sigma_p_qrc
        )
        
        # Step 5: Fused features (for analysis)
        fused_features = self.fusion.forward(qrc_factors, qtc_patterns)
        
        # Step 6: Price
        price_result = price_call_option_corrected(S, strike, maturity, r, sigma_p_combined)
        
        return {
            'price': price_result['price'],
            'sigma_p_qrc': sigma_p_qrc,
            'sigma_p_combined': sigma_p_combined,
            'qrc_factors': qrc_factors,
            'qtc_patterns': qtc_patterns,
            'fused_features': fused_features,
            'method': 'QRC_QTC_combined'
        }


def run_ablation_study():
    """
    Ablation study comparing:
    1. Base PCA (stale)
    2. QRC only
    3. QTC only
    4. QRC + QTC combined
    """
    print("=" * 70)
    print("QRC + QTC ABLATION STUDY")
    print("=" * 70)
    
    pricer = QTCIntegratedPricer(qrc_beta=0.1, fusion_method='weighted')
    adapter = BaseModelAdapter(beta=0.1)
    
    # Calibration on calm market
    n = 5
    asset_prices = np.full(n, 100.0)
    asset_vols = np.full(n, 0.20)
    weights = np.ones(n) / n
    
    rho_calm = 0.3
    corr_calm = np.eye(n) + rho_calm * (1 - np.eye(n))
    pca_calm = adapter.prepare_for_pca_pricing(asset_prices, asset_vols, corr_calm, weights)
    sigma_pca_stale = pca_calm['sigma_p_pca']
    
    print(f"\n📅 Morning calibration (ρ=0.3): σ_p,PCA = {sigma_pca_stale:.4f}")
    
    # Test scenarios
    scenarios = [
        {'label': 'Calm Uptrend', 'rho': 0.3, 'prices': [99.5, 100.0, 100.2, 100.5, 100.8, 101.0], 'stress': 0.2},
        {'label': 'Medium Flat', 'rho': 0.5, 'prices': [100.0, 100.1, 99.9, 100.2, 100.0, 100.1], 'stress': 0.5},
        {'label': 'High Volatile', 'rho': 0.7, 'prices': [100.0, 101.5, 99.0, 102.0, 98.5, 101.0], 'stress': 0.7},
        {'label': 'Crisis Spike', 'rho': 0.8, 'prices': [100.0, 98.0, 95.0, 97.0, 94.0, 96.0], 'stress': 0.9},
    ]
    
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    
    print(f"\n{'Scenario':<15} {'True σ_p':<10} {'PCA Err':<10} {'QRC Err':<10} {'QTC Err':<10} {'Combined':<10}")
    print("-" * 70)
    
    for scenario in scenarios:
        corr = np.eye(n) + scenario['rho'] * (1 - np.eye(n))
        price_history = np.array(scenario['prices'])
        
        # TRUE σ_p
        pca_true = adapter.prepare_for_pca_pricing(asset_prices, asset_vols, corr, weights)
        sigma_true = pca_true['sigma_p_pca']
        price_true = price_call_option_corrected(S, K, T, r, sigma_true)['price']
        
        # Method 1: Stale PCA
        price_pca = price_call_option_corrected(S, K, T, r, sigma_pca_stale)['price']
        error_pca = abs(price_pca - price_true) / price_true * 100
        
        # Method 2: QRC only
        market_data = {
            'spot_prices': asset_prices,
            'volatilities': asset_vols,
            'correlation_matrix': corr,
            'weights': weights
        }
        result_qrc = pricer.price_with_qrc_only(market_data, K, T, r)
        error_qrc = abs(result_qrc['price'] - price_true) / price_true * 100
        
        # Method 3: QTC only
        result_qtc = pricer.price_with_qtc_only(price_history, K, sigma_true, S, T, r)
        error_qtc = abs(result_qtc['price'] - price_true) / price_true * 100
        
        # Method 4: QRC + QTC combined
        result_combined = pricer.price_with_qrc_qtc(market_data, price_history, K, T, r)
        error_combined = abs(result_combined['price'] - price_true) / price_true * 100
        
        print(f"{scenario['label']:<15} {sigma_true:<10.4f} {error_pca:<9.2f}% {error_qrc:<9.2f}% {error_qtc:<9.2f}% {error_combined:<9.2f}%")
    
    print("\n" + "=" * 70)
    print("✅ ABLATION STUDY COMPLETE")
    print("=" * 70)


# ============================================================================
# VALIDATION
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ablation_study()
