"""
Model Comparison Benchmark
==========================

Compares error rates across different model configurations:
1. Classical (Stale PCA) - Baseline
2. QRC+QTC alone (without FB-IQFT quantum circuit)
3. FB-IQFT alone (without QRC+QTC enhancement)
4. Full Combined: QRC+QTC+FB-IQFT

For portfolio sizes N = 2, 5, 10, 50 and multiple market regimes.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.fb_iqft_pricing import FBIQFTPricing
from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
from qfdp.integrated.corrected_qtc_integrated_pricer import CorrectedQTCIntegratedPricer
from qfdp.unified.enhanced_factor_constructor import EnhancedFactorConstructor
from qfdp.qtc.quantum_temporal_convolution import QuantumTemporalConvolution
from qfdp.fusion.feature_fusion import FeatureFusion
from qrc import QuantumRecurrentCircuit
from scipy.stats import norm

logging.basicConfig(level=logging.WARNING)


# ============================================================================
# CONFIGURATION
# ============================================================================

PORTFOLIO_SIZES = [2, 5, 10, 50]

MARKET_REGIMES = {
    'calm': {'rho': 0.3, 'trend': 'up'},
    'medium': {'rho': 0.5, 'trend': 'flat'},
    'high': {'rho': 0.7, 'trend': 'volatile'},
    'crisis': {'rho': 0.85, 'trend': 'down'},
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_correlation_matrix(n: int, rho: float) -> np.ndarray:
    """Generate N×N correlation matrix with uniform off-diagonal correlation."""
    return np.eye(n) * (1 - rho) + rho


def generate_price_history(trend: str = 'up', base_price: float = 100.0) -> np.ndarray:
    """Generate 6-point price history based on trend."""
    if trend == 'up':
        return base_price + np.array([-2, -1, 0, 1, 2, 3])
    elif trend == 'down':
        return base_price + np.array([3, 2, 1, -1, -3, -5])
    elif trend == 'flat':
        return base_price + np.array([0, 0.1, -0.1, 0.2, 0, 0.1])
    elif trend == 'volatile':
        return base_price + np.array([0, 3, -2, 4, -3, 2])
    else:
        return np.full(6, base_price)


def compute_true_sigma_p(n: int, rho: float, base_vol: float = 0.20) -> float:
    """Compute true portfolio volatility given correlation."""
    corr = generate_correlation_matrix(n, rho)
    vol_matrix = np.diag(np.full(n, base_vol))
    cov = vol_matrix @ corr @ vol_matrix
    weights = np.ones(n) / n
    return float(np.sqrt(weights.T @ cov @ weights))


# ============================================================================
# MODEL IMPLEMENTATIONS
# ============================================================================

class ModelBenchmark:
    """
    Benchmark different model configurations.
    """
    
    def __init__(self):
        # Initialize all components
        self.fb_iqft = FBIQFTPricing(M=16, alpha=1.0)
        
        # Full integrated pricer
        self.combined_pricer = CorrectedQTCIntegratedPricer(
            self.fb_iqft, 
            qrc_beta=0.1, 
            qtc_gamma=0.05
        )
        
        # Individual components for standalone testing
        self.qrc = QuantumRecurrentCircuit(n_factors=4)
        self.qtc = QuantumTemporalConvolution(kernel_size=3, n_kernels=4)
        self.fusion = FeatureFusion(method='weighted')
        self.factor_constructor = EnhancedFactorConstructor(n_factors=4, beta=0.1, gamma=0.05)
    
    def price_classical_stale_pca(
        self, 
        n: int, 
        rho_calibrated: float, 
        rho_actual: float,
        S: float = 100.0, 
        K: float = 100.0
    ) -> Tuple[float, float]:
        """
        Price using stale PCA (morning calibration).
        Returns: (price, error_vs_true)
        """
        # Stale σ_p from morning calibration
        sigma_stale = compute_true_sigma_p(n, rho_calibrated)
        price_stale = price_call_option_corrected(S, K, 1.0, 0.05, sigma_stale)['price']
        
        # True price
        sigma_true = compute_true_sigma_p(n, rho_actual)
        price_true = price_call_option_corrected(S, K, 1.0, 0.05, sigma_true)['price']
        
        error = abs(price_stale - price_true) / price_true * 100
        return price_stale, error
    
    def price_qrc_qtc_only(
        self, 
        n: int, 
        rho_actual: float,
        trend: str,
        S: float = 100.0, 
        K: float = 100.0
    ) -> Tuple[float, float]:
        """
        Price using QRC+QTC enhancement with classical Carr-Madan.
        NO FB-IQFT quantum circuit.
        Returns: (price, error_vs_true)
        """
        base_vol = 0.20
        asset_vols = np.full(n, base_vol)
        weights = np.ones(n) / n
        corr = generate_correlation_matrix(n, rho_actual)
        price_history = generate_price_history(trend)
        
        # QRC factors
        self.qrc.reset_hidden_state()
        stress = rho_actual - 0.3
        qrc_input = {
            'prices': S,
            'volatility': base_vol,
            'corr_change': stress,
            'stress': max(0, min(1, stress * 2))
        }
        qrc_result = self.qrc.forward(qrc_input)
        qrc_factors = qrc_result.factors
        
        # QTC patterns
        qtc_patterns = self.qtc.forward_with_pooling(price_history)
        
        # Enhanced factors
        L_enhanced, D_enhanced, _ = self.factor_constructor.construct_enhanced_factors(
            qrc_factors, qtc_patterns, corr, asset_vols
        )
        
        # Enhanced σ_p
        sigma_enhanced = self.factor_constructor.compute_portfolio_volatility(
            L_enhanced, D_enhanced, weights, asset_vols,
            base_correlation=corr, qrc_factors=qrc_factors, qtc_patterns=qtc_patterns
        )
        
        # Price with classical Carr-Madan (NO FB-IQFT quantum!)
        price_qrc_qtc = price_call_option_corrected(S, K, 1.0, 0.05, sigma_enhanced)['price']
        
        # True price
        sigma_true = compute_true_sigma_p(n, rho_actual)
        price_true = price_call_option_corrected(S, K, 1.0, 0.05, sigma_true)['price']
        
        error = abs(price_qrc_qtc - price_true) / price_true * 100
        return price_qrc_qtc, error
    
    def price_fb_iqft_only(
        self, 
        n: int, 
        rho_calibrated: float,  # Use stale calibration
        rho_actual: float,      # Actual market correlation
        S: float = 100.0, 
        K: float = 100.0
    ) -> Tuple[float, float]:
        """
        Price using FB-IQFT quantum circuit with STALE PCA.
        (Using morning correlation, not real-time)
        NO QRC+QTC enhancement.
        Returns: (price, error_vs_true)
        """
        base_vol = 0.20
        asset_prices = np.full(n, S)
        asset_vols = np.full(n, base_vol)
        weights = np.ones(n) / n
        
        # Use STALE calibrated correlation (morning PCA - this is the fair comparison)
        corr_stale = generate_correlation_matrix(n, rho_calibrated)
        
        # FB-IQFT with stale PCA (standard factor decomposition)
        result = self.fb_iqft.price_option(
            asset_prices=asset_prices,
            asset_volatilities=asset_vols,
            correlation_matrix=corr_stale,  # STALE!
            portfolio_weights=weights,
            K=K,
            T=1.0,
            r=0.05
        )
        
        # FB-IQFT computes sigma_p from the stale correlation
        sigma_p_fb = result['sigma_p']
        
        # Price using classical with same σ_p
        price_fb_iqft = price_call_option_corrected(S, K, 1.0, 0.05, sigma_p_fb)['price']
        
        # True price (using actual market correlation)
        sigma_true = compute_true_sigma_p(n, rho_actual)
        price_true = price_call_option_corrected(S, K, 1.0, 0.05, sigma_true)['price']
        
        error = abs(price_fb_iqft - price_true) / price_true * 100
        return price_fb_iqft, error
    
    def price_combined_qrc_qtc_fb_iqft(
        self, 
        n: int, 
        rho_actual: float,
        trend: str,
        S: float = 100.0, 
        K: float = 100.0
    ) -> Tuple[float, float, int]:
        """
        Price using full combined QRC+QTC+FB-IQFT pipeline.
        Returns: (price, error_vs_true, circuit_depth)
        """
        base_vol = 0.20
        asset_prices = np.full(n, S)
        asset_vols = np.full(n, base_vol)
        weights = np.ones(n) / n
        corr = generate_correlation_matrix(n, rho_actual)
        price_history = generate_price_history(trend)
        
        market_data = {
            'spot_prices': asset_prices,
            'volatilities': asset_vols,
            'correlation_matrix': corr,
            'weights': weights,
            'maturity': 1.0,
            'risk_free_rate': 0.05
        }
        
        result = self.combined_pricer.price_with_full_quantum_pipeline(
            market_data, price_history, strike=K, use_quantum_circuit=True
        )
        
        price_combined = result['price_quantum']
        circuit_depth = result.get('circuit_depth', 0)
        
        # True price
        sigma_true = compute_true_sigma_p(n, rho_actual)
        price_true = price_call_option_corrected(S, K, 1.0, 0.05, sigma_true)['price']
        
        error = abs(price_combined - price_true) / price_true * 100
        return price_combined, error, circuit_depth


def run_comprehensive_comparison():
    """
    Run comprehensive comparison across all models and configurations.
    """
    print("=" * 100)
    print("COMPREHENSIVE MODEL COMPARISON BENCHMARK")
    print("=" * 100)
    print("\nComparing 4 configurations:")
    print("  1. Classical (Stale PCA) - Morning calibration, no adaptation")
    print("  2. QRC+QTC Only - Adaptive factors with classical pricing")
    print("  3. FB-IQFT Only - Quantum circuit without QRC+QTC")
    print("  4. Combined QRC+QTC+FB-IQFT - Full quantum pipeline")
    print("=" * 100)
    
    benchmark = ModelBenchmark()
    
    # Store results for summary
    all_results = []
    
    for n in PORTFOLIO_SIZES:
        print(f"\n{'='*80}")
        print(f"PORTFOLIO SIZE: N = {n}")
        print(f"{'='*80}")
        
        # Calibration correlation (morning/stale)
        rho_calibrated = 0.3
        
        print(f"\n{'Regime':<12} {'ρ_actual':<10} {'Classical':<12} {'QRC+QTC':<12} {'FB-IQFT':<12} {'Combined':<12} {'Best':<10}")
        print("-" * 80)
        
        for regime_name, regime_params in MARKET_REGIMES.items():
            rho_actual = regime_params['rho']
            trend = regime_params['trend']
            
            # 1. Classical (Stale PCA)
            _, err_classical = benchmark.price_classical_stale_pca(n, rho_calibrated, rho_actual)
            
            # 2. QRC+QTC Only
            _, err_qrc_qtc = benchmark.price_qrc_qtc_only(n, rho_actual, trend)
            
            # 3. FB-IQFT Only
            _, err_fb_iqft = benchmark.price_fb_iqft_only(n, rho_calibrated, rho_actual)
            
            # 4. Combined
            _, err_combined, depth = benchmark.price_combined_qrc_qtc_fb_iqft(n, rho_actual, trend)
            
            # Find best
            errors = {
                'Classical': err_classical,
                'QRC+QTC': err_qrc_qtc,
                'FB-IQFT': err_fb_iqft,
                'Combined': err_combined
            }
            best = min(errors, key=errors.get)
            
            print(f"{regime_name:<12} {rho_actual:<10.2f} {err_classical:<11.2f}% {err_qrc_qtc:<11.2f}% {err_fb_iqft:<11.2f}% {err_combined:<11.2f}% {best:<10}")
            
            all_results.append({
                'n': n,
                'regime': regime_name,
                'rho': rho_actual,
                'classical': err_classical,
                'qrc_qtc': err_qrc_qtc,
                'fb_iqft': err_fb_iqft,
                'combined': err_combined,
                'best': best
            })
    
    # Summary Statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    
    # Aggregate by model
    models = ['classical', 'qrc_qtc', 'fb_iqft', 'combined']
    model_labels = ['Stale PCA', 'QRC+QTC', 'FB-IQFT', 'Combined']
    
    print(f"\n{'Model':<15} {'Mean Error':<15} {'Max Error':<15} {'Wins':<10}")
    print("-" * 55)
    
    for model, label in zip(models, model_labels):
        errors = [r[model] for r in all_results]
        wins = sum(1 for r in all_results if r['best'] == label.replace('Stale PCA', 'Classical'))
        print(f"{label:<15} {np.mean(errors):<14.2f}% {np.max(errors):<14.2f}% {wins}/{len(all_results)}")
    
    # Improvement analysis
    print("\n" + "-" * 55)
    print("IMPROVEMENT OVER STALE PCA")
    print("-" * 55)
    
    for model, label in zip(models[1:], model_labels[1:]):
        improvements = [r['classical'] - r[model] for r in all_results]
        avg_improvement = np.mean(improvements)
        print(f"{label:<15} Average: {avg_improvement:+.2f}% improvement")
    
    # Stressed regime analysis (exclude calm)
    stressed_results = [r for r in all_results if r['regime'] != 'calm']
    
    print("\n" + "-" * 55)
    print("STRESSED REGIMES ONLY (Medium, High, Crisis)")
    print("-" * 55)
    
    for model, label in zip(models, model_labels):
        errors = [r[model] for r in stressed_results]
        print(f"{label:<15} Mean Error: {np.mean(errors):.2f}%")
    
    # Best model by regime
    print("\n" + "-" * 55)
    print("BEST MODEL BY REGIME")
    print("-" * 55)
    
    for regime_name in MARKET_REGIMES.keys():
        regime_results = [r for r in all_results if r['regime'] == regime_name]
        wins = {}
        for model in model_labels:
            model_key = model.lower().replace('stale pca', 'classical').replace(' ', '_').replace('+', '_')
            if model_key == 'stale_pca':
                model_key = 'classical'
            wins[model] = sum(1 for r in regime_results if r['best'] == model.replace('Stale PCA', 'Classical'))
        best = max(wins, key=wins.get)
        print(f"{regime_name:<12} Best: {best} ({wins[best]}/{len(regime_results)} wins)")
    
    print("\n" + "=" * 100)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    run_comprehensive_comparison()
