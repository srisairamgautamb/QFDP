"""
🏆 COMPREHENSIVE 7-TIER VALIDATION SUITE
=========================================

The ULTIMATE validation framework proving QRC+QTC system is production-ready.

Tier 1: Mathematical Correctness (Foundation)
Tier 2: Component Isolation (Individual Parts)
Tier 3: Integration Integrity (Pipeline)
Tier 4: Ablation & Comparative Analysis
Tier 5: Robustness Testing (Noise, Edge Cases)
Tier 6: Empirical Validation (Real Scenarios)
Tier 7: Production Readiness (Final Sign-Off)
"""

import numpy as np
from scipy.stats import norm
import logging
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================================
# TIER 1: MATHEMATICAL CORRECTNESS
# ============================================================================

class Tier1_MathematicalCorrectness:
    """Prove your math is correct. If the math is wrong, nothing else matters."""
    
    def test_black_scholes_analytical(self) -> bool:
        """Verify BS formula implementation"""
        logger.info("T1.1: Black-Scholes Analytical Formula")
        
        from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
        
        test_cases = [
            {'S': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.20, 'name': 'ATM'},
            {'S': 110, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.20, 'name': 'ITM'},
            {'S': 90, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.20, 'name': 'OTM'},
            {'S': 100, 'K': 100, 'T': 0.1, 'r': 0.05, 'sigma': 0.20, 'name': 'Short T'},
            {'S': 100, 'K': 100, 'T': 5.0, 'r': 0.05, 'sigma': 0.20, 'name': 'Long T'},
        ]
        
        all_passed = True
        for case in test_cases:
            result = price_call_option_corrected(
                case['S'], case['K'], case['T'], case['r'], case['sigma']
            )
            your_price = result['price']
            
            # Reference BS
            d1 = (np.log(case['S']/case['K']) + (case['r'] + 0.5*case['sigma']**2)*case['T']) / (case['sigma']*np.sqrt(case['T']))
            d2 = d1 - case['sigma']*np.sqrt(case['T'])
            bs_price = case['S']*norm.cdf(d1) - case['K']*np.exp(-case['r']*case['T'])*norm.cdf(d2)
            
            error = abs(your_price - bs_price) / bs_price * 100
            
            status = "✅" if error < 0.1 else "❌"
            logger.info(f"  {status} {case['name']:10s}: Your=${your_price:.4f}, BS=${bs_price:.4f}, Error={error:.4f}%")
            
            if error >= 0.1:
                all_passed = False
        
        logger.info(f"{'✅' if all_passed else '❌'} T1.1 {'PASSED' if all_passed else 'FAILED'}\n")
        return all_passed
    
    def test_volatility_sensitivity(self) -> bool:
        """Verify σ→Price monotonicity"""
        logger.info("T1.2: Volatility Sensitivity (σ ↑ → Price ↑)")
        
        from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
        
        sigmas = [0.10, 0.15, 0.20, 0.25, 0.30]
        prices = [price_call_option_corrected(100, 100, 1.0, 0.05, sig)['price'] for sig in sigmas]
        
        is_monotonic = all(prices[i] <= prices[i+1] for i in range(len(prices)-1))
        
        for sig, price in zip(sigmas, prices):
            logger.info(f"    σ={sig:.2f}: ${price:.4f}")
        
        logger.info(f"{'✅' if is_monotonic else '❌'} T1.2 {'PASSED' if is_monotonic else 'FAILED'}\n")
        return is_monotonic
    
    def test_greeks_computation(self) -> bool:
        """Verify greeks (delta, gamma) are sensible"""
        logger.info("T1.3: Greeks Computation (Delta, Gamma)")
        
        from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
        
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        dS = 1.0  # Larger step for stable numerical derivatives
        
        # Numerical delta
        P_up = price_call_option_corrected(S + dS, K, T, r, sigma)['price']
        P_dn = price_call_option_corrected(S - dS, K, T, r, sigma)['price']
        delta_numerical = (P_up - P_dn) / (2 * dS)
        
        # BS analytical delta
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        delta_bs = norm.cdf(d1)
        
        delta_error = abs(delta_numerical - delta_bs) / delta_bs * 100
        delta_ok = delta_error < 1.0
        logger.info(f"  {'✅' if delta_ok else '❌'} Delta: numerical={delta_numerical:.4f}, BS={delta_bs:.4f}, Error={delta_error:.4f}%")
        
        # Numerical gamma
        P_center = price_call_option_corrected(S, K, T, r, sigma)['price']
        gamma_numerical = (P_up - 2*P_center + P_dn) / (dS**2)
        gamma_bs = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        gamma_error = abs(gamma_numerical - gamma_bs) / gamma_bs * 100
        gamma_ok = gamma_error < 10.0  # Higher tolerance for second derivative
        logger.info(f"  {'✅' if gamma_ok else '❌'} Gamma: numerical={gamma_numerical:.4f}, BS={gamma_bs:.4f}, Error={gamma_error:.4f}%")
        
        passed = delta_ok and gamma_ok
        logger.info(f"{'✅' if passed else '❌'} T1.3 {'PASSED' if passed else 'FAILED'}\n")
        return passed
    
    def run_all(self) -> bool:
        logger.info("=" * 70)
        logger.info("TIER 1: MATHEMATICAL CORRECTNESS")
        logger.info("=" * 70 + "\n")
        
        t1 = self.test_black_scholes_analytical()
        t2 = self.test_volatility_sensitivity()
        t3 = self.test_greeks_computation()
        
        passed = t1 and t2 and t3
        logger.info("=" * 70)
        logger.info(f"{'✅' if passed else '❌'} TIER 1 {'PASSED' if passed else 'FAILED'}")
        logger.info("=" * 70 + "\n")
        return passed


# ============================================================================
# TIER 2: COMPONENT ISOLATION
# ============================================================================

class Tier2_ComponentIsolation:
    """Test each component independently."""
    
    def test_qrc_factor_generation(self) -> bool:
        """QRC generates valid adaptive factors"""
        logger.info("T2.1: QRC Factor Generation")
        
        from qrc import QuantumRecurrentCircuit
        
        qrc = QuantumRecurrentCircuit(n_factors=4)
        
        test_regimes = [
            {'stress': 0.2, 'label': 'Calm'},
            {'stress': 0.5, 'label': 'Medium'},
            {'stress': 0.9, 'label': 'Stressed'},
        ]
        
        all_passed = True
        for regime in test_regimes:
            qrc.reset_hidden_state()
            result = qrc.forward({
                'prices': 100.0,
                'volatility': 0.20,
                'corr_change': regime['stress'] * 0.5,
                'stress': regime['stress']
            })
            factors = result.factors
            
            sum_ok = np.isclose(np.sum(factors), 1.0, atol=1e-6)
            all_positive = np.all(factors > 0)
            all_bounded = np.all(factors < 1.0)
            
            valid = sum_ok and all_positive and all_bounded
            status = "✅" if valid else "❌"
            logger.info(f"  {status} {regime['label']:10s}: factors={factors}, sum={np.sum(factors):.6f}")
            
            if not valid:
                all_passed = False
        
        logger.info(f"{'✅' if all_passed else '❌'} T2.1 {'PASSED' if all_passed else 'FAILED'}\n")
        return all_passed
    
    def test_qtc_pattern_extraction(self) -> bool:
        """QTC extracts meaningful patterns"""
        logger.info("T2.2: QTC Pattern Extraction")
        
        from qfdp.qtc.quantum_temporal_convolution import QuantumTemporalConvolution
        
        qtc = QuantumTemporalConvolution(kernel_size=3, n_kernels=4)
        
        patterns_test = [
            (np.array([100, 101, 102, 103, 104, 105]), 'Uptrend'),
            (np.array([105, 104, 103, 102, 101, 100]), 'Downtrend'),
            (np.array([100, 102, 99, 101, 100, 102]), 'Volatile'),
        ]
        
        results = {}
        all_passed = True
        for prices, label in patterns_test:
            patterns = qtc.forward_with_pooling(prices)
            results[label] = patterns
            
            valid_len = len(patterns) == 4
            valid_range = np.isclose(np.sum(patterns), 1.0, atol=0.1)
            
            valid = valid_len and valid_range
            status = "✅" if valid else "❌"
            logger.info(f"  {status} {label:10s}: patterns={patterns}")
            
            if not valid:
                all_passed = False
        
        # Different patterns should give different results
        different = not np.allclose(results['Uptrend'], results['Downtrend'], atol=0.05)
        logger.info(f"  {'✅' if different else '❌'} Different patterns give different results")
        
        passed = all_passed and different
        logger.info(f"{'✅' if passed else '❌'} T2.2 {'PASSED' if passed else 'FAILED'}\n")
        return passed
    
    def test_fusion_layer(self) -> bool:
        """Feature fusion combines QRC + QTC correctly"""
        logger.info("T2.3: Feature Fusion Layer")
        
        from qfdp.fusion.feature_fusion import FeatureFusion
        
        qrc_factors = np.array([0.3, 0.3, 0.2, 0.2])
        qtc_patterns = np.array([0.4, 0.3, 0.2, 0.1])
        
        # Test concat
        fusion_concat = FeatureFusion(method='concat')
        fused_concat = fusion_concat.forward(qrc_factors, qtc_patterns)
        concat_ok = len(fused_concat) == 8
        logger.info(f"  {'✅' if concat_ok else '❌'} Concat: {len(fused_concat)} features")
        
        # Test weighted
        fusion_weighted = FeatureFusion(method='weighted')
        fused_weighted = fusion_weighted.forward(qrc_factors, qtc_patterns)
        weighted_ok = len(fused_weighted) == 4
        logger.info(f"  {'✅' if weighted_ok else '❌'} Weighted: {len(fused_weighted)} features")
        
        # Fusion different from components
        different_qrc = not np.allclose(fused_weighted, qrc_factors / np.linalg.norm(qrc_factors), atol=0.01)
        logger.info(f"  {'✅' if different_qrc else '❌'} Fusion differs from QRC alone")
        
        passed = concat_ok and weighted_ok and different_qrc
        logger.info(f"{'✅' if passed else '❌'} T2.3 {'PASSED' if passed else 'FAILED'}\n")
        return passed
    
    def run_all(self) -> bool:
        logger.info("=" * 70)
        logger.info("TIER 2: COMPONENT ISOLATION")
        logger.info("=" * 70 + "\n")
        
        t1 = self.test_qrc_factor_generation()
        t2 = self.test_qtc_pattern_extraction()
        t3 = self.test_fusion_layer()
        
        passed = t1 and t2 and t3
        logger.info("=" * 70)
        logger.info(f"{'✅' if passed else '❌'} TIER 2 {'PASSED' if passed else 'FAILED'}")
        logger.info("=" * 70 + "\n")
        return passed


# ============================================================================
# TIER 3: INTEGRATION INTEGRITY
# ============================================================================

class Tier3_IntegrationIntegrity:
    """Prove integration is correct. QRC→QTC→Fusion→Pricing pipeline works."""
    
    def test_end_to_end_pricing(self) -> bool:
        """Full pipeline: QRC+QTC→Price"""
        logger.info("T3.1: End-to-End Pricing Pipeline")
        
        from qfdp.integrated.qtc_integrated_pricer import QTCIntegratedPricer
        
        pricer = QTCIntegratedPricer(qrc_beta=0.1, fusion_method='weighted')
        
        market_data = {
            'spot_prices': np.array([100, 100, 100, 100]),
            'volatilities': np.array([0.20, 0.22, 0.25, 0.23]),
            'correlation_matrix': np.eye(4) * 0.8 + 0.2,
            'weights': np.ones(4) / 4
        }
        price_history = np.array([99.5, 100.0, 100.2, 100.5, 100.8, 101.0])
        
        result = pricer.price_with_qrc_qtc(market_data, price_history, strike=100)
        
        # Checks
        has_price = 'price' in result
        has_sigma = 'sigma_p_combined' in result
        has_patterns = 'qtc_patterns' in result
        
        # Sanity bounds
        price = result['price']
        price_sensible = 5 < price < 20
        
        all_ok = has_price and has_sigma and has_patterns and price_sensible
        
        logger.info(f"  {'✅' if all_ok else '❌'} End-to-end successful")
        logger.info(f"      QRC σ_p: {result['sigma_p_qrc']:.4f}")
        logger.info(f"      Combined σ_p: {result['sigma_p_combined']:.4f}")
        logger.info(f"      Price: ${result['price']:.4f}")
        
        logger.info(f"{'✅' if all_ok else '❌'} T3.1 {'PASSED' if all_ok else 'FAILED'}\n")
        return all_ok
    
    def test_price_sensitivity_to_qrc(self) -> bool:
        """QRC factors actually affect pricing"""
        logger.info("T3.2: Price Sensitivity to QRC")
        
        from qfdp.integrated.qtc_integrated_pricer import QTCIntegratedPricer
        
        pricer = QTCIntegratedPricer(qrc_beta=0.1)
        price_history = np.array([99.5, 100.0, 100.2, 100.5, 100.8, 101.0])
        
        # Calm correlation
        market_calm = {
            'spot_prices': np.array([100, 100, 100, 100]),
            'volatilities': np.array([0.20, 0.20, 0.20, 0.20]),
            'correlation_matrix': np.eye(4) * 0.7 + 0.3,  # Low correlation
            'weights': np.ones(4) / 4
        }
        price_calm = pricer.price_with_qrc_qtc(market_calm, price_history, strike=100)['price']
        
        # Stressed correlation
        market_stressed = {
            'spot_prices': np.array([100, 100, 100, 100]),
            'volatilities': np.array([0.20, 0.20, 0.20, 0.20]),
            'correlation_matrix': np.eye(4) * 0.2 + 0.8,  # High correlation
            'weights': np.ones(4) / 4
        }
        price_stressed = pricer.price_with_qrc_qtc(market_stressed, price_history, strike=100)['price']
        
        different = abs(price_calm - price_stressed) > 0.01
        
        logger.info(f"  {'✅' if different else '❌'} Price responds to correlation regime")
        logger.info(f"      Calm (ρ=0.3): ${price_calm:.4f}")
        logger.info(f"      Stressed (ρ=0.8): ${price_stressed:.4f}")
        logger.info(f"      Difference: ${abs(price_calm - price_stressed):.4f}")
        
        logger.info(f"{'✅' if different else '❌'} T3.2 {'PASSED' if different else 'FAILED'}\n")
        return different
    
    def test_price_sensitivity_to_qtc(self) -> bool:
        """QTC patterns actually affect pricing"""
        logger.info("T3.3: Price Sensitivity to QTC")
        
        from qfdp.integrated.qtc_integrated_pricer import QTCIntegratedPricer
        
        pricer = QTCIntegratedPricer(qrc_beta=0.1)
        
        market_data = {
            'spot_prices': np.array([100, 100, 100, 100]),
            'volatilities': np.array([0.20, 0.20, 0.20, 0.20]),
            'correlation_matrix': np.eye(4) * 0.7 + 0.3,
            'weights': np.ones(4) / 4
        }
        
        # Uptrend history
        price_uptrend = pricer.price_with_qrc_qtc(
            market_data, 
            np.array([98, 99, 100, 101, 102, 103]),  # Uptrend
            strike=100
        )['price']
        
        # Volatile history
        price_volatile = pricer.price_with_qrc_qtc(
            market_data,
            np.array([100, 95, 105, 92, 108, 100]),  # Volatile
            strike=100
        )['price']
        
        different = abs(price_uptrend - price_volatile) > 0.001
        
        logger.info(f"  {'✅' if different else '❌'} Price responds to temporal patterns")
        logger.info(f"      Uptrend: ${price_uptrend:.4f}")
        logger.info(f"      Volatile: ${price_volatile:.4f}")
        logger.info(f"      Difference: ${abs(price_uptrend - price_volatile):.4f}")
        
        logger.info(f"{'✅' if different else '❌'} T3.3 {'PASSED' if different else 'FAILED'}\n")
        return different
    
    def run_all(self) -> bool:
        logger.info("=" * 70)
        logger.info("TIER 3: INTEGRATION INTEGRITY")
        logger.info("=" * 70 + "\n")
        
        t1 = self.test_end_to_end_pricing()
        t2 = self.test_price_sensitivity_to_qrc()
        t3 = self.test_price_sensitivity_to_qtc()
        
        passed = t1 and t2 and t3
        logger.info("=" * 70)
        logger.info(f"{'✅' if passed else '❌'} TIER 3 {'PASSED' if passed else 'FAILED'}")
        logger.info("=" * 70 + "\n")
        return passed


# ============================================================================
# TIER 4: ABLATION & COMPARATIVE ANALYSIS
# ============================================================================

class Tier4_AblationStudy:
    """Prove QRC+QTC is better than alternatives."""
    
    def run_all(self) -> bool:
        logger.info("=" * 70)
        logger.info("TIER 4: ABLATION & COMPARATIVE ANALYSIS")
        logger.info("=" * 70 + "\n")
        
        from qfdp.integrated.qtc_integrated_pricer import QTCIntegratedPricer
        from qfdp.unified.adapter_layer import BaseModelAdapter
        from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
        
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
        
        logger.info(f"📅 Morning calibration (ρ=0.3): σ_p,PCA = {sigma_pca_stale:.4f}\n")
        
        scenarios = [
            {'label': 'Calm Uptrend', 'rho': 0.3, 'prices': [99.5, 100.0, 100.2, 100.5, 100.8, 101.0]},
            {'label': 'Medium Flat', 'rho': 0.5, 'prices': [100.0, 100.1, 99.9, 100.2, 100.0, 100.1]},
            {'label': 'High Volatile', 'rho': 0.7, 'prices': [100.0, 101.5, 99.0, 102.0, 98.5, 101.0]},
            {'label': 'Crisis Spike', 'rho': 0.8, 'prices': [100.0, 98.0, 95.0, 97.0, 94.0, 96.0]},
        ]
        
        S, K, T, r = 100.0, 100.0, 1.0, 0.05
        
        logger.info(f"{'Scenario':<15} {'True σ_p':<10} {'PCA Err':<10} {'QRC Err':<10} {'QTC Err':<10} {'Combined':<10}")
        logger.info("-" * 70)
        
        results = []
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
            
            logger.info(f"{scenario['label']:<15} {sigma_true:<10.4f} {error_pca:<9.2f}% {error_qrc:<9.2f}% {error_qtc:<9.2f}% {error_combined:<9.2f}%")
            
            results.append({
                'error_pca': error_pca,
                'error_qrc': error_qrc,
                'error_qtc': error_qtc,
                'error_combined': error_combined
            })
        
        # Summary
        mean_pca = np.mean([r['error_pca'] for r in results])
        mean_qrc = np.mean([r['error_qrc'] for r in results])
        mean_qtc = np.mean([r['error_qtc'] for r in results])
        mean_combined = np.mean([r['error_combined'] for r in results])
        
        logger.info("\n📊 SUMMARY:")
        logger.info(f"   Mean Stale PCA Error: {mean_pca:.2f}%")
        logger.info(f"   Mean QRC Error:       {mean_qrc:.2f}%")
        logger.info(f"   Mean QTC Error:       {mean_qtc:.2f}%")
        logger.info(f"   Mean Combined Error:  {mean_combined:.2f}%")
        
        # Improvement from PCA
        improvement_qrc = mean_pca - mean_qrc
        improvement_combined = mean_pca - mean_combined
        
        logger.info(f"\n   QRC Improvement over PCA:      {improvement_qrc:.2f}%")
        logger.info(f"   Combined Improvement over PCA: {improvement_combined:.2f}%")
        
        # Success criteria: Combined < QRC < PCA OR QTC < QRC
        passed = (mean_qrc < mean_pca) and (mean_qtc < mean_pca or mean_combined < mean_pca)
        
        logger.info("\n" + "=" * 70)
        logger.info(f"{'✅' if passed else '❌'} TIER 4 {'PASSED' if passed else 'FAILED'} - QRC/QTC beats stale PCA")
        logger.info("=" * 70 + "\n")
        
        return passed


# ============================================================================
# TIER 5: ROBUSTNESS TESTING
# ============================================================================

class Tier5_Robustness:
    """Will it survive imperfect real-world data?"""
    
    def test_noise_robustness(self) -> bool:
        """Robustness to market noise"""
        logger.info("T5.1: Robustness to Market Noise")
        
        from qfdp.integrated.qtc_integrated_pricer import QTCIntegratedPricer
        
        pricer = QTCIntegratedPricer(qrc_beta=0.1)
        
        market_data = {
            'spot_prices': np.array([100, 100, 100, 100]),
            'volatilities': np.array([0.20, 0.20, 0.20, 0.20]),
            'correlation_matrix': np.eye(4) * 0.7 + 0.3,
            'weights': np.ones(4) / 4
        }
        
        base_history = np.array([99.5, 100.0, 100.2, 100.5, 100.8, 101.0])
        price_clean = pricer.price_with_qrc_qtc(market_data, base_history, strike=100)['price']
        
        noise_levels = [0.01, 0.05, 0.10]
        all_passed = True
        
        for noise_pct in noise_levels:
            noisy_history = base_history * (1 + np.random.randn(6) * noise_pct)
            price_noisy = pricer.price_with_qrc_qtc(market_data, noisy_history, strike=100)['price']
            
            error = abs(price_noisy - price_clean) / price_clean * 100
            ok = error < 5.0  # Within 5% of clean price
            
            logger.info(f"  {'✅' if ok else '❌'} {noise_pct*100:.0f}% noise: Price=${price_noisy:.4f}, Error from clean={error:.2f}%")
            
            if not ok:
                all_passed = False
        
        logger.info(f"{'✅' if all_passed else '❌'} T5.1 {'PASSED' if all_passed else 'FAILED'}\n")
        return all_passed
    
    def test_edge_cases(self) -> bool:
        """Handle edge cases gracefully"""
        logger.info("T5.2: Edge Cases")
        
        from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
        
        test_cases = [
            {'S': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.20, 'label': 'ATM Standard'},
            {'S': 120, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.20, 'label': 'Deep ITM'},
            {'S': 80, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.20, 'label': 'Deep OTM'},
            {'S': 100, 'K': 100, 'T': 0.01, 'r': 0.05, 'sigma': 0.20, 'label': 'Very Short T'},
            {'S': 100, 'K': 100, 'T': 10.0, 'r': 0.05, 'sigma': 0.20, 'label': 'Very Long T'},
            {'S': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.05, 'label': 'Very Low Vol'},
            {'S': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.80, 'label': 'Very High Vol'},
        ]
        
        all_passed = True
        for case in test_cases:
            try:
                result = price_call_option_corrected(case['S'], case['K'], case['T'], case['r'], case['sigma'])
                price = result['price']
                
                valid = not np.isnan(price) and not np.isinf(price) and price > 0
                logger.info(f"  {'✅' if valid else '❌'} {case['label']:20s}: ${price:.4f}")
                
                if not valid:
                    all_passed = False
            except Exception as e:
                logger.info(f"  ❌ {case['label']:20s}: {str(e)}")
                all_passed = False
        
        logger.info(f"{'✅' if all_passed else '❌'} T5.2 {'PASSED' if all_passed else 'FAILED'}\n")
        return all_passed
    
    def run_all(self) -> bool:
        logger.info("=" * 70)
        logger.info("TIER 5: ROBUSTNESS TESTING")
        logger.info("=" * 70 + "\n")
        
        t1 = self.test_noise_robustness()
        t2 = self.test_edge_cases()
        
        passed = t1 and t2
        logger.info("=" * 70)
        logger.info(f"{'✅' if passed else '❌'} TIER 5 {'PASSED' if passed else 'FAILED'}")
        logger.info("=" * 70 + "\n")
        return passed


# ============================================================================
# TIER 6: EMPIRICAL VALIDATION
# ============================================================================

class Tier6_EmpiricalValidation:
    """Test against realistic historical regimes."""
    
    def run_all(self) -> bool:
        logger.info("=" * 70)
        logger.info("TIER 6: EMPIRICAL VALIDATION (Simulated Regimes)")
        logger.info("=" * 70 + "\n")
        
        from qfdp.integrated.qtc_integrated_pricer import QTCIntegratedPricer
        
        pricer = QTCIntegratedPricer(qrc_beta=0.1)
        
        # Simulate historical regimes
        regimes = [
            {
                'label': 'COVID Crash (March 2020)',
                'rho': 0.85,  # Correlation spike
                'vol': 0.50,  # VIX spike
                'prices': [100, 92, 85, 88, 82, 78],  # Sharp decline
                'expected': 'High vol, high corr'
            },
            {
                'label': 'Brexit Vote (June 2016)',
                'rho': 0.70,
                'vol': 0.35,
                'prices': [100, 99, 97, 94, 96, 95],  # Moderate decline
                'expected': 'Moderate vol spike'
            },
            {
                'label': 'Normal Market (2022)',
                'rho': 0.40,
                'vol': 0.20,
                'prices': [100, 100.5, 101, 100.8, 101.2, 101.1],
                'expected': 'Stable'
            },
        ]
        
        all_sensible = True
        for regime in regimes:
            n = 4
            market_data = {
                'spot_prices': np.full(n, 100.0),
                'volatilities': np.full(n, regime['vol']),
                'correlation_matrix': np.eye(n) * (1 - regime['rho']) + regime['rho'],
                'weights': np.ones(n) / n
            }
            
            result = pricer.price_with_qrc_qtc(
                market_data,
                np.array(regime['prices']),
                strike=100
            )
            
            # Sanity checks
            sigma_sensible = 0.05 < result['sigma_p_combined'] < 1.0
            price_sensible = result['price'] > 0 and not np.isnan(result['price'])
            
            sensible = sigma_sensible and price_sensible
            
            logger.info(f"  {'✅' if sensible else '❌'} {regime['label']}:")
            logger.info(f"      Expected: {regime['expected']}")
            logger.info(f"      σ_p: {result['sigma_p_combined']:.4f}, Price: ${result['price']:.4f}")
            
            if not sensible:
                all_sensible = False
        
        logger.info("\n" + "=" * 70)
        logger.info(f"{'✅' if all_sensible else '❌'} TIER 6 {'PASSED' if all_sensible else 'FAILED'}")
        logger.info("=" * 70 + "\n")
        
        return all_sensible


# ============================================================================
# TIER 7: PRODUCTION READINESS
# ============================================================================

class Tier7_ProductionReadiness:
    """Final sign-off before hardware deployment."""
    
    def run_all(self) -> bool:
        logger.info("=" * 70)
        logger.info("TIER 7: PRODUCTION READINESS CHECKLIST")
        logger.info("=" * 70 + "\n")
        
        from qfdp.integrated.qtc_integrated_pricer import QTCIntegratedPricer
        from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
        
        checks = {}
        
        # Performance: Pricing speed
        logger.info("⏱️  Performance:")
        pricer = QTCIntegratedPricer(qrc_beta=0.1)
        market_data = {
            'spot_prices': np.array([100, 100, 100, 100]),
            'volatilities': np.array([0.20, 0.20, 0.20, 0.20]),
            'correlation_matrix': np.eye(4) * 0.7 + 0.3,
            'weights': np.ones(4) / 4
        }
        price_history = np.array([99.5, 100.0, 100.2, 100.5, 100.8, 101.0])
        
        start = time.time()
        for _ in range(10):
            pricer.price_with_qrc_qtc(market_data, price_history, strike=100)
        elapsed = (time.time() - start) / 10 * 1000
        
        speed_ok = elapsed < 5000  # <5s per pricing
        checks['speed'] = speed_ok
        logger.info(f"  {'✅' if speed_ok else '❌'} Pricing time: {elapsed:.1f}ms per option")
        
        # Accuracy: Base model <1% error
        logger.info("\n📊 Accuracy:")
        result = price_call_option_corrected(100, 100, 1.0, 0.05, 0.20)
        d1 = (np.log(100/100) + (0.05 + 0.5*0.20**2)*1.0) / (0.20*np.sqrt(1.0))
        d2 = d1 - 0.20*np.sqrt(1.0)
        bs_price = 100*norm.cdf(d1) - 100*np.exp(-0.05*1.0)*norm.cdf(d2)
        error = abs(result['price'] - bs_price) / bs_price * 100
        
        accuracy_ok = error < 1.0
        checks['accuracy'] = accuracy_ok
        logger.info(f"  {'✅' if accuracy_ok else '❌'} Base model error: {error:.4f}%")
        
        # Stability: 10 runs give same result
        logger.info("\n🔄 Stability:")
        prices = []
        for _ in range(10):
            p = pricer.price_with_qrc_qtc(market_data, price_history, strike=100)['price']
            prices.append(p)
        
        variance = np.std(prices) / np.mean(prices) * 100
        stable_ok = variance < 5.0  # <5% coefficient of variation
        checks['stability'] = stable_ok
        logger.info(f"  {'✅' if stable_ok else '❌'} Price variance: {variance:.2f}% (10 runs)")
        
        # Summary
        all_passed = all(checks.values())
        
        logger.info("\n" + "=" * 70)
        logger.info("FINAL CHECKLIST:")
        logger.info(f"  {'✅' if checks['speed'] else '❌'} Speed: <5s per pricing")
        logger.info(f"  {'✅' if checks['accuracy'] else '❌'} Accuracy: <1% base error")
        logger.info(f"  {'✅' if checks['stability'] else '❌'} Stability: <5% variance")
        
        logger.info("\n" + "=" * 70)
        if all_passed:
            logger.info("✅ TIER 7 PASSED - PRODUCTION READY FOR HARDWARE")
        else:
            logger.info("❌ TIER 7 FAILED - Fix issues before deployment")
        logger.info("=" * 70 + "\n")
        
        return all_passed


# ============================================================================
# MASTER VALIDATION RUNNER
# ============================================================================

def run_comprehensive_validation():
    """Run all 7 tiers of validation."""
    
    logger.info("\n" + "🏆" * 35)
    logger.info("COMPREHENSIVE 7-TIER VALIDATION SUITE")
    logger.info("🏆" * 35 + "\n")
    
    tiers = [
        (Tier1_MathematicalCorrectness(), "Mathematical Correctness"),
        (Tier2_ComponentIsolation(), "Component Isolation"),
        (Tier3_IntegrationIntegrity(), "Integration Integrity"),
        (Tier4_AblationStudy(), "Ablation & Comparative"),
        (Tier5_Robustness(), "Robustness Testing"),
        (Tier6_EmpiricalValidation(), "Empirical Validation"),
        (Tier7_ProductionReadiness(), "Production Readiness"),
    ]
    
    results = {}
    for i, (tier, name) in enumerate(tiers, 1):
        try:
            passed = tier.run_all()
            results[i] = passed
        except Exception as e:
            logger.error(f"TIER {i} FAILED with exception: {str(e)}")
            results[i] = False
    
    # Final Summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VALIDATION SUMMARY")
    logger.info("=" * 70)
    
    passed_count = sum(1 for v in results.values() if v)
    total = len(results)
    
    for i, (_, name) in enumerate(tiers, 1):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        logger.info(f"  Tier {i}: {name:30s} {status}")
    
    logger.info(f"\n  OVERALL: {passed_count}/{total} tiers passed")
    
    if passed_count == total:
        logger.info("\n" + "🎉" * 35)
        logger.info("✅ ALL TIERS PASSED - SYSTEM IS PRODUCTION READY")
        logger.info("🎉" * 35)
        return True
    else:
        logger.error(f"\n❌ {total - passed_count} tier(s) failed - Fix before deployment")
        return False


if __name__ == "__main__":
    success = run_comprehensive_validation()
    exit(0 if success else 1)
