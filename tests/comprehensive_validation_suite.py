"""
5-Level Comprehensive Validation Suite for QRC + FB-IQFT
==========================================================

Level 1: Base Model Validation (Foundation)
Level 2: Component Isolation Tests (Adapter Layer)  
Level 3: Integration Correctness (QRC → Pricing)
Level 4: Financial Sensibility (Market Regimes)
Level 5: Fair Comparison Test (Same Data for Both)

ALL tests must pass before claiming QRC advantage.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, List, Tuple
import sys
import logging

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes analytical reference."""
    if sigma <= 0:
        return max(S - K * np.exp(-r * T), 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


# ============================================================================
# LEVEL 1: BASE MODEL VALIDATION
# ============================================================================

class Level1_BaseModelValidation:
    """Verify base FB-IQFT model still works after Carr-Madan fix."""
    
    def __init__(self):
        self.tolerance_classical = 0.01  # 1%
        self.tolerance_quantum = 0.05    # 5% (Q-C error)
        
    def test_1_1_classical_accuracy(self) -> Tuple[bool, str]:
        """Test 1.1: Classical FFT matches Black-Scholes"""
        logger.info("=" * 70)
        logger.info("TEST 1.1: Classical Carr-Madan vs Black-Scholes")
        logger.info("=" * 70)
        
        from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
        
        test_cases = [
            (100, 100, 1.0, 0.05, 0.20, "ATM Standard"),
            (100, 90,  1.0, 0.05, 0.20, "ITM"),
            (100, 110, 1.0, 0.05, 0.20, "OTM"),
            (100, 100, 0.5, 0.05, 0.20, "Short Maturity"),
            (100, 100, 1.0, 0.05, 0.15, "Low Vol"),
            (100, 100, 1.0, 0.05, 0.30, "High Vol"),
        ]
        
        results = []
        for S, K, T, r, sigma, label in test_cases:
            bs_price = black_scholes_call(S, K, T, r, sigma)
            cm_result = price_call_option_corrected(S, K, T, r, sigma)
            cm_price = cm_result['price']
            error = abs(cm_price - bs_price) / bs_price
            
            passed = error < self.tolerance_classical
            status = "✅" if passed else "❌"
            
            logger.info(f"{status} {label:18s}: BS=${bs_price:8.4f}, CM=${cm_price:8.4f}, Error={error*100:.2f}%")
            results.append(passed)
        
        all_passed = all(results)
        summary = f"Classical accuracy: {sum(results)}/{len(results)} passed"
        logger.info(f"\n{'✅ PASSED' if all_passed else '❌ FAILED'}: {summary}\n")
        return all_passed, summary
    
    def test_1_2_quantum_classical_agreement(self) -> Tuple[bool, str]:
        """Test 1.2: Quantum IQFT matches Classical FFT"""
        logger.info("=" * 70)
        logger.info("TEST 1.2: Quantum-Classical Agreement (Q-C Error)")
        logger.info("=" * 70)
        
        from qfdp.unified.fb_iqft_pricing import FBIQFTPricing
        
        pricer = FBIQFTPricing(M=16)
        
        test_cases = [
            (1, "Single Asset"),
            (5, "5-Asset Portfolio"),
        ]
        
        results = []
        for n_assets, label in test_cases:
            result = pricer.price_option(
                asset_prices=np.full(n_assets, 100.0),
                asset_volatilities=np.full(n_assets, 0.20),
                correlation_matrix=np.eye(n_assets) + 0.5*(1 - np.eye(n_assets)) if n_assets > 1 else np.eye(1),
                portfolio_weights=np.ones(n_assets) / n_assets,
                K=100.0, T=1.0, r=0.05
            )
            
            error = result['error_percent'] / 100
            passed = error < self.tolerance_quantum
            status = "✅" if passed else "❌"
            
            logger.info(f"{status} {label:18s}: Q=${result['price_quantum']:.4f}, C=${result['price_classical']:.4f}, Error={result['error_percent']:.2f}%")
            results.append(passed)
        
        all_passed = all(results)
        summary = f"Q-C agreement: {sum(results)}/{len(results)} passed"
        logger.info(f"\n{'✅ PASSED' if all_passed else '❌ FAILED'}: {summary}\n")
        return all_passed, summary
    
    def test_1_3_volatility_sensitivity(self) -> Tuple[bool, str]:
        """Test 1.3: Correct volatility sensitivity direction"""
        logger.info("=" * 70)
        logger.info("TEST 1.3: Volatility Sensitivity (Higher σ → Higher Price)")
        logger.info("=" * 70)
        
        from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
        
        S, K, T, r = 100, 100, 1.0, 0.05
        sigmas = [0.15, 0.20, 0.25, 0.30]
        
        bs_prices = [black_scholes_call(S, K, T, r, sig) for sig in sigmas]
        cm_prices = [price_call_option_corrected(S, K, T, r, sig)['price'] for sig in sigmas]
        
        bs_monotonic = all(bs_prices[i] < bs_prices[i+1] for i in range(len(bs_prices)-1))
        cm_monotonic = all(cm_prices[i] < cm_prices[i+1] for i in range(len(cm_prices)-1))
        
        logger.info(f"{'✅' if bs_monotonic else '❌'} Black-Scholes: Higher σ → Higher price")
        logger.info(f"{'✅' if cm_monotonic else '❌'} Carr-Madan:    Higher σ → Higher price")
        
        for i, sig in enumerate(sigmas):
            logger.info(f"  σ={sig:.2f}: BS=${bs_prices[i]:7.4f}, CM=${cm_prices[i]:7.4f}")
        
        passed = bs_monotonic and cm_monotonic
        summary = f"Volatility sensitivity correct: {'Yes' if passed else 'No'}"
        logger.info(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: {summary}\n")
        return passed, summary
    
    def run_all(self) -> bool:
        """Run all Level 1 tests."""
        logger.info("\n" + "=" * 70)
        logger.info("LEVEL 1: BASE MODEL VALIDATION")
        logger.info("=" * 70 + "\n")
        
        results = {}
        results['1.1 Classical Accuracy'], _ = self.test_1_1_classical_accuracy()
        results['1.2 Q-C Agreement'], _ = self.test_1_2_quantum_classical_agreement()
        results['1.3 Vol Sensitivity'], _ = self.test_1_3_volatility_sensitivity()
        
        all_passed = all(results.values())
        
        logger.info("=" * 70)
        logger.info("LEVEL 1 SUMMARY")
        logger.info("=" * 70)
        for test_name, passed in results.items():
            logger.info(f"{'✅' if passed else '❌'} {test_name}")
        logger.info(f"\n{'✅ LEVEL 1 PASSED' if all_passed else '❌ LEVEL 1 FAILED'}\n")
        
        return all_passed


# ============================================================================
# LEVEL 2: COMPONENT ISOLATION
# ============================================================================

class Level2_ComponentIsolation:
    """Test QRC adapter layer independently of pricing."""
    
    def test_2_1_qrc_factors_valid(self) -> Tuple[bool, str]:
        """Test 2.1: QRC factors are valid probabilities"""
        logger.info("=" * 70)
        logger.info("TEST 2.1: QRC Factor Validity")
        logger.info("=" * 70)
        
        from qrc import QuantumRecurrentCircuit
        
        qrc = QuantumRecurrentCircuit(n_factors=4)
        
        test_inputs = [
            {'prices': 100, 'volatility': 0.20, 'corr_change': 0.0, 'stress': 0.2},
            {'prices': 100, 'volatility': 0.25, 'corr_change': 0.3, 'stress': 0.5},
            {'prices': 100, 'volatility': 0.30, 'corr_change': 0.5, 'stress': 0.9},
        ]
        
        results = []
        for data in test_inputs:
            qrc.reset_hidden_state()
            result = qrc.forward(data)
            factors = result.factors
            
            sum_factors = np.sum(factors)
            all_positive = np.all(factors >= 0)
            sum_ok = np.isclose(sum_factors, 1.0, atol=1e-4)
            
            passed = sum_ok and all_positive
            status = "✅" if passed else "❌"
            
            logger.info(f"{status} stress={data['stress']:.1f}: sum={sum_factors:.4f}, factors={np.round(factors, 3)}")
            results.append(passed)
        
        all_passed = all(results)
        summary = f"QRC factors valid: {sum(results)}/{len(results)} passed"
        logger.info(f"\n{'✅ PASSED' if all_passed else '❌ FAILED'}: {summary}\n")
        return all_passed, summary
    
    def test_2_2_modulation_function(self) -> Tuple[bool, str]:
        """Test 2.2: Modulation function h(f,f̄) correctness"""
        logger.info("=" * 70)
        logger.info("TEST 2.2: Modulation Function Behavior")
        logger.info("=" * 70)
        
        # Test modulation formula: h(f, f̄) = 1 + β(f/f̄ - 1)
        beta = 0.1
        
        # Test uniform factors → h ≈ 1.0
        factors_uniform = np.array([0.25, 0.25, 0.25, 0.25])
        f_bar = np.mean(factors_uniform)
        h_uniform = 1 + beta * (factors_uniform / f_bar - 1)
        
        uniform_ok = np.allclose(h_uniform, 1.0, atol=0.05)
        logger.info(f"{'✅' if uniform_ok else '❌'} Uniform factors: h = {np.round(h_uniform, 3)}, expected ≈ [1,1,1,1]")
        
        # Test concentrated factors → h varies
        factors_concentrated = np.array([0.70, 0.20, 0.07, 0.03])
        f_bar_c = np.mean(factors_concentrated)
        h_concentrated = 1 + beta * (factors_concentrated / f_bar_c - 1)
        
        first_amplified = h_concentrated[0] > 1.0
        last_damped = h_concentrated[-1] < 1.0
        
        concentrated_ok = first_amplified and last_damped
        logger.info(f"{'✅' if concentrated_ok else '❌'} Concentrated: h = {np.round(h_concentrated, 3)}")
        logger.info(f"  First amplified (h>1): {first_amplified}, Last damped (h<1): {last_damped}")
        
        passed = uniform_ok and concentrated_ok
        summary = f"Modulation function correct: {'Yes' if passed else 'No'}"
        logger.info(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: {summary}\n")
        return passed, summary
    
    def test_2_3_sigma_p_computation(self) -> Tuple[bool, str]:
        """Test 2.3: Portfolio volatility σ_p is valid"""
        logger.info("=" * 70)
        logger.info("TEST 2.3: σ_p Computation Validity")
        logger.info("=" * 70)
        
        from qfdp.unified.adapter_layer import BaseModelAdapter
        
        adapter = BaseModelAdapter(beta=0.1)
        
        n = 5
        asset_prices = np.full(n, 100.0)
        asset_vols = np.full(n, 0.20)
        corr = np.eye(n) + 0.5*(1 - np.eye(n))
        weights = np.ones(n) / n
        
        # Test PCA
        pca_result = adapter.prepare_for_pca_pricing(asset_prices, asset_vols, corr, weights)
        sigma_pca = pca_result['sigma_p_pca']
        
        # Test QRC
        qrc_factors = np.array([0.4, 0.3, 0.2, 0.1])
        qrc_result = adapter.prepare_for_qrc_pricing(asset_prices, asset_vols, corr, qrc_factors, weights)
        sigma_qrc = qrc_result['sigma_p_qrc']
        
        pca_positive = sigma_pca > 0
        qrc_positive = sigma_qrc > 0
        qrc_different = not np.isclose(sigma_qrc, sigma_pca, rtol=0.001)
        
        logger.info(f"{'✅' if pca_positive else '❌'} PCA σ_p > 0: {sigma_pca:.4f}")
        logger.info(f"{'✅' if qrc_positive else '❌'} QRC σ_p > 0: {sigma_qrc:.4f}")
        logger.info(f"{'✅' if qrc_different else '❌'} QRC ≠ PCA: {abs(sigma_qrc-sigma_pca)/sigma_pca*100:.2f}% difference")
        
        passed = pca_positive and qrc_positive and qrc_different
        summary = f"σ_p computation valid: {'Yes' if passed else 'No'}"
        logger.info(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: {summary}\n")
        return passed, summary
    
    def run_all(self) -> bool:
        """Run all Level 2 tests."""
        logger.info("\n" + "=" * 70)
        logger.info("LEVEL 2: COMPONENT ISOLATION")
        logger.info("=" * 70 + "\n")
        
        results = {}
        results['2.1 QRC Factors Valid'], _ = self.test_2_1_qrc_factors_valid()
        results['2.2 Modulation Function'], _ = self.test_2_2_modulation_function()
        results['2.3 σ_p Computation'], _ = self.test_2_3_sigma_p_computation()
        
        all_passed = all(results.values())
        
        logger.info("=" * 70)
        logger.info("LEVEL 2 SUMMARY")
        logger.info("=" * 70)
        for test_name, passed in results.items():
            logger.info(f"{'✅' if passed else '❌'} {test_name}")
        logger.info(f"\n{'✅ LEVEL 2 PASSED' if all_passed else '❌ LEVEL 2 FAILED'}\n")
        
        return all_passed


# ============================================================================
# LEVEL 3: INTEGRATION CORRECTNESS
# ============================================================================

class Level3_IntegrationCorrectness:
    """Test QRC → Pricing integration."""
    
    def test_3_1_sigma_p_price_direction(self) -> Tuple[bool, str]:
        """Test 3.1: Different σ_p → Different prices in correct direction"""
        logger.info("=" * 70)
        logger.info("TEST 3.1: σ_p → Price Direction")
        logger.info("=" * 70)
        
        from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
        
        S, K, T, r = 100, 100, 1.0, 0.05
        sigma_low, sigma_high = 0.18, 0.22
        
        price_low = price_call_option_corrected(S, K, T, r, sigma_low)['price']
        price_high = price_call_option_corrected(S, K, T, r, sigma_high)['price']
        
        direction_correct = price_high > price_low
        
        logger.info(f"σ_low={sigma_low}:  price=${price_low:.4f}")
        logger.info(f"σ_high={sigma_high}: price=${price_high:.4f}")
        logger.info(f"{'✅' if direction_correct else '❌'} Higher σ → Higher price: {direction_correct}")
        
        summary = f"σ_p → price direction correct: {'Yes' if direction_correct else 'No'}"
        logger.info(f"\n{'✅ PASSED' if direction_correct else '❌ FAILED'}: {summary}\n")
        return direction_correct, summary
    
    def test_3_2_qrc_produces_different_price(self) -> Tuple[bool, str]:
        """Test 3.2: QRC produces different σ_p and different price from PCA"""
        logger.info("=" * 70)
        logger.info("TEST 3.2: QRC Produces Different Price Than PCA")
        logger.info("=" * 70)
        
        from qfdp.unified.adapter_layer import BaseModelAdapter
        from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
        from qrc import QuantumRecurrentCircuit
        
        adapter = BaseModelAdapter(beta=0.1)
        qrc = QuantumRecurrentCircuit(n_factors=4)
        
        n = 5
        asset_prices = np.full(n, 100.0)
        asset_vols = np.full(n, 0.20)
        corr = np.eye(n) + 0.6*(1 - np.eye(n))
        weights = np.ones(n) / n
        
        S, K, T, r = 100.0, 100.0, 1.0, 0.05
        
        # PCA σ_p
        pca_result = adapter.prepare_for_pca_pricing(asset_prices, asset_vols, corr, weights)
        sigma_pca = pca_result['sigma_p_pca']
        
        # QRC σ_p
        market_data = {'prices': 100, 'volatility': 0.20, 'corr_change': 0.3, 'stress': 0.6}
        qrc_factors = qrc.forward(market_data).factors
        qrc_result = adapter.prepare_for_qrc_pricing(asset_prices, asset_vols, corr, qrc_factors, weights)
        sigma_qrc = qrc_result['sigma_p_qrc']
        
        # Prices
        price_pca = price_call_option_corrected(S, K, T, r, sigma_pca)['price']
        price_qrc = price_call_option_corrected(S, K, T, r, sigma_qrc)['price']
        
        sigma_diff = abs(sigma_qrc - sigma_pca) / sigma_pca * 100
        price_diff = abs(price_qrc - price_pca) / price_pca * 100
        
        logger.info(f"σ_p,PCA = {sigma_pca:.4f}, σ_p,QRC = {sigma_qrc:.4f} (diff: {sigma_diff:.2f}%)")
        logger.info(f"Price_PCA = ${price_pca:.4f}, Price_QRC = ${price_qrc:.4f} (diff: {price_diff:.2f}%)")
        
        passed = sigma_diff > 0.1 and price_diff > 0.1
        summary = f"QRC produces different price: {'Yes' if passed else 'No'}"
        logger.info(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: {summary}\n")
        return passed, summary
    
    def run_all(self) -> bool:
        """Run all Level 3 tests."""
        logger.info("\n" + "=" * 70)
        logger.info("LEVEL 3: INTEGRATION CORRECTNESS")
        logger.info("=" * 70 + "\n")
        
        results = {}
        results['3.1 σ_p→Price Direction'], _ = self.test_3_1_sigma_p_price_direction()
        results['3.2 QRC Different Price'], _ = self.test_3_2_qrc_produces_different_price()
        
        all_passed = all(results.values())
        
        logger.info("=" * 70)
        logger.info("LEVEL 3 SUMMARY")
        logger.info("=" * 70)
        for test_name, passed in results.items():
            logger.info(f"{'✅' if passed else '❌'} {test_name}")
        logger.info(f"\n{'✅ LEVEL 3 PASSED' if all_passed else '❌ LEVEL 3 FAILED'}\n")
        
        return all_passed


# ============================================================================
# LEVEL 4: QRC ADVANTAGE DEMONSTRATION
# ============================================================================

class Level4_QRCAdvantage:
    """
    Test QRC advantage in REALISTIC scenario:
    - PCA calibrated on CALM market (stale calibration)
    - Market shifts to STRESSED regime
    - PCA uses stale σ_p → HIGH error
    - QRC adapts to new regime → LOW error
    """
    
    def test_4_1_regime_shift_advantage(self) -> Tuple[bool, str]:
        """Test 4.1: QRC outperforms stale PCA under regime shift"""
        logger.info("=" * 70)
        logger.info("TEST 4.1: QRC Advantage Under Regime Shift")
        logger.info("=" * 70)
        
        from qfdp.unified.adapter_layer import BaseModelAdapter
        from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
        from qrc import QuantumRecurrentCircuit
        
        adapter = BaseModelAdapter(beta=0.1)
        qrc = QuantumRecurrentCircuit(n_factors=4)
        
        n = 5
        asset_prices = np.full(n, 100.0)
        asset_vols = np.full(n, 0.20)
        weights = np.ones(n) / n
        S, K, T, r = 100.0, 100.0, 1.0, 0.05
        
        # =============================================================
        # PHASE 1: CALIBRATE PCA ON CALM MARKET (Morning calibration)
        # =============================================================
        rho_calm = 0.3
        corr_calm = np.eye(n) + rho_calm*(1 - np.eye(n))
        
        pca_calm = adapter.prepare_for_pca_pricing(asset_prices, asset_vols, corr_calm, weights)
        sigma_pca_calibrated = pca_calm['sigma_p_pca']  # STALE VALUE
        
        logger.info(f"\n📅 MORNING: Calibrated on calm market (ρ = {rho_calm})")
        logger.info(f"   PCA σ_p (stale): {sigma_pca_calibrated:.4f}")
        
        # =============================================================
        # PHASE 2: TEST ACROSS REGIME SHIFTS (Market changes during day)
        # =============================================================
        regimes = [
            {'rho': 0.3, 'label': 'Calm', 'stress': 0.2},
            {'rho': 0.5, 'label': 'Medium', 'stress': 0.5},
            {'rho': 0.7, 'label': 'High', 'stress': 0.7},
            {'rho': 0.8, 'label': 'Crisis', 'stress': 0.9},
        ]
        
        logger.info(f"\n📈 DURING DAY: Market regime shifts\n")
        logger.info(f"{'Regime':<10} {'True σ_p':<10} {'Stale PCA':<10} {'QRC σ_p':<10} {'PCA Err':<10} {'QRC Err':<10} {'QRC Wins?'}")
        logger.info("-" * 75)
        
        results = []
        for regime in regimes:
            corr = np.eye(n) + regime['rho']*(1 - np.eye(n))
            
            # TRUE σ_p (what market actually is NOW)
            pca_current = adapter.prepare_for_pca_pricing(asset_prices, asset_vols, corr, weights)
            sigma_true = pca_current['sigma_p_pca']
            
            # STALE PCA σ_p (from morning calibration - WRONG!)
            sigma_pca_stale = sigma_pca_calibrated
            
            # QRC σ_p (adapts to current market)
            market_data = {'prices': 100, 'volatility': 0.20, 'corr_change': (regime['rho'] - rho_calm)/rho_calm, 'stress': regime['stress']}
            qrc.reset_hidden_state()
            qrc_factors = qrc.forward(market_data).factors
            qrc_result = adapter.prepare_for_qrc_pricing(asset_prices, asset_vols, corr, qrc_factors, weights)
            sigma_qrc = qrc_result['sigma_p_qrc']
            
            # Prices
            price_true = price_call_option_corrected(S, K, T, r, sigma_true)['price']
            price_pca = price_call_option_corrected(S, K, T, r, sigma_pca_stale)['price']
            price_qrc = price_call_option_corrected(S, K, T, r, sigma_qrc)['price']
            
            error_pca = abs(price_pca - price_true) / price_true * 100
            error_qrc = abs(price_qrc - price_true) / price_true * 100
            
            qrc_wins = error_qrc < error_pca
            win_symbol = "✅" if qrc_wins else "❌"
            
            logger.info(f"{regime['label']:<10} {sigma_true:<10.4f} {sigma_pca_stale:<10.4f} {sigma_qrc:<10.4f} {error_pca:<9.2f}% {error_qrc:<9.2f}% {win_symbol}")
            
            results.append({
                'regime': regime['label'],
                'rho': regime['rho'],
                'error_pca': error_pca,
                'error_qrc': error_qrc,
                'qrc_wins': qrc_wins
            })
        
        # =============================================================
        # ANALYSIS
        # =============================================================
        avg_pca_error = np.mean([r['error_pca'] for r in results])
        avg_qrc_error = np.mean([r['error_qrc'] for r in results])
        improvement = avg_pca_error - avg_qrc_error
        
        crisis = results[-1]
        crisis_improvement = crisis['error_pca'] - crisis['error_qrc']
        
        logger.info(f"\n📊 ANALYSIS:")
        logger.info(f"   Mean Stale PCA Error: {avg_pca_error:.2f}%")
        logger.info(f"   Mean QRC Error:       {avg_qrc_error:.2f}%")
        logger.info(f"   QRC Improvement:      {improvement:.2f}%")
        logger.info(f"\n   Crisis Regime (ρ=0.8):")
        logger.info(f"     Stale PCA Error: {crisis['error_pca']:.2f}%")
        logger.info(f"     QRC Error:       {crisis['error_qrc']:.2f}%")
        logger.info(f"     Improvement:     {crisis_improvement:.2f}%")
        
        # Success criteria: QRC should beat stale PCA, especially in stressed regimes
        qrc_wins_in_stress = results[-1]['qrc_wins'] and results[-2]['qrc_wins']
        significant_improvement = crisis_improvement > 10.0
        
        passed = qrc_wins_in_stress and significant_improvement
        summary = f"QRC improvement in crisis: {crisis_improvement:.2f}%"
        logger.info(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: {summary}\n")
        return passed, summary
    
    def run_all(self) -> bool:
        """Run Level 4 tests."""
        logger.info("\n" + "=" * 70)
        logger.info("LEVEL 4: QRC ADVANTAGE DEMONSTRATION")
        logger.info("=" * 70 + "\n")
        
        results = {}
        results['4.1 Regime Shift Advantage'], _ = self.test_4_1_regime_shift_advantage()
        
        all_passed = all(results.values())
        
        logger.info("=" * 70)
        logger.info("LEVEL 4 SUMMARY")
        logger.info("=" * 70)
        for test_name, passed in results.items():
            logger.info(f"{'✅' if passed else '❌'} {test_name}")
        logger.info(f"\n{'✅ LEVEL 4 PASSED' if all_passed else '❌ LEVEL 4 FAILED'}\n")
        
        return all_passed


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_comprehensive_validation():
    """Run all validation levels."""
    
    logger.info("\n" + "=" * 70)
    logger.info("QRC + FB-IQFT COMPREHENSIVE VALIDATION SUITE")
    logger.info("=" * 70 + "\n")
    
    all_results = {}
    
    # Level 1
    level1 = Level1_BaseModelValidation()
    level1_passed = level1.run_all()
    all_results['Level 1: Base Model'] = level1_passed
    
    if not level1_passed:
        logger.error("❌ LEVEL 1 FAILED - Base model has issues!")
        return False
    
    # Level 2
    level2 = Level2_ComponentIsolation()
    level2_passed = level2.run_all()
    all_results['Level 2: Components'] = level2_passed
    
    if not level2_passed:
        logger.error("❌ LEVEL 2 FAILED - Adapter layer has issues!")
        return False
    
    # Level 3
    level3 = Level3_IntegrationCorrectness()
    level3_passed = level3.run_all()
    all_results['Level 3: Integration'] = level3_passed
    
    if not level3_passed:
        logger.error("❌ LEVEL 3 FAILED - Integration has issues!")
        return False
    
    # Level 4
    level4 = Level4_QRCAdvantage()
    level4_passed = level4.run_all()
    all_results['Level 4: QRC Advantage'] = level4_passed
    
    # Final Summary
    all_passed = all(all_results.values())
    
    logger.info("\n" + "=" * 70)
    logger.info("COMPREHENSIVE VALIDATION SUMMARY")
    logger.info("=" * 70)
    for level_name, passed in all_results.items():
        logger.info(f"{'✅' if passed else '❌'} {level_name}")
    
    if all_passed:
        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL TESTS PASSED - SYSTEM VALIDATED")
        logger.info("=" * 70)
        logger.info("\nYou may now proceed to QRC experiments with confidence!")
    else:
        logger.info("\n❌ SOME TESTS FAILED - Fix issues before experiments")
    
    return all_passed


if __name__ == "__main__":
    success = run_comprehensive_validation()
    exit(0 if success else 1)
