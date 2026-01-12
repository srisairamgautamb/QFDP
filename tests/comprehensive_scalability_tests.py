"""
Comprehensive Scalability Test Suite
====================================

Tests the full QRC+QTC+FB-IQFT pipeline across:
- Portfolio sizes: N = 2, 5, 10, 50
- Market regimes: Calm, Medium, High, Crisis
- All base model tests + QRC/QTC integration tests

NO HARDCODING - everything is parameterized.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
import sys
import time

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.fb_iqft_pricing import FBIQFTPricing
from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
from qfdp.integrated.corrected_qtc_integrated_pricer import CorrectedQTCIntegratedPricer
from qfdp.unified.enhanced_factor_constructor import EnhancedFactorConstructor
from scipy.stats import norm

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================================
# PARAMETERIZED TEST CONFIGURATION
# ============================================================================

# Portfolio sizes to test
PORTFOLIO_SIZES = [2, 5, 10, 50]

# Market regimes (parameterized)
MARKET_REGIMES = {
    'calm': {'rho': 0.3, 'vol_multiplier': 1.0, 'trend': 'up'},
    'medium': {'rho': 0.5, 'vol_multiplier': 1.2, 'trend': 'flat'},
    'high': {'rho': 0.7, 'vol_multiplier': 1.5, 'trend': 'volatile'},
    'crisis': {'rho': 0.85, 'vol_multiplier': 2.0, 'trend': 'down'},
}

# Test parameters
TEST_PARAMS = {
    'base_volatility': 0.20,
    'risk_free_rate': 0.05,
    'maturity': 1.0,
    'spot_price': 100.0,
    'strike_moneyness': [0.9, 1.0, 1.1],  # ATM, ITM, OTM
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_correlation_matrix(n: int, rho: float) -> np.ndarray:
    """Generate N×N correlation matrix with uniform off-diagonal correlation."""
    return np.eye(n) * (1 - rho) + rho


def generate_price_history(n_points: int = 6, trend: str = 'up', base_price: float = 100.0) -> np.ndarray:
    """Generate price history based on trend type."""
    if trend == 'up':
        return base_price + np.linspace(-2, 3, n_points)
    elif trend == 'down':
        return base_price + np.linspace(3, -5, n_points)
    elif trend == 'flat':
        return base_price + np.array([0, 0.1, -0.1, 0.2, 0, 0.1])[:n_points]
    elif trend == 'volatile':
        return base_price + np.array([0, 3, -2, 4, -3, 2])[:n_points]
    else:
        return np.full(n_points, base_price)


def compute_bs_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ============================================================================
# TEST CLASSES
# ============================================================================

class TestResults:
    """Collect and format test results."""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def add(self, test_name: str, passed: bool, details: str = ""):
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results.append({
            'name': test_name,
            'passed': passed,
            'details': details,
            'status': status
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def summary(self) -> str:
        total = self.passed + self.failed
        return f"{self.passed}/{total} tests passed"


class ScalabilityTestSuite:
    """
    Comprehensive test suite for N = 2, 5, 10, 50 assets.
    """
    
    def __init__(self, portfolio_sizes: List[int] = None):
        self.portfolio_sizes = portfolio_sizes or PORTFOLIO_SIZES
        self.results = TestResults()
        
        # Initialize pricers
        self.fb_iqft = FBIQFTPricing(M=16, alpha=1.0)
        self.qrc_qtc_pricer = CorrectedQTCIntegratedPricer(
            self.fb_iqft, 
            qrc_beta=0.1, 
            qtc_gamma=0.05
        )
        self.factor_constructor = EnhancedFactorConstructor(n_factors=4)
    
    def run_all(self) -> Tuple[bool, TestResults]:
        """Run all tests across all portfolio sizes."""
        print("=" * 80)
        print("COMPREHENSIVE SCALABILITY TEST SUITE")
        print("=" * 80)
        print(f"Testing N = {self.portfolio_sizes}")
        print("=" * 80 + "\n")
        
        for n in self.portfolio_sizes:
            self._run_tests_for_n(n)
        
        # Final summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(self.results.summary())
        
        all_passed = self.results.failed == 0
        return all_passed, self.results
    
    def _run_tests_for_n(self, n: int):
        """Run all test categories for a given portfolio size N."""
        print(f"\n{'='*60}")
        print(f"TESTING N = {n} ASSETS")
        print(f"{'='*60}\n")
        
        # Category 1: Base Model Tests
        self._test_base_model(n)
        
        # Category 2: QRC Tests
        self._test_qrc(n)
        
        # Category 3: QTC Tests
        self._test_qtc(n)
        
        # Category 4: Enhanced Factor Constructor Tests
        self._test_enhanced_factors(n)
        
        # Category 5: Full Pipeline Integration Tests
        self._test_full_pipeline(n)
        
        # Category 6: Regime Comparison Tests
        self._test_regime_comparison(n)
        
        # Category 7: Stress Tests
        self._test_stress_conditions(n)
    
    # -----------------------------------------------------------------------
    # CATEGORY 1: BASE MODEL TESTS
    # -----------------------------------------------------------------------
    
    def _test_base_model(self, n: int):
        """Test base model correctness."""
        print(f"  Category 1: Base Model Tests (N={n})")
        
        # Test 1.1: Black-Scholes accuracy
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        result = price_call_option_corrected(S, K, T, r, sigma)
        bs_price = compute_bs_price(S, K, T, r, sigma)
        error = abs(result['price'] - bs_price) / bs_price * 100
        passed = error < 0.1
        self.results.add(
            f"[N={n}] 1.1 BS Accuracy",
            passed,
            f"Error: {error:.4f}%"
        )
        print(f"    {'✅' if passed else '❌'} 1.1 BS Accuracy: {error:.4f}%")
        
        # Test 1.2: Price monotonicity in volatility
        prices = []
        for vol in [0.1, 0.15, 0.2, 0.25, 0.3]:
            prices.append(price_call_option_corrected(S, K, T, r, vol)['price'])
        monotonic = all(prices[i] <= prices[i+1] for i in range(len(prices)-1))
        self.results.add(f"[N={n}] 1.2 Vol Monotonicity", monotonic)
        print(f"    {'✅' if monotonic else '❌'} 1.2 Volatility Monotonicity")
        
        # Test 1.3: Portfolio volatility computation
        asset_vols = np.full(n, TEST_PARAMS['base_volatility'])
        weights = np.ones(n) / n
        corr = generate_correlation_matrix(n, 0.5)
        vol_matrix = np.diag(asset_vols)
        cov = vol_matrix @ corr @ vol_matrix
        sigma_p = np.sqrt(weights.T @ cov @ weights)
        
        # Check bounds
        min_bound = np.min(asset_vols) * (1 - 0.5) ** 0.5  # Lower bound with correlation
        max_bound = np.max(asset_vols)  # Upper bound
        bounds_ok = sigma_p > 0 and sigma_p <= max_bound
        self.results.add(f"[N={n}] 1.3 σ_p Bounds", bounds_ok, f"σ_p={sigma_p:.4f}")
        print(f"    {'✅' if bounds_ok else '❌'} 1.3 Portfolio Vol Bounds: σ_p={sigma_p:.4f}")
    
    # -----------------------------------------------------------------------
    # CATEGORY 2: QRC TESTS
    # -----------------------------------------------------------------------
    
    def _test_qrc(self, n: int):
        """Test QRC module."""
        print(f"  Category 2: QRC Tests (N={n})")
        
        from qrc import QuantumRecurrentCircuit
        
        qrc = QuantumRecurrentCircuit(n_factors=4)
        
        # Test 2.1: Factor generation
        qrc.reset_hidden_state()
        qrc_input = {
            'prices': 100.0,
            'volatility': 0.20,
            'corr_change': 0.1,
            'stress': 0.3
        }
        result = qrc.forward(qrc_input)
        factors = result.factors
        
        # Factors should sum to 1 and be positive
        sum_check = np.isclose(np.sum(factors), 1.0, rtol=0.01)
        positive_check = np.all(factors >= 0)
        passed = sum_check and positive_check
        self.results.add(
            f"[N={n}] 2.1 QRC Factors Valid",
            passed,
            f"sum={np.sum(factors):.4f}, min={np.min(factors):.4f}"
        )
        print(f"    {'✅' if passed else '❌'} 2.1 QRC Factors: sum={np.sum(factors):.4f}")
        
        # Test 2.2: QRC responds to stress
        qrc.reset_hidden_state()
        low_stress = qrc.forward({'prices': 100, 'volatility': 0.2, 'corr_change': 0, 'stress': 0.1})
        qrc.reset_hidden_state()
        high_stress = qrc.forward({'prices': 100, 'volatility': 0.2, 'corr_change': 0.5, 'stress': 0.9})
        
        # Factors should differ
        diff = np.linalg.norm(low_stress.factors - high_stress.factors)
        responds = diff > 0.01
        self.results.add(f"[N={n}] 2.2 QRC Stress Response", responds, f"diff={diff:.4f}")
        print(f"    {'✅' if responds else '❌'} 2.2 QRC Stress Response: diff={diff:.4f}")
    
    # -----------------------------------------------------------------------
    # CATEGORY 3: QTC TESTS
    # -----------------------------------------------------------------------
    
    def _test_qtc(self, n: int):
        """Test QTC module."""
        print(f"  Category 3: QTC Tests (N={n})")
        
        from qfdp.qtc.quantum_temporal_convolution import QuantumTemporalConvolution
        
        qtc = QuantumTemporalConvolution(kernel_size=3, n_kernels=4)
        
        # Test 3.1: Pattern extraction
        price_history = generate_price_history(6, 'up', 100)
        patterns = qtc.forward_with_pooling(price_history)
        
        # Patterns should be valid probabilities
        sum_check = np.isclose(np.sum(patterns), 1.0, rtol=0.01)
        positive_check = np.all(patterns >= 0)
        passed = sum_check and positive_check
        self.results.add(
            f"[N={n}] 3.1 QTC Patterns Valid",
            passed,
            f"sum={np.sum(patterns):.4f}"
        )
        print(f"    {'✅' if passed else '❌'} 3.1 QTC Patterns: sum={np.sum(patterns):.4f}")
        
        # Test 3.2: Different trends give different patterns
        patterns_up = qtc.forward_with_pooling(generate_price_history(6, 'up'))
        patterns_down = qtc.forward_with_pooling(generate_price_history(6, 'down'))
        patterns_volatile = qtc.forward_with_pooling(generate_price_history(6, 'volatile'))
        
        diff_up_down = np.linalg.norm(patterns_up - patterns_down)
        diff_up_vol = np.linalg.norm(patterns_up - patterns_volatile)
        
        different = diff_up_down > 0.05 or diff_up_vol > 0.05
        self.results.add(
            f"[N={n}] 3.2 QTC Pattern Differentiation",
            different,
            f"diff_up_down={diff_up_down:.4f}, diff_up_vol={diff_up_vol:.4f}"
        )
        print(f"    {'✅' if different else '❌'} 3.2 Pattern Differentiation: diff={diff_up_down:.4f}")
    
    # -----------------------------------------------------------------------
    # CATEGORY 4: ENHANCED FACTOR CONSTRUCTOR TESTS
    # -----------------------------------------------------------------------
    
    def _test_enhanced_factors(self, n: int):
        """Test Enhanced Factor Constructor."""
        print(f"  Category 4: Enhanced Factor Tests (N={n})")
        
        # Setup
        qrc_factors = np.array([0.3, 0.3, 0.2, 0.2])
        qtc_patterns = np.array([0.25, 0.25, 0.25, 0.25])
        corr = generate_correlation_matrix(n, 0.5)
        asset_vols = np.full(n, TEST_PARAMS['base_volatility'])
        weights = np.ones(n) / n
        
        # Test 4.1: Factor construction shapes
        L_enhanced, D_enhanced, mu_enhanced = self.factor_constructor.construct_enhanced_factors(
            qrc_factors, qtc_patterns, corr, asset_vols
        )
        
        k = min(4, n)
        shape_ok = L_enhanced.shape == (n, k) and D_enhanced.shape == (k, k)
        self.results.add(
            f"[N={n}] 4.1 Factor Shapes",
            shape_ok,
            f"L={L_enhanced.shape}, D={D_enhanced.shape}"
        )
        print(f"    {'✅' if shape_ok else '❌'} 4.1 Factor Shapes: L={L_enhanced.shape}, D={D_enhanced.shape}")
        
        # Test 4.2: Enhanced σ_p computation
        sigma_p = self.factor_constructor.compute_portfolio_volatility(
            L_enhanced, D_enhanced, weights, asset_vols,
            base_correlation=corr, qrc_factors=qrc_factors, qtc_patterns=qtc_patterns
        )
        
        valid = sigma_p > 0 and sigma_p < 1  # Reasonable bounds
        self.results.add(f"[N={n}] 4.2 Enhanced σ_p Valid", valid, f"σ_p={sigma_p:.4f}")
        print(f"    {'✅' if valid else '❌'} 4.2 Enhanced σ_p: {sigma_p:.4f}")
    
    # -----------------------------------------------------------------------
    # CATEGORY 5: FULL PIPELINE TESTS
    # -----------------------------------------------------------------------
    
    def _test_full_pipeline(self, n: int):
        """Test full QRC+QTC+FB-IQFT pipeline."""
        print(f"  Category 5: Full Pipeline Tests (N={n})")
        
        # Setup market data
        asset_prices = np.full(n, TEST_PARAMS['spot_price'])
        asset_vols = np.full(n, TEST_PARAMS['base_volatility'])
        weights = np.ones(n) / n
        corr = generate_correlation_matrix(n, 0.5)
        price_history = generate_price_history(6, 'up')
        
        market_data = {
            'spot_prices': asset_prices,
            'volatilities': asset_vols,
            'correlation_matrix': corr,
            'weights': weights,
            'maturity': TEST_PARAMS['maturity'],
            'risk_free_rate': TEST_PARAMS['risk_free_rate']
        }
        
        # Test 5.1: Pipeline executes without error
        try:
            result = self.qrc_qtc_pricer.price_with_full_quantum_pipeline(
                market_data, price_history, strike=100.0, use_quantum_circuit=True
            )
            executed = True
            price = result['price_quantum']
        except Exception as e:
            executed = False
            price = None
            logger.error(f"Pipeline failed: {e}")
        
        self.results.add(
            f"[N={n}] 5.1 Pipeline Execution",
            executed,
            f"price={price:.4f}" if price else "Failed"
        )
        print(f"    {'✅' if executed else '❌'} 5.1 Pipeline Execution: price={price if price else 'N/A'}")
        
        if executed:
            # Test 5.2: Price is reasonable
            reasonable = 0 < price < 100
            self.results.add(f"[N={n}] 5.2 Price Reasonable", reasonable, f"price={price:.4f}")
            print(f"    {'✅' if reasonable else '❌'} 5.2 Price Reasonable: {price:.4f}")
            
            # Test 5.3: Circuit depth exists
            depth = result.get('circuit_depth', 0)
            has_depth = depth > 0
            self.results.add(f"[N={n}] 5.3 Quantum Circuit Ran", has_depth, f"depth={depth}")
            print(f"    {'✅' if has_depth else '❌'} 5.3 Quantum Circuit: depth={depth}")
    
    # -----------------------------------------------------------------------
    # CATEGORY 6: REGIME COMPARISON TESTS
    # -----------------------------------------------------------------------
    
    def _test_regime_comparison(self, n: int):
        """Test QRC+QTC improvement over stale PCA across regimes."""
        print(f"  Category 6: Regime Comparison Tests (N={n})")
        
        asset_prices = np.full(n, TEST_PARAMS['spot_price'])
        asset_vols = np.full(n, TEST_PARAMS['base_volatility'])
        weights = np.ones(n) / n
        
        # Calibrate on calm market (stale PCA)
        corr_calm = generate_correlation_matrix(n, MARKET_REGIMES['calm']['rho'])
        vol_matrix = np.diag(asset_vols)
        cov_calm = vol_matrix @ corr_calm @ vol_matrix
        sigma_pca_stale = float(np.sqrt(weights.T @ cov_calm @ weights))
        
        improvements = []
        
        for regime_name, regime_params in MARKET_REGIMES.items():
            corr = generate_correlation_matrix(n, regime_params['rho'])
            price_history = generate_price_history(6, regime_params['trend'])
            
            # True σ_p (what the market actually is now)
            cov_true = vol_matrix @ corr @ vol_matrix
            sigma_true = float(np.sqrt(weights.T @ cov_true @ weights))
            price_true = price_call_option_corrected(100, 100, 1.0, 0.05, sigma_true)['price']
            
            # Stale PCA error (using morning calibration)
            price_stale = price_call_option_corrected(100, 100, 1.0, 0.05, sigma_pca_stale)['price']
            error_stale = abs(price_stale - price_true) / price_true * 100
            
            # QRC+QTC (with CURRENT market conditions - should adapt!)
            market_data = {
                'spot_prices': asset_prices,
                'volatilities': asset_vols,  # Same base vols
                'correlation_matrix': corr,  # Current correlation (regime-specific)
                'weights': weights,
                'maturity': 1.0,
                'risk_free_rate': 0.05
            }
            
            result = self.qrc_qtc_pricer.price_with_full_quantum_pipeline(
                market_data, price_history, strike=100.0, use_quantum_circuit=True
            )
            error_qrc_qtc = abs(result['price_quantum'] - price_true) / price_true * 100
            
            improvement = error_stale - error_qrc_qtc
            improvements.append(improvement)
        
        # Test 6.1: Any improvement in at least one stressed regime
        max_improvement = max(improvements[1:])  # Exclude calm
        any_improves = max_improvement > 0
        self.results.add(
            f"[N={n}] 6.1 Any Stressed Improvement",
            any_improves,
            f"max={max_improvement:.2f}%"
        )
        print(f"    {'✅' if any_improves else '❌'} 6.1 Any Stressed Improvement: max={max_improvement:.2f}%")
        
        # Test 6.2: Crisis scenario - should see improvement
        crisis_improvement = improvements[3]  # Crisis is last
        crisis_ok = crisis_improvement > 0  # Any positive improvement
        self.results.add(
            f"[N={n}] 6.2 Crisis Improvement >0%",
            crisis_ok,
            f"improvement={crisis_improvement:.2f}%"
        )
        print(f"    {'✅' if crisis_ok else '❌'} 6.2 Crisis Improvement: {crisis_improvement:.2f}%")
    
    # -----------------------------------------------------------------------
    # CATEGORY 7: STRESS TESTS
    # -----------------------------------------------------------------------
    
    def _test_stress_conditions(self, n: int):
        """Test under extreme conditions."""
        print(f"  Category 7: Stress Tests (N={n})")
        
        # Test 7.1: Extreme correlation (ρ = 0.99)
        corr_extreme = generate_correlation_matrix(n, 0.99)
        asset_vols = np.full(n, 0.20)
        weights = np.ones(n) / n
        vol_matrix = np.diag(asset_vols)
        
        try:
            cov_extreme = vol_matrix @ corr_extreme @ vol_matrix
            sigma_p = np.sqrt(weights.T @ cov_extreme @ weights)
            # Should be close to average vol
            expected = np.mean(asset_vols)
            close = abs(sigma_p - expected) < 0.05
            self.results.add(f"[N={n}] 7.1 Extreme Correlation", close, f"σ_p={sigma_p:.4f}")
            print(f"    {'✅' if close else '❌'} 7.1 Extreme Correlation (ρ=0.99): σ_p={sigma_p:.4f}")
        except:
            self.results.add(f"[N={n}] 7.1 Extreme Correlation", False, "Failed")
            print(f"    ❌ 7.1 Extreme Correlation: Failed")
        
        # Test 7.2: Zero correlation (ρ = 0)
        corr_zero = np.eye(n)
        try:
            cov_zero = vol_matrix @ corr_zero @ vol_matrix
            sigma_p = np.sqrt(weights.T @ cov_zero @ weights)
            # Should be lower than average vol (diversification)
            expected = np.mean(asset_vols) / np.sqrt(n)
            close = abs(sigma_p - expected) < 0.05
            self.results.add(f"[N={n}] 7.2 Zero Correlation", close, f"σ_p={sigma_p:.4f}")
            print(f"    {'✅' if close else '❌'} 7.2 Zero Correlation (ρ=0): σ_p={sigma_p:.4f}")
        except:
            self.results.add(f"[N={n}] 7.2 Zero Correlation", False, "Failed")
            print(f"    ❌ 7.2 Zero Correlation: Failed")
        
        # Test 7.3: Very high volatility
        try:
            asset_vols_high = np.full(n, 0.80)
            vol_matrix_high = np.diag(asset_vols_high)
            corr = generate_correlation_matrix(n, 0.5)
            cov_high = vol_matrix_high @ corr @ vol_matrix_high
            sigma_p_high = np.sqrt(weights.T @ cov_high @ weights)
            
            price = price_call_option_corrected(100, 100, 1.0, 0.05, sigma_p_high)['price']
            valid = 0 < price < 100
            self.results.add(f"[N={n}] 7.3 High Volatility", valid, f"price={price:.4f}")
            print(f"    {'✅' if valid else '❌'} 7.3 High Volatility (σ=0.80): price=${price:.4f}")
        except:
            self.results.add(f"[N={n}] 7.3 High Volatility", False, "Failed")
            print(f"    ❌ 7.3 High Volatility: Failed")
        
        # Test 7.4: Performance timing
        start_time = time.time()
        asset_prices = np.full(n, 100.0)
        market_data = {
            'spot_prices': asset_prices,
            'volatilities': np.full(n, 0.20),
            'correlation_matrix': generate_correlation_matrix(n, 0.5),
            'weights': np.ones(n) / n,
            'maturity': 1.0,
            'risk_free_rate': 0.05
        }
        price_history = generate_price_history(6, 'up')
        
        for _ in range(5):
            self.qrc_qtc_pricer.price_with_full_quantum_pipeline(
                market_data, price_history, strike=100.0, use_quantum_circuit=True
            )
        
        elapsed = (time.time() - start_time) / 5
        fast = elapsed < 5.0  # Less than 5 seconds per pricing
        self.results.add(f"[N={n}] 7.4 Performance", fast, f"time={elapsed:.3f}s")
        print(f"    {'✅' if fast else '❌'} 7.4 Performance: {elapsed:.3f}s per pricing")


# ============================================================================
# MAIN
# ============================================================================

def run_comprehensive_tests():
    """Run all tests."""
    suite = ScalabilityTestSuite(PORTFOLIO_SIZES)
    all_passed, results = suite.run_all()
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {results.failed} test(s) failed")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    run_comprehensive_tests()
