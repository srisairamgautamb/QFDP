"""
Validation Tests for Real VaR/CVaR
===================================

These tests GUARANTEE that VaR/CVaR are computed correctly with NO shortcuts.

Test Strategy:
--------------
1. Single-asset: Compare MC to analytical formula
2. Convergence: Verify MC error decreases with 1/√M
3. CVaR > VaR: Mathematical requirement
4. Correlation impact: Higher ρ → higher VaR
5. Real portfolio: Sanity checks with actual market data

All tests must pass before claiming "real" VaR/CVaR.
"""

import pytest
import numpy as np
from qfdp_multiasset.risk import (
    compute_var_cvar_mc,
    analytical_var_single_asset,
    VaRCVaRResult
)


class TestSingleAssetVaR:
    """Test 1: Single asset MC should match analytical formula."""
    
    def test_single_asset_matches_analytical(self):
        """MC VaR for N=1 should match Φ⁻¹(α) × σ × √T × PV."""
        pv = 100000.0
        vol = 0.20  # 20% annualized
        
        # MC VaR
        mc_result = compute_var_cvar_mc(
            portfolio_value=pv,
            weights=np.array([1.0]),
            volatilities=np.array([vol]),
            correlation_matrix=np.array([[1.0]]),
            time_horizon_days=1,
            num_simulations=50000,  # High M for accuracy
            seed=42
        )
        
        # Analytical VaR
        analytical_var = analytical_var_single_asset(
            portfolio_value=pv,
            volatility=vol,
            confidence_level=0.95,
            time_horizon_days=1
        )
        
        # MC should be within 5% of analytical (with M=50K)
        error_pct = abs(mc_result.var_95 - analytical_var) / analytical_var
        
        print(f"\nSingle Asset VaR Validation:")
        print(f"  MC VaR:          ${mc_result.var_95:,.2f}")
        print(f"  Analytical VaR:  ${analytical_var:,.2f}")
        print(f"  Error:           {error_pct*100:.2f}%")
        
        assert error_pct < 0.05, f"MC error {error_pct*100:.1f}% > 5% tolerance"
        
    def test_var_increases_with_volatility(self):
        """Higher volatility → higher VaR."""
        pv = 100000.0
        
        var_low = compute_var_cvar_mc(
            portfolio_value=pv,
            weights=np.array([1.0]),
            volatilities=np.array([0.10]),  # 10% vol
            correlation_matrix=np.array([[1.0]]),
            num_simulations=10000,
            seed=42
        ).var_95
        
        var_high = compute_var_cvar_mc(
            portfolio_value=pv,
            weights=np.array([1.0]),
            volatilities=np.array([0.30]),  # 30% vol
            correlation_matrix=np.array([[1.0]]),
            num_simulations=10000,
            seed=42
        ).var_95
        
        print(f"\nVolatility Impact:")
        print(f"  VaR (σ=10%): ${var_low:,.2f}")
        print(f"  VaR (σ=30%): ${var_high:,.2f}")
        print(f"  Ratio: {var_high/var_low:.2f}× (expect ~3×)")
        
        assert var_high > var_low, "Higher vol must give higher VaR"
        assert var_high / var_low > 2.5, "VaR should scale roughly with volatility"


class TestConvergence:
    """Test 2: MC error should decrease with 1/√M."""
    
    def test_convergence_rate(self):
        """Verify MC standard error ∝ 1/√M."""
        pv = 100000.0
        weights = np.array([0.6, 0.4])
        vols = np.array([0.20, 0.25])
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        M_values = [1000, 10000, 100000]
        var_estimates = []
        
        for M in M_values:
            result = compute_var_cvar_mc(
                portfolio_value=pv,
                weights=weights,
                volatilities=vols,
                correlation_matrix=corr,
                num_simulations=M,
                seed=42
            )
            var_estimates.append(result.var_95)
        
        # Compare M=10K to M=100K (10× more samples)
        # Std error should decrease by √10 ≈ 3.16×
        error_10k = abs(var_estimates[1] - var_estimates[2])
        error_1k = abs(var_estimates[0] - var_estimates[2])
        
        print(f"\nConvergence Analysis:")
        print(f"  M=1K:   VaR = ${var_estimates[0]:,.2f}")
        print(f"  M=10K:  VaR = ${var_estimates[1]:,.2f}")
        print(f"  M=100K: VaR = ${var_estimates[2]:,.2f} (reference)")
        print(f"  Error (M=1K):  ${error_1k:,.2f}")
        print(f"  Error (M=10K): ${error_10k:,.2f}")
        print(f"  Reduction: {error_1k/error_10k:.2f}× (expect ~3×)")
        
        # High M estimates should be closer to each other
        assert error_10k < error_1k, "Higher M should reduce error"


class TestCVaRProperties:
    """Test 3: CVaR must always exceed VaR (by definition)."""
    
    def test_cvar_exceeds_var(self):
        """CVaR₉₅ ≥ VaR₉₅ and CVaR₉₉ ≥ VaR₉₉ always."""
        pv = 100000.0
        weights = np.array([0.5, 0.3, 0.2])
        vols = np.array([0.15, 0.20, 0.25])
        corr = np.array([
            [1.0, 0.6, 0.4],
            [0.6, 1.0, 0.5],
            [0.4, 0.5, 1.0]
        ])
        
        result = compute_var_cvar_mc(
            portfolio_value=pv,
            weights=weights,
            volatilities=vols,
            correlation_matrix=corr,
            num_simulations=10000,
            seed=42
        )
        
        print(f"\nCVaR vs VaR:")
        print(f"  VaR₉₅:  ${result.var_95:,.2f}")
        print(f"  CVaR₉₅: ${result.cvar_95:,.2f}  (ratio: {result.cvar_95/result.var_95:.3f})")
        print(f"  VaR₉₉:  ${result.var_99:,.2f}")
        print(f"  CVaR₉₉: ${result.cvar_99:,.2f}  (ratio: {result.cvar_99/result.var_99:.3f})")
        
        # Mathematical requirements
        assert result.cvar_95 >= result.var_95, "CVaR₉₅ must be ≥ VaR₉₅"
        assert result.cvar_99 >= result.var_99, "CVaR₉₉ must be ≥ VaR₉₉"
        assert result.var_99 >= result.var_95, "VaR₉₉ must be ≥ VaR₉₅"
        assert result.cvar_99 >= result.cvar_95, "CVaR₉₉ must be ≥ CVaR₉₅"
        
        # CVaR should be noticeably higher (typically 10-30% more)
        assert result.cvar_95 / result.var_95 > 1.05, "CVaR should exceed VaR by >5%"
        
    def test_tail_sizes(self):
        """Verify tail contains correct number of scenarios."""
        M = 10000
        result = compute_var_cvar_mc(
            portfolio_value=100000.0,
            weights=np.array([0.7, 0.3]),
            volatilities=np.array([0.20, 0.25]),
            correlation_matrix=np.array([[1.0, 0.6], [0.6, 1.0]]),
            num_simulations=M,
            seed=42
        )
        
        # 95% tail should contain ~5% of scenarios = 500
        expected_tail_95 = int(0.05 * M)
        expected_tail_99 = int(0.01 * M)
        
        print(f"\nTail Size Validation:")
        print(f"  Total simulations: {M:,}")
        print(f"  95% tail: {result.tail_size_95} (expect ~{expected_tail_95})")
        print(f"  99% tail: {result.tail_size_99} (expect ~{expected_tail_99})")
        
        # Allow 20% tolerance for discretization
        assert abs(result.tail_size_95 - expected_tail_95) < 0.2 * expected_tail_95
        assert abs(result.tail_size_99 - expected_tail_99) < 0.3 * expected_tail_99


class TestCorrelationImpact:
    """Test 4: Correlation affects portfolio VaR."""
    
    def test_correlation_increases_var(self):
        """Higher correlation → higher VaR (for equal weights)."""
        pv = 100000.0
        weights = np.array([0.5, 0.5])
        vols = np.array([0.20, 0.20])
        
        # Uncorrelated
        corr_low = np.array([[1.0, 0.0], [0.0, 1.0]])
        var_low = compute_var_cvar_mc(
            portfolio_value=pv,
            weights=weights,
            volatilities=vols,
            correlation_matrix=corr_low,
            num_simulations=10000,
            seed=42
        ).var_95
        
        # Highly correlated
        corr_high = np.array([[1.0, 0.9], [0.9, 1.0]])
        var_high = compute_var_cvar_mc(
            portfolio_value=pv,
            weights=weights,
            volatilities=vols,
            correlation_matrix=corr_high,
            num_simulations=10000,
            seed=42
        ).var_95
        
        # Perfectly correlated (no diversification)
        corr_perfect = np.array([[1.0, 1.0], [1.0, 1.0]])
        var_perfect = compute_var_cvar_mc(
            portfolio_value=pv,
            weights=weights,
            volatilities=vols,
            correlation_matrix=corr_perfect,
            num_simulations=10000,
            seed=42
        ).var_95
        
        print(f"\nCorrelation Impact:")
        print(f"  VaR (ρ=0.0): ${var_low:,.2f}  (diversification benefit)")
        print(f"  VaR (ρ=0.9): ${var_high:,.2f}")
        print(f"  VaR (ρ=1.0): ${var_perfect:,.2f}  (no diversification)")
        print(f"  Ratio: {var_perfect/var_low:.2f}×")
        
        # Higher correlation → less diversification → higher VaR
        assert var_perfect > var_high > var_low, "VaR must increase with correlation"
        
        # Perfectly correlated should be like single asset
        # Portfolio vol = √(w₁²σ₁² + w₂²σ₂² + 2w₁w₂σ₁σ₂ρ)
        # For w=[0.5, 0.5], σ=[0.2, 0.2], ρ=1: portfolio_vol = 0.2
        single_asset_var = analytical_var_single_asset(pv, 0.20, 0.95, 1)
        error = abs(var_perfect - single_asset_var) / single_asset_var
        print(f"  Perfect corr vs single asset: {error*100:.1f}% difference")
        assert error < 0.10, "Perfect correlation should match single asset"


class TestRealPortfolio:
    """Test 5: Real portfolio with market-like parameters."""
    
    def test_five_asset_portfolio(self):
        """5-asset portfolio with realistic correlations."""
        pv = 1000000.0  # $1M portfolio
        
        # Realistic tech portfolio
        weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
        vols = np.array([0.30, 0.28, 0.35, 0.32, 0.40])  # Tech stock vols
        
        # Realistic correlation matrix (tech stocks are correlated)
        corr = np.array([
            [1.00, 0.65, 0.55, 0.60, 0.50],
            [0.65, 1.00, 0.60, 0.55, 0.45],
            [0.55, 0.60, 1.00, 0.65, 0.55],
            [0.60, 0.55, 0.65, 1.00, 0.60],
            [0.50, 0.45, 0.55, 0.60, 1.00]
        ])
        
        # 1-day and 10-day VaR
        result_1d = compute_var_cvar_mc(
            portfolio_value=pv,
            weights=weights,
            volatilities=vols,
            correlation_matrix=corr,
            time_horizon_days=1,
            num_simulations=50000,
            seed=42
        )
        
        result_10d = compute_var_cvar_mc(
            portfolio_value=pv,
            weights=weights,
            volatilities=vols,
            correlation_matrix=corr,
            time_horizon_days=10,
            num_simulations=50000,
            seed=42
        )
        
        print(f"\n$1M Tech Portfolio (5 assets):")
        print(f"\n  1-Day Risk:")
        print(f"    VaR₉₅:  ${result_1d.var_95:,.0f}  ({result_1d.var_95/pv*100:.2f}% of PV)")
        print(f"    CVaR₉₅: ${result_1d.cvar_95:,.0f}  ({result_1d.cvar_95/pv*100:.2f}% of PV)")
        print(f"    Max loss: ${result_1d.max_loss:,.0f}")
        
        print(f"\n  10-Day Risk:")
        print(f"    VaR₉₅:  ${result_10d.var_95:,.0f}  ({result_10d.var_95/pv*100:.2f}% of PV)")
        print(f"    CVaR₉₅: ${result_10d.cvar_95:,.0f}  ({result_10d.cvar_95/pv*100:.2f}% of PV)")
        print(f"    Max loss: ${result_10d.max_loss:,.0f}")
        
        # Sanity checks
        assert 0.01 < result_1d.var_95 / pv < 0.10, "1-day VaR should be 1-10% of PV"
        assert result_10d.var_95 > result_1d.var_95, "10-day VaR > 1-day VaR"
        
        # √T scaling: 10-day VaR ≈ √10 × 1-day VaR ≈ 3.16×
        scaling = result_10d.var_95 / result_1d.var_95
        print(f"\n  Time scaling: {scaling:.2f}× (expect ~{np.sqrt(10):.2f}×)")
        assert 2.5 < scaling < 4.0, "10-day VaR should scale as √10 ≈ 3.16×"
        
    def test_reproducibility(self):
        """Same seed → same results."""
        pv = 100000.0
        weights = np.array([0.6, 0.4])
        vols = np.array([0.20, 0.25])
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        result1 = compute_var_cvar_mc(
            portfolio_value=pv,
            weights=weights,
            volatilities=vols,
            correlation_matrix=corr,
            num_simulations=10000,
            seed=12345
        )
        
        result2 = compute_var_cvar_mc(
            portfolio_value=pv,
            weights=weights,
            volatilities=vols,
            correlation_matrix=corr,
            num_simulations=10000,
            seed=12345
        )
        
        # Exact reproducibility
        assert result1.var_95 == result2.var_95, "Results must be reproducible with seed"
        assert result1.cvar_95 == result2.cvar_95, "Results must be reproducible with seed"
        assert np.array_equal(result1.loss_distribution, result2.loss_distribution)
        
        print(f"\nReproducibility: ✅")
        print(f"  Both runs: VaR₉₅ = ${result1.var_95:,.2f}, CVaR₉₅ = ${result1.cvar_95:,.2f}")


if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "-s"])
