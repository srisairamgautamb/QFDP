#!/usr/bin/env python3
"""
Real VaR/CVaR Demo
==================

Demonstrates REAL Value-at-Risk and Conditional VaR calculation
using Monte Carlo simulation with correlated returns.

NO shortcuts. NO approximations. ONLY real simulated paths.

Usage:
    python demo_real_var_cvar.py
"""

import numpy as np
from qfdp_multiasset.risk import compute_var_cvar_mc


def demo_single_asset():
    """Demo 1: Single asset VaR/CVaR."""
    print("=" * 80)
    print("DEMO 1: Single Asset VaR/CVaR")
    print("=" * 80)
    
    pv = 1000000.0  # $1M portfolio
    vol = 0.25  # 25% annual volatility
    
    result = compute_var_cvar_mc(
        portfolio_value=pv,
        weights=np.array([1.0]),
        volatilities=np.array([vol]),
        correlation_matrix=np.array([[1.0]]),
        time_horizon_days=1,
        num_simulations=100000,  # High M for accuracy
        seed=42
    )
    
    print(f"\nPortfolio: ${pv:,.0f} (single asset, σ={vol*100:.0f}%)")
    print(f"\n1-Day Risk Metrics:")
    print(f"  VaR₉₅:  ${result.var_95:,.0f}  ({result.var_95/pv*100:.2f}% of portfolio)")
    print(f"  CVaR₉₅: ${result.cvar_95:,.0f}  ({result.cvar_95/pv*100:.2f}% of portfolio)")
    print(f"  VaR₉₉:  ${result.var_99:,.0f}  ({result.var_99/pv*100:.2f}% of portfolio)")
    print(f"  CVaR₉₉: ${result.cvar_99:,.0f}  ({result.cvar_99/pv*100:.2f}% of portfolio)")
    
    print(f"\nInterpretation:")
    print(f"  - 95% of days: losses ≤ ${result.var_95:,.0f}")
    print(f"  - Worst 5% of days: average loss = ${result.cvar_95:,.0f}")
    print(f"  - Worst single day (simulated): ${result.max_loss:,.0f}")
    
    print(f"\nSimulation Details:")
    print(f"  Paths: {result.num_simulations:,}")
    print(f"  95% tail: {result.tail_size_95} scenarios")
    print(f"  99% tail: {result.tail_size_99} scenarios")


def demo_diversified_portfolio():
    """Demo 2: Multi-asset portfolio with diversification benefit."""
    print("\n" + "=" * 80)
    print("DEMO 2: Diversified Portfolio - Correlation Impact")
    print("=" * 80)
    
    pv = 1000000.0
    weights = np.array([0.4, 0.3, 0.3])
    vols = np.array([0.30, 0.25, 0.35])  # Different volatilities
    
    # Test different correlation scenarios
    scenarios = [
        ("Uncorrelated (ρ=0)", np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
        ("Moderate (ρ=0.5)", np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])),
        ("High (ρ=0.9)", np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]])),
    ]
    
    print(f"\n3-Asset Portfolio: ${pv:,.0f}")
    print(f"  Weights: {weights}")
    print(f"  Volatilities: {vols*100}%")
    
    print(f"\n{'Scenario':<25} {'VaR₉₅':>15} {'CVaR₉₅':>15} {'Div. Benefit':>15}")
    print("-" * 75)
    
    base_var = None
    for name, corr in scenarios:
        result = compute_var_cvar_mc(
            portfolio_value=pv,
            weights=weights,
            volatilities=vols,
            correlation_matrix=corr,
            time_horizon_days=1,
            num_simulations=50000,
            seed=42
        )
        
        if base_var is None:
            base_var = result.var_95
            benefit = "baseline"
        else:
            benefit = f"{(1 - base_var/result.var_95)*100:+.1f}%"
        
        print(f"{name:<25} ${result.var_95:>13,.0f} ${result.cvar_95:>13,.0f} {benefit:>15}")
    
    print("\nKey Insight:")
    print("  Lower correlation → Better diversification → Lower VaR")
    print("  Higher correlation → Less diversification → Higher VaR")


def demo_time_scaling():
    """Demo 3: Time horizon scaling (√T rule)."""
    print("\n" + "=" * 80)
    print("DEMO 3: Time Horizon Scaling")
    print("=" * 80)
    
    pv = 1000000.0
    weights = np.array([0.5, 0.5])
    vols = np.array([0.28, 0.32])
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    
    horizons = [1, 5, 10, 21]  # 1 day, 1 week, 2 weeks, 1 month
    
    print(f"\n2-Asset Portfolio: ${pv:,.0f}")
    print(f"  Correlation: 0.6")
    
    print(f"\n{'Horizon':<15} {'VaR₉₅':>15} {'CVaR₉₅':>15} {'√T Scaling':>15}")
    print("-" * 65)
    
    var_1d = None
    for days in horizons:
        result = compute_var_cvar_mc(
            portfolio_value=pv,
            weights=weights,
            volatilities=vols,
            correlation_matrix=corr,
            time_horizon_days=days,
            num_simulations=50000,
            seed=42
        )
        
        if var_1d is None:
            var_1d = result.var_95
            scaling = "1.00×"
        else:
            scaling = f"{result.var_95/var_1d:.2f}× (√{days}={np.sqrt(days):.2f})"
        
        print(f"{days}-day{'':<9} ${result.var_95:>13,.0f} ${result.cvar_95:>13,.0f} {scaling:>15}")
    
    print("\nKey Insight:")
    print("  VaR scales approximately as √T (square root of time)")
    print("  10-day VaR ≈ √10 × 1-day VaR ≈ 3.16 × 1-day VaR")


def demo_real_tech_portfolio():
    """Demo 4: Realistic tech portfolio."""
    print("\n" + "=" * 80)
    print("DEMO 4: Realistic Tech Portfolio ($10M)")
    print("=" * 80)
    
    pv = 10000000.0  # $10M
    
    # 5 tech stocks with realistic parameters
    symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
    weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
    vols = np.array([0.30, 0.28, 0.32, 0.45, 0.60])  # TSLA much more volatile
    
    # Realistic correlation (tech stocks are correlated)
    corr = np.array([
        [1.00, 0.70, 0.65, 0.55, 0.40],
        [0.70, 1.00, 0.68, 0.58, 0.35],
        [0.65, 0.68, 1.00, 0.60, 0.38],
        [0.55, 0.58, 0.60, 1.00, 0.50],
        [0.40, 0.35, 0.38, 0.50, 1.00]
    ])
    
    print(f"\nPortfolio Composition:")
    for sym, w, v in zip(symbols, weights, vols):
        print(f"  {sym}: {w*100:5.1f}%  (σ={v*100:.0f}%)")
    
    # 1-day VaR
    result_1d = compute_var_cvar_mc(
        portfolio_value=pv,
        weights=weights,
        volatilities=vols,
        correlation_matrix=corr,
        time_horizon_days=1,
        num_simulations=100000,
        seed=42
    )
    
    # 10-day VaR (regulatory requirement)
    result_10d = compute_var_cvar_mc(
        portfolio_value=pv,
        weights=weights,
        volatilities=vols,
        correlation_matrix=corr,
        time_horizon_days=10,
        num_simulations=100000,
        seed=42
    )
    
    print(f"\n1-Day Risk:")
    print(f"  VaR₉₅:  ${result_1d.var_95:,.0f}  ({result_1d.var_95/pv*100:.2f}% of $10M)")
    print(f"  CVaR₉₅: ${result_1d.cvar_95:,.0f}  ({result_1d.cvar_95/pv*100:.2f}% of $10M)")
    
    print(f"\n10-Day Risk (Regulatory):")
    print(f"  VaR₉₅:  ${result_10d.var_95:,.0f}  ({result_10d.var_95/pv*100:.2f}% of $10M)")
    print(f"  CVaR₉₅: ${result_10d.cvar_95:,.0f}  ({result_10d.cvar_95/pv*100:.2f}% of $10M)")
    
    print(f"\nCapital Requirement (Basel III-like):")
    capital_req = result_10d.cvar_95  # CVaR₉₅ for 10 days
    print(f"  Required capital: ${capital_req:,.0f}  ({capital_req/pv*100:.1f}% of portfolio)")
    
    print(f"\nSimulation Quality:")
    print(f"  Paths simulated: {result_1d.num_simulations:,}")
    print(f"  95% tail scenarios: {result_1d.tail_size_95}")
    print(f"  Loss std dev: ${result_1d.std_loss:,.0f}")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "REAL VAR/CVAR DEMONSTRATION" + " " * 31 + "║")
    print("║" + " " * 15 + "NO Shortcuts | NO Approximations | ONLY Real MC" + " " * 15 + "║")
    print("╚" + "═" * 78 + "╝")
    
    demo_single_asset()
    demo_diversified_portfolio()
    demo_time_scaling()
    demo_real_tech_portfolio()
    
    print("\n" + "=" * 80)
    print("✅ ALL DEMOS COMPLETE - VAR/CVAR ARE 100% REAL")
    print("=" * 80)
    print("\nMethod:")
    print("  1. Cholesky decomposition of correlation matrix")
    print("  2. Sample correlated normal returns (10K-100K paths)")
    print("  3. Compute portfolio losses for each path")
    print("  4. VaR = 95th percentile of losses")
    print("  5. CVaR = mean of losses exceeding VaR")
    print("\nNo parametric formulas. No shortcuts. Pure Monte Carlo simulation.")
    print()


if __name__ == "__main__":
    main()
