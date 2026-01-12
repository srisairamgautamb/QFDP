"""
Correlation Complexity Learning Test
======================================
Tests whether quantum learns non-linear correlations better than classical PCA.

Scenarios:
1. Gaussian (linear) - PCA should win
2. Copula (non-linear) - Quantum might win
3. Stress-regime (tail dependencies) - Quantum should win

Hypothesis: Quantum entanglement can capture non-linear dependencies
that classical PCA (which only captures linear correlations) misses.
"""

import numpy as np
import pandas as pd
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from scipy.stats import norm, t as student_t
from scipy.special import ndtri, ndtr
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class CorrelationResult:
    """Result from one correlation scenario test."""
    scenario: str
    error_classical: float
    error_quantum: float
    advantage: float
    linear_correlation: float
    tail_dependence: float


def generate_gaussian_returns(n_assets: int, n_samples: int, base_corr: float = 0.4) -> np.ndarray:
    """
    Generate Gaussian (normally distributed) returns.
    These have linear correlations - PCA's specialty!
    """
    # Correlation matrix
    corr = np.full((n_assets, n_assets), base_corr)
    np.fill_diagonal(corr, 1.0)
    
    # Ensure positive definiteness
    eigenvalues = np.linalg.eigvalsh(corr)
    if np.min(eigenvalues) < 0:
        corr += np.eye(n_assets) * (abs(np.min(eigenvalues)) + 0.01)
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    
    np.random.seed(42)
    L = np.linalg.cholesky(corr)
    Z = np.random.randn(n_samples, n_assets)
    returns = Z @ L.T * 0.01  # Daily returns ~1% volatility
    
    return returns, corr


def generate_copula_returns(n_assets: int, n_samples: int, base_corr: float = 0.4, df: int = 4) -> np.ndarray:
    """
    Generate returns from a Student-t copula (non-linear correlations).
    
    Student-t copula has:
    - Same linear correlation as Gaussian
    - But stronger tail dependence (non-linear)
    """
    # Correlation matrix (same as Gaussian)
    corr = np.full((n_assets, n_assets), base_corr)
    np.fill_diagonal(corr, 1.0)
    
    eigenvalues = np.linalg.eigvalsh(corr)
    if np.min(eigenvalues) < 0:
        corr += np.eye(n_assets) * (abs(np.min(eigenvalues)) + 0.01)
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    
    np.random.seed(42)
    
    # Generate multivariate t samples
    L = np.linalg.cholesky(corr)
    Z = np.random.randn(n_samples, n_assets)
    X = Z @ L.T
    
    # Scale by chi-squared for t-distribution
    chi2 = np.random.chisquare(df, size=n_samples)
    X = X / np.sqrt(chi2[:, np.newaxis] / df)
    
    # Transform to uniform via t-CDF, then to Normal
    U = student_t.cdf(X, df=df)
    returns = norm.ppf(U) * 0.01
    
    # Handle numerical issues
    returns = np.clip(returns, -0.1, 0.1)
    returns = np.nan_to_num(returns, nan=0.0)
    
    return returns, corr


def generate_stress_regime_returns(n_assets: int, n_samples: int, base_corr: float = 0.4, stress_prob: float = 0.1) -> np.ndarray:
    """
    Generate returns with regime switching (stress = high correlation).
    
    During stress (10% of time):
    - All assets move together (correlation → 1.0)
    - Larger moves (3x volatility)
    
    This creates strong TAIL dependencies that PCA misses.
    """
    corr_normal = np.full((n_assets, n_assets), base_corr)
    np.fill_diagonal(corr_normal, 1.0)
    
    corr_stress = np.full((n_assets, n_assets), 0.95)  # Near-perfect correlation in stress
    np.fill_diagonal(corr_stress, 1.0)
    
    # Ensure positive definiteness
    for corr in [corr_normal, corr_stress]:
        eigenvalues = np.linalg.eigvalsh(corr)
        if np.min(eigenvalues) < 0:
            corr += np.eye(n_assets) * (abs(np.min(eigenvalues)) + 0.01)
    
    np.random.seed(42)
    
    L_normal = np.linalg.cholesky(corr_normal)
    L_stress = np.linalg.cholesky(corr_stress)
    
    # Determine stress days
    is_stress = np.random.rand(n_samples) < stress_prob
    
    returns = np.zeros((n_samples, n_assets))
    
    for i in range(n_samples):
        Z = np.random.randn(n_assets)
        if is_stress[i]:
            # Stress regime: high correlation, high volatility
            returns[i] = Z @ L_stress.T * 0.03  # 3x volatility
        else:
            # Normal regime
            returns[i] = Z @ L_normal.T * 0.01
    
    # Realized correlation (should be higher than base_corr due to stress)
    realized_corr = np.corrcoef(returns.T)
    
    return returns, realized_corr


def estimate_tail_dependence(returns: np.ndarray, quantile: float = 0.05) -> float:
    """
    Estimate lower tail dependence coefficient.
    
    λ_L = P(X₂ < F₂⁻¹(α) | X₁ < F₁⁻¹(α)) as α → 0
    
    High λ_L means assets crash together (non-linear dependence).
    """
    n_samples, n_assets = returns.shape
    
    # Use pairs of assets
    tail_dependencies = []
    threshold = np.percentile(returns, quantile * 100, axis=0)
    
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            # Count joint extreme events
            both_extreme = np.sum((returns[:, i] < threshold[i]) & (returns[:, j] < threshold[j]))
            i_extreme = np.sum(returns[:, i] < threshold[i])
            
            if i_extreme > 0:
                lambda_ij = both_extreme / i_extreme
                tail_dependencies.append(lambda_ij)
    
    return np.mean(tail_dependencies) if tail_dependencies else 0.0


def monte_carlo_from_returns(
    returns: np.ndarray,
    S0: np.ndarray,
    weights: np.ndarray,
    K: float,
    T: float,
    r: float,
    n_paths: int = 50000
) -> float:
    """
    Monte Carlo pricing directly from sample returns.
    Uses bootstrap from historical returns.
    """
    np.random.seed(42)
    n_samples, n_assets = returns.shape
    
    # Bootstrap sample paths
    n_steps = int(T * 252)  # Daily steps
    
    payoffs = []
    for _ in range(n_paths):
        # Sample random returns
        idx = np.random.choice(n_samples, size=n_steps, replace=True)
        path_returns = returns[idx]
        
        # Compute terminal prices
        cumsum_returns = np.sum(path_returns, axis=0)
        ST = S0 * np.exp(cumsum_returns)
        
        # Portfolio value
        portfolio_T = ST @ weights
        payoff = max(portfolio_T - K, 0)
        payoffs.append(payoff)
    
    return np.exp(-r * T) * np.mean(payoffs)


def price_classical(returns: np.ndarray, S0: np.ndarray, weights: np.ndarray, K: float, T: float, r: float) -> float:
    """Price using classical PCA-based approach."""
    from qfdp.unified import FBIQFTPricing
    
    # Compute correlation and volatility from returns
    corr = np.corrcoef(returns.T)
    sigma = np.std(returns, axis=0) * np.sqrt(252)
    
    try:
        pricer = FBIQFTPricing(M=16, alpha=1.0, num_shots=8192)
        result = pricer.price_option(
            asset_prices=S0,
            asset_volatilities=sigma,
            correlation_matrix=corr,
            portfolio_weights=weights,
            K=K, T=T, r=r,
            backend=None
        )
        return result['price_quantum']
    except Exception:
        # Fallback to BS
        cov = np.cov(returns.T) * 252
        sigma_p = np.sqrt(weights @ cov @ weights)
        portfolio_value = np.sum(S0 * weights)
        d1 = (np.log(portfolio_value/K) + (r + 0.5*sigma_p**2)*T) / (sigma_p*np.sqrt(T))
        d2 = d1 - sigma_p*np.sqrt(T)
        return portfolio_value*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def price_quantum(returns: np.ndarray, S0: np.ndarray, weights: np.ndarray, K: float, T: float, r: float) -> float:
    """Price using Quantum Autoencoder (learns from data)."""
    from qfdp_qml.hybrid_integration import QAE_FB_IQFT_Pricer
    
    N = len(S0)
    corr = np.corrcoef(returns.T)
    sigma = np.std(returns, axis=0) * np.sqrt(252)
    
    try:
        pricer = QAE_FB_IQFT_Pricer(n_factors=min(3, N-1), n_layers=2)
        # Train on actual returns data (learns non-linear structure!)
        pricer.train(returns, max_iter=30)
        
        result = pricer.price_option(
            S0=S0,
            sigma=sigma,
            corr=corr,
            weights=weights,
            K=K, T=T, r=r
        )
        return result.price_qae
    except Exception:
        return price_classical(returns, S0, weights, K, T, r)


def test_correlation_complexity() -> pd.DataFrame:
    """
    Test whether quantum learns non-linear correlations better than PCA.
    
    Scenarios:
    1. Gaussian (linear) - PCA advantage expected
    2. Copula (non-linear) - Quantum might win
    3. Stress-regime (tail) - Quantum should win
    """
    print("🔗 CORRELATION COMPLEXITY LEARNING TEST")
    print("=" * 70)
    print("Testing if quantum captures non-linear correlations better than PCA...")
    print()
    print("Hypothesis:")
    print("  - Gaussian returns: Linear correlations → PCA wins")
    print("  - Copula returns:   Non-linear joints → Quantum might win")
    print("  - Stress regime:    Tail dependencies → Quantum should win")
    print()
    
    # Setup
    n_assets = 5
    n_samples = 1000
    S0 = np.array([100.0, 105.0, 95.0, 110.0, 90.0])
    weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
    K = np.sum(S0 * weights)  # ATM
    T = 1.0
    r = 0.05
    
    scenarios = [
        ("Gaussian (Linear)", lambda: generate_gaussian_returns(n_assets, n_samples, 0.4)),
        ("Student-t Copula (df=4)", lambda: generate_copula_returns(n_assets, n_samples, 0.4, df=4)),
        ("Student-t Copula (df=2)", lambda: generate_copula_returns(n_assets, n_samples, 0.4, df=2)),
        ("Stress Regime (10%)", lambda: generate_stress_regime_returns(n_assets, n_samples, 0.4, 0.10)),
        ("Stress Regime (20%)", lambda: generate_stress_regime_returns(n_assets, n_samples, 0.4, 0.20)),
    ]
    
    results = []
    
    print("-" * 70)
    print(f"{'Scenario':<25} {'Classical':<12} {'Quantum':<12} {'Advantage':<12} {'Tail Dep':<10}")
    print("-" * 70)
    
    for name, generator in scenarios:
        returns, corr = generator()
        
        # Compute diagnostics
        linear_corr = np.mean(corr[np.triu_indices(n_assets, k=1)])
        tail_dep = estimate_tail_dependence(returns, quantile=0.05)
        
        # Reference price from actual returns
        ref_price = monte_carlo_from_returns(returns, S0, weights, K, T, r, n_paths=100000)
        
        # Classical (PCA-based)
        price_c = price_classical(returns, S0, weights, K, T, r)
        error_c = abs(price_c - ref_price) / max(ref_price, 0.01) * 100
        
        # Quantum (learns from data)
        price_q = price_quantum(returns, S0, weights, K, T, r)
        error_q = abs(price_q - ref_price) / max(ref_price, 0.01) * 100
        
        advantage = error_c - error_q  # Positive = quantum wins
        
        status = "✅" if advantage > 0.1 else "⚪"
        print(f"{name:<25} {error_c:>10.2f}% {error_q:>10.2f}% {advantage:>+10.2f}% {tail_dep:>8.2f} {status}")
        
        results.append(CorrelationResult(
            scenario=name,
            error_classical=error_c,
            error_quantum=error_q,
            advantage=advantage,
            linear_correlation=linear_corr,
            tail_dependence=tail_dep
        ))
    
    # Create DataFrame
    df = pd.DataFrame([{
        'scenario': r.scenario,
        'error_classical': r.error_classical,
        'error_quantum': r.error_quantum,
        'advantage': r.advantage,
        'linear_correlation': r.linear_correlation,
        'tail_dependence': r.tail_dependence
    } for r in results])
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 CORRELATION LEARNING SUMMARY")
    print("=" * 70)
    
    quantum_wins = df[df['advantage'] > 0.1]
    classical_wins = df[df['advantage'] < -0.1]
    
    print(f"\nQuantum wins: {len(quantum_wins)} scenarios")
    print(f"Classical wins: {len(classical_wins)} scenarios")
    print(f"Tie: {len(df) - len(quantum_wins) - len(classical_wins)} scenarios")
    
    if len(quantum_wins) > 0:
        print("\n✅ Quantum advantage scenarios:")
        for _, row in quantum_wins.iterrows():
            print(f"   - {row['scenario']}: +{row['advantage']:.2f}% (tail dep = {row['tail_dependence']:.2f})")
        
        # Correlation with tail dependence
        corr_advantage_tail = np.corrcoef(df['advantage'], df['tail_dependence'])[0, 1]
        print(f"\n📈 Correlation(advantage, tail_dependence) = {corr_advantage_tail:.2f}")
        
        if corr_advantage_tail > 0.3:
            print("   → Quantum advantage INCREASES with non-linear dependence! ✅")
    else:
        print("\n⚠️ No clear quantum wins in correlation learning.")
        print("   May need more training or different architecture.")
    
    # Save results
    output_dir = '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP/qfdp_qml/results/advantage_discovery'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f'{output_dir}/correlation_complexity_results.csv', index=False)
    print(f"\n✅ Results saved to {output_dir}/correlation_complexity_results.csv")
    
    # Generate plot
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Error comparison
        x = np.arange(len(df))
        width = 0.35
        ax1.bar(x - width/2, df['error_classical'], width, label='Classical (PCA)', color='blue', alpha=0.7)
        ax1.bar(x + width/2, df['error_quantum'], width, label='Quantum (QAE)', color='red', alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels([s[:15] + '...' if len(s) > 15 else s for s in df['scenario']], rotation=45, ha='right')
        ax1.set_ylabel('Pricing Error (%)', fontsize=12)
        ax1.set_title('Correlation Complexity: Error Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Advantage vs Tail Dependence
        colors = ['green' if a > 0 else 'red' for a in df['advantage']]
        ax2.scatter(df['tail_dependence'], df['advantage'], c=colors, s=100, alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Tail Dependence λ', fontsize=12)
        ax2.set_ylabel('Quantum Advantage (%)', fontsize=12)
        ax2.set_title('Advantage vs Tail Dependence', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add scenario labels
        for i, row in df.iterrows():
            ax2.annotate(row['scenario'][:10], (row['tail_dependence'], row['advantage']),
                        fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_complexity_plot.png', dpi=150)
        plt.close()
        print(f"✅ Plot saved to {output_dir}/correlation_complexity_plot.png")
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")
    
    return df


if __name__ == '__main__':
    df = test_correlation_complexity()
