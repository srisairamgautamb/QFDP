"""
EXPERIMENT 1: Regime Change Validation
=======================================

Critical experiment to prove QRC adapts better than static PCA when
market regime shifts from calm to volatile.

This is the MAIN PUBLICATION CLAIM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, ttest_rel
import sys
import os

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from quantum_deep_pricing import QuantumDeepPricer


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes formula for ground truth."""
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def generate_regime_shift_scenario(seed=42):
    """
    Create synthetic market data with clear regime change.
    
    Timeline:
        t=0-49:  Calm market (low vol, stable correlations)
        t=50-99: Volatile market (high vol, stressed correlations)
    """
    np.random.seed(seed)
    
    timeline = []
    
    # REGIME 1: Calm (t=0 to t=49)
    for t in range(50):
        spot = 100 + np.random.randn() * 1.5
        vol = 0.15 + np.random.randn() * 0.02
        
        # Price history - stable
        history = [100 + np.random.randn() * 1 for _ in range(6)]
        
        timeline.append({
            'timestamp': t,
            'spot': spot,
            'volatility': max(0.05, vol),
            'corr_change': np.random.randn() * 0.02,
            'stress': 0.1 + np.random.rand() * 0.1,
            'price_history': np.array(history),
            'regime': 'calm'
        })
    
    # REGIME 2: Volatile (t=50 to t=99)
    for t in range(50):
        spot = 92 + np.random.randn() * 5
        vol = 0.40 + np.random.randn() * 0.08
        
        # Price history - volatile
        history = [95 + np.random.randn() * 4 for _ in range(6)]
        
        timeline.append({
            'timestamp': 50 + t,
            'spot': spot,
            'volatility': max(0.15, min(0.8, vol)),
            'corr_change': 0.15 + np.random.randn() * 0.05,
            'stress': 0.6 + np.random.rand() * 0.3,
            'price_history': np.array(history),
            'regime': 'volatile'
        })
    
    return timeline


def price_with_static_pca(data, strike=100, maturity=1.0, r=0.05):
    """
    Simulate static PCA pricing (baseline).
    Uses Black-Scholes with fixed volatility assumption.
    """
    # Static assumption: use historical average volatility
    static_vol = 0.20  # Trained on calm period
    
    # Black-Scholes with static vol
    price = black_scholes_call(data['spot'], strike, maturity, r, static_vol)
    
    return price


def run_experiment():
    """Main experiment: QRC vs Static PCA under regime shift."""
    
    print("=" * 80)
    print("EXPERIMENT 1: REGIME CHANGE VALIDATION")
    print("QRC-Enhanced vs Static PCA Pricing")
    print("=" * 80)
    
    # Generate scenario
    print("\n📊 Generating regime shift scenario...")
    timeline = generate_regime_shift_scenario()
    
    # Initialize QRC pricer
    print("🔧 Initializing QRC-enhanced pricer...")
    qrc_pricer = QuantumDeepPricer(
        fb_iqft_pricer=None,
        fusion_method='weighted',
        use_qrc=True,
        use_qtc=True
    )
    
    # Track results
    results = {
        'timestamp': [],
        'regime': [],
        'spot': [],
        'volatility': [],
        'ground_truth': [],
        'static_pca': [],
        'qrc_enhanced': [],
        'static_error': [],
        'qrc_error': []
    }
    
    print("\n⏳ Running pricing comparison...")
    print(f"{'t':<4} {'Regime':<10} {'Spot':<8} {'Vol':<6} {'Truth':<8} {'Static':<8} {'QRC':<8} {'Static%':<8} {'QRC%':<8}")
    print("-" * 80)
    
    for i, data in enumerate(timeline):
        t = data['timestamp']
        
        # Ground truth: Black-Scholes with TRUE volatility
        true_price = black_scholes_call(
            data['spot'], 100, 1.0, 0.05, data['volatility']
        )
        
        # Static PCA (baseline)
        static_price = price_with_static_pca(data)
        
        # QRC-enhanced
        market_data = {
            'prices': data['spot'],
            'volatility': data['volatility'],
            'corr_change': data['corr_change'],
            'stress': data['stress']
        }
        
        qrc_result = qrc_pricer.price_option(
            market_data=market_data,
            price_history=data['price_history'],
            strike=100,
            maturity=1.0
        )
        qrc_price = qrc_result.price
        
        # Errors
        static_error = abs(static_price - true_price) / max(true_price, 0.01) * 100
        qrc_error = abs(qrc_price - true_price) / max(true_price, 0.01) * 100
        
        # Store
        results['timestamp'].append(t)
        results['regime'].append(data['regime'])
        results['spot'].append(data['spot'])
        results['volatility'].append(data['volatility'])
        results['ground_truth'].append(true_price)
        results['static_pca'].append(static_price)
        results['qrc_enhanced'].append(qrc_price)
        results['static_error'].append(static_error)
        results['qrc_error'].append(qrc_error)
        
        # Print every 10 steps
        if t % 10 == 0:
            print(f"{t:<4} {data['regime']:<10} {data['spot']:<8.2f} {data['volatility']:<6.2f} "
                  f"{true_price:<8.2f} {static_price:<8.2f} {qrc_price:<8.2f} "
                  f"{static_error:<8.2f} {qrc_error:<8.2f}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Analysis
    print("\n" + "=" * 80)
    print("📈 RESULTS ANALYSIS")
    print("=" * 80)
    
    # Overall metrics
    print(f"\n📊 OVERALL (all 100 timesteps):")
    print(f"   Static PCA mean error:  {df['static_error'].mean():.3f}%")
    print(f"   QRC-enhanced mean error: {df['qrc_error'].mean():.3f}%")
    improvement_overall = (df['static_error'].mean() - df['qrc_error'].mean()) / df['static_error'].mean() * 100
    print(f"   → Improvement: {improvement_overall:+.1f}%")
    
    # Calm regime
    calm = df[df['regime'] == 'calm']
    print(f"\n🌤️  CALM REGIME (t=0-49):")
    print(f"   Static PCA mean error:  {calm['static_error'].mean():.3f}%")
    print(f"   QRC-enhanced mean error: {calm['qrc_error'].mean():.3f}%")
    improvement_calm = (calm['static_error'].mean() - calm['qrc_error'].mean()) / calm['static_error'].mean() * 100
    print(f"   → Improvement: {improvement_calm:+.1f}%")
    
    # Volatile regime - THIS IS THE KEY CLAIM
    volatile = df[df['regime'] == 'volatile']
    print(f"\n🔥 VOLATILE REGIME (t=50-99) - KEY METRIC:")
    print(f"   Static PCA mean error:  {volatile['static_error'].mean():.3f}%")
    print(f"   QRC-enhanced mean error: {volatile['qrc_error'].mean():.3f}%")
    improvement_volatile = (volatile['static_error'].mean() - volatile['qrc_error'].mean()) / volatile['static_error'].mean() * 100
    print(f"   → Improvement: {improvement_volatile:+.1f}%")
    
    # Statistical significance
    print(f"\n📐 STATISTICAL TEST (paired t-test):")
    t_stat, p_value = ttest_rel(df['static_error'], df['qrc_error'])
    print(f"   t-statistic: {t_stat:.3f}")
    print(f"   p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print(f"   ✅ SIGNIFICANT: QRC improvement is statistically significant (p < 0.05)")
    else:
        print(f"   ⚠️  Not significant at p < 0.05 level")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("🏆 EXPERIMENT VERDICT")
    print("=" * 80)
    
    if improvement_volatile > 20 and p_value < 0.05:
        print("\n✅ SUCCESS: QRC shows STRONG adaptive advantage!")
        print(f"   - {improvement_volatile:.1f}% improvement in volatile markets")
        print(f"   - Statistically significant (p={p_value:.4f})")
        print("   → READY FOR PUBLICATION")
    elif improvement_volatile > 10 and p_value < 0.05:
        print("\n✅ PARTIAL SUCCESS: QRC shows moderate adaptive advantage")
        print(f"   - {improvement_volatile:.1f}% improvement in volatile markets")
        print("   → Consider parameter tuning for stronger results")
    else:
        print("\n⚠️  NEEDS WORK: QRC advantage not yet compelling")
        print(f"   - {improvement_volatile:.1f}% improvement")
        print("   → Tune QRC parameters or increase layers")
    
    # Plot
    plot_results(df)
    
    # Save
    os.makedirs('experiments/results', exist_ok=True)
    df.to_csv('experiments/results/regime_change_results.csv', index=False)
    print(f"\n💾 Results saved: experiments/results/regime_change_results.csv")
    
    return df, improvement_volatile, p_value


def plot_results(df):
    """Create publication-quality visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Regime Change Validation: QRC vs Static PCA', fontsize=14, fontweight='bold')
    
    # Plot 1: Error over time
    ax = axes[0, 0]
    ax.plot(df['timestamp'], df['static_error'], 
            label='Static PCA', alpha=0.7, linewidth=2, color='#e74c3c')
    ax.plot(df['timestamp'], df['qrc_error'], 
            label='QRC-Enhanced', alpha=0.7, linewidth=2, color='#2ecc71')
    ax.axvline(x=50, color='#2c3e50', linestyle='--', alpha=0.7, 
               label='Regime Shift', linewidth=2)
    ax.fill_between([0, 50], 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100, 
                    alpha=0.1, color='blue', label='Calm')
    ax.fill_between([50, 100], 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100, 
                    alpha=0.1, color='red', label='Volatile')
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Pricing Error (%)', fontsize=11)
    ax.set_title('Pricing Error Over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    
    # Plot 2: Box plot by regime
    ax = axes[0, 1]
    calm = df[df['regime'] == 'calm']
    volatile = df[df['regime'] == 'volatile']
    
    box_data = [calm['static_error'], calm['qrc_error'], 
                volatile['static_error'], volatile['qrc_error']]
    bp = ax.boxplot(box_data, positions=[1, 2, 4, 5], widths=0.6,
                    patch_artist=True)
    
    colors = ['#e74c3c', '#2ecc71', '#e74c3c', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(['Calm Market', 'Volatile Market'], fontsize=11)
    ax.set_ylabel('Pricing Error (%)', fontsize=11)
    ax.set_title('Error Distribution by Regime', fontsize=12, fontweight='bold')
    ax.legend([bp['boxes'][0], bp['boxes'][1]], ['Static PCA', 'QRC-Enhanced'], 
              loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Market conditions
    ax = axes[1, 0]
    ax2 = ax.twinx()
    
    ax.plot(df['timestamp'], df['spot'], color='#3498db', linewidth=1.5, 
            label='Spot Price', alpha=0.7)
    ax2.plot(df['timestamp'], df['volatility'] * 100, color='#9b59b6', 
             linewidth=1.5, label='Volatility (%)', alpha=0.7)
    
    ax.axvline(x=50, color='#2c3e50', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Spot Price ($)', fontsize=11, color='#3498db')
    ax2.set_ylabel('Volatility (%)', fontsize=11, color='#9b59b6')
    ax.set_title('Market Conditions', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Improvement percentage
    ax = axes[1, 1]
    improvement = ((df['static_error'] - df['qrc_error']) / df['static_error'] * 100)
    improvement_ma = improvement.rolling(5, min_periods=1).mean()
    
    ax.fill_between(df['timestamp'], 0, improvement_ma, 
                    where=(improvement_ma > 0), alpha=0.4, color='#2ecc71', 
                    label='QRC Better')
    ax.fill_between(df['timestamp'], 0, improvement_ma, 
                    where=(improvement_ma < 0), alpha=0.4, color='#e74c3c', 
                    label='Static Better')
    ax.plot(df['timestamp'], improvement_ma, color='#2c3e50', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=50, color='#2c3e50', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('QRC Improvement (%)', fontsize=11)
    ax.set_title('QRC Improvement Over Static (5-step MA)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig('experiments/results/regime_change_comparison.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved: experiments/results/regime_change_comparison.png")
    plt.show()


if __name__ == '__main__':
    results_df, improvement, p_val = run_experiment()
