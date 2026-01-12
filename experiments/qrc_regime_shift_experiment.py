"""
QRC vs PCA Regime Shift Experiment
==================================

This experiment demonstrates QRC's ability to adapt to correlation regime shifts.

Scenario:
- PCA is calibrated on CALM market (ρ=0.3)
- Market shifts to STRESSED (ρ=0.8)
- PCA uses STALE calibration (wrong σ_p)
- QRC adapts factors to new regime (correct σ_p)

Expected Result:
- PCA error increases when market shifts
- QRC error stays low by adapting
"""

import numpy as np
from scipy.stats import norm
import sys
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
from qfdp.unified.adapter_layer import BaseModelAdapter
from qrc import QuantumRecurrentCircuit


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes analytical reference."""
    if sigma <= 0:
        return max(S - K * np.exp(-r * T), 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def run_experiment():
    """Main experiment: QRC vs PCA under regime shift."""
    
    print("=" * 80)
    print("🧪 QRC vs PCA REGIME SHIFT EXPERIMENT")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize components
    qrc = QuantumRecurrentCircuit(n_factors=4)
    adapter = BaseModelAdapter(beta=0.1)
    
    # Market parameters
    n_assets = 5
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    base_vol = 0.20
    
    asset_prices = np.full(n_assets, S)
    asset_vols = np.full(n_assets, base_vol)
    weights = np.ones(n_assets) / n_assets
    
    # =================================================================
    # PHASE 1: CALIBRATION ON CALM MARKET
    # =================================================================
    print("\n" + "=" * 80)
    print("📊 PHASE 1: CALIBRATION (Calm Market, ρ = 0.3)")
    print("=" * 80)
    
    rho_calm = 0.3
    corr_calm = np.eye(n_assets) + rho_calm * (1 - np.eye(n_assets))
    
    # PCA calibration
    pca_calm = adapter.prepare_for_pca_pricing(asset_prices, asset_vols, corr_calm, weights)
    sigma_p_calibrated = pca_calm['sigma_p_pca']
    
    # True price in calm market
    bs_calm = black_scholes_call(S, K, T, r, sigma_p_calibrated)
    cm_calm = price_call_option_corrected(S, K, T, r, sigma_p_calibrated)['price']
    
    print(f"   σ_p (calibrated): {sigma_p_calibrated:.4f}")
    print(f"   BS price:         ${bs_calm:.4f}")
    print(f"   CM price:         ${cm_calm:.4f}")
    print(f"   CM-BS error:      {abs(cm_calm-bs_calm)/bs_calm*100:.4f}%")
    
    # =================================================================
    # PHASE 2: TEST ACROSS REGIME SHIFTS
    # =================================================================
    print("\n" + "=" * 80)
    print("📈 PHASE 2: REGIME SHIFT TESTING")
    print("=" * 80)
    
    regimes = [
        {'name': 'Calm', 'rho': 0.3, 'stress': 0.2},
        {'name': 'Mild Stress', 'rho': 0.5, 'stress': 0.4},
        {'name': 'Moderate', 'rho': 0.6, 'stress': 0.6},
        {'name': 'High Stress', 'rho': 0.7, 'stress': 0.7},
        {'name': 'Crisis', 'rho': 0.8, 'stress': 0.9},
    ]
    
    results = []
    
    print(f"\n{'Regime':<14} {'True σ_p':<10} {'PCA σ_p':<10} {'QRC σ_p':<10} {'True Price':<12} {'PCA Error':<12} {'QRC Error':<12}")
    print("-" * 90)
    
    for regime in regimes:
        # Current market correlation
        rho = regime['rho']
        corr = np.eye(n_assets) + rho * (1 - np.eye(n_assets))
        
        # TRUE sigma_p (what the market actually has)
        pca_current = adapter.prepare_for_pca_pricing(asset_prices, asset_vols, corr, weights)
        sigma_p_true = pca_current['sigma_p_pca']
        
        # TRUE price
        true_price = black_scholes_call(S, K, T, r, sigma_p_true)
        
        # PCA price using STALE calibration (from calm market)
        pca_price = black_scholes_call(S, K, T, r, sigma_p_calibrated)
        pca_error = abs(pca_price - true_price) / true_price * 100
        
        # QRC: Generate factors based on current market
        qrc.reset_hidden_state()
        market_data = {
            'prices': S,
            'volatility': base_vol,
            'corr_change': (rho - rho_calm) / rho_calm,  # Relative change from calm
            'stress': regime['stress']
        }
        qrc_result = qrc.forward(market_data)
        qrc_factors = qrc_result.factors
        
        # QRC sigma_p (adapts to current correlation)
        qrc_adapted = adapter.prepare_for_qrc_pricing(
            asset_prices, asset_vols, corr, qrc_factors, weights
        )
        sigma_p_qrc = qrc_adapted['sigma_p_qrc']
        
        # QRC price
        qrc_price = black_scholes_call(S, K, T, r, sigma_p_qrc)
        qrc_error = abs(qrc_price - true_price) / true_price * 100
        
        print(f"{regime['name']:<14} {sigma_p_true:<10.4f} {sigma_p_calibrated:<10.4f} {sigma_p_qrc:<10.4f} ${true_price:<11.4f} {pca_error:<11.2f}% {qrc_error:<11.2f}%")
        
        results.append({
            'regime': regime['name'],
            'rho': rho,
            'sigma_true': sigma_p_true,
            'sigma_pca': sigma_p_calibrated,
            'sigma_qrc': sigma_p_qrc,
            'true_price': true_price,
            'pca_price': pca_price,
            'qrc_price': qrc_price,
            'pca_error': pca_error,
            'qrc_error': qrc_error,
            'qrc_factors': qrc_factors.tolist()
        })
    
    # =================================================================
    # PHASE 3: ANALYSIS
    # =================================================================
    print("\n" + "=" * 80)
    print("📊 PHASE 3: ANALYSIS")
    print("=" * 80)
    
    # Calculate improvements
    pca_errors = [r['pca_error'] for r in results]
    qrc_errors = [r['qrc_error'] for r in results]
    
    print(f"\nMean PCA Error: {np.mean(pca_errors):.2f}%")
    print(f"Mean QRC Error: {np.mean(qrc_errors):.2f}%")
    print(f"QRC Improvement: {np.mean(pca_errors) - np.mean(qrc_errors):.2f}%")
    
    # Stressed regime analysis
    calm_result = results[0]
    crisis_result = results[-1]
    
    print(f"\nCALM regime (ρ=0.3):")
    print(f"   PCA error: {calm_result['pca_error']:.2f}%")
    print(f"   QRC error: {calm_result['qrc_error']:.2f}%")
    
    print(f"\nCRISIS regime (ρ=0.8):")
    print(f"   PCA error: {crisis_result['pca_error']:.2f}%")
    print(f"   QRC error: {crisis_result['qrc_error']:.2f}%")
    print(f"   QRC Improvement: {crisis_result['pca_error'] - crisis_result['qrc_error']:.2f}%")
    
    # =================================================================
    # PHASE 4: VISUALIZATION
    # =================================================================
    print("\n" + "=" * 80)
    print("📈 PHASE 4: GENERATING PLOTS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    rhos = [r['rho'] for r in results]
    
    # Plot 1: Error comparison
    ax1 = axes[0, 0]
    ax1.plot(rhos, pca_errors, 'ro-', linewidth=2, markersize=8, label='PCA (stale)')
    ax1.plot(rhos, qrc_errors, 'go-', linewidth=2, markersize=8, label='QRC (adaptive)')
    ax1.set_xlabel('Correlation ρ', fontsize=12)
    ax1.set_ylabel('Pricing Error (%)', fontsize=12)
    ax1.set_title('Pricing Error vs Correlation Regime', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.25, 0.85)
    
    # Plot 2: σ_p comparison
    ax2 = axes[0, 1]
    sigma_trues = [r['sigma_true'] for r in results]
    sigma_pcas = [r['sigma_pca'] for r in results]
    sigma_qrcs = [r['sigma_qrc'] for r in results]
    
    ax2.plot(rhos, sigma_trues, 'b^-', linewidth=2, markersize=8, label='True σ_p')
    ax2.plot(rhos, sigma_pcas, 'r--', linewidth=2, label='PCA σ_p (stale)')
    ax2.plot(rhos, sigma_qrcs, 'gs-', linewidth=2, markersize=8, label='QRC σ_p')
    ax2.set_xlabel('Correlation ρ', fontsize=12)
    ax2.set_ylabel('Portfolio Volatility σ_p', fontsize=12)
    ax2.set_title('Volatility Tracking', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Price comparison
    ax3 = axes[1, 0]
    true_prices = [r['true_price'] for r in results]
    pca_prices = [r['pca_price'] for r in results]
    qrc_prices = [r['qrc_price'] for r in results]
    
    ax3.plot(rhos, true_prices, 'b^-', linewidth=2, markersize=8, label='True Price')
    ax3.plot(rhos, pca_prices, 'r--', linewidth=2, label='PCA Price (stale)')
    ax3.plot(rhos, qrc_prices, 'gs-', linewidth=2, markersize=8, label='QRC Price')
    ax3.set_xlabel('Correlation ρ', fontsize=12)
    ax3.set_ylabel('Option Price ($)', fontsize=12)
    ax3.set_title('Option Prices Across Regimes', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: QRC factor evolution
    ax4 = axes[1, 1]
    factor_names = ['f₁', 'f₂', 'f₃', 'f₄']
    for i in range(4):
        factors = [r['qrc_factors'][i] for r in results]
        ax4.plot(rhos, factors, 'o-', linewidth=2, markersize=6, label=factor_names[i])
    ax4.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Uniform (0.25)')
    ax4.set_xlabel('Correlation ρ', fontsize=12)
    ax4.set_ylabel('QRC Factor Value', fontsize=12)
    ax4.set_title('QRC Factor Adaptation', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP/experiments/qrc_regime_shift_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    
    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    print("\n" + "=" * 80)
    print("🎯 EXPERIMENT SUMMARY")
    print("=" * 80)
    
    improvement_stressed = crisis_result['pca_error'] - crisis_result['qrc_error']
    
    print(f"""
📌 KEY FINDINGS:

1. PCA Error Increases with Regime Shift
   - Calm (ρ=0.3):  {calm_result['pca_error']:.2f}%
   - Crisis (ρ=0.8): {crisis_result['pca_error']:.2f}%
   
2. QRC Adapts to Maintain Accuracy
   - Calm (ρ=0.3):  {calm_result['qrc_error']:.2f}%
   - Crisis (ρ=0.8): {crisis_result['qrc_error']:.2f}%

3. QRC Improvement in Crisis: {improvement_stressed:.2f}%

4. σ_p Tracking:
   - True σ_p in crisis:  {crisis_result['sigma_true']:.4f}
   - PCA σ_p (stale):     {crisis_result['sigma_pca']:.4f}
   - QRC σ_p (adaptive):  {crisis_result['sigma_qrc']:.4f}
""")
    
    if improvement_stressed > 5:
        print("✅ SIGNIFICANT QRC ADVANTAGE DEMONSTRATED")
    elif improvement_stressed > 0:
        print("✅ MODERATE QRC ADVANTAGE")
    else:
        print("⚠️  NO QRC ADVANTAGE (further tuning needed)")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
