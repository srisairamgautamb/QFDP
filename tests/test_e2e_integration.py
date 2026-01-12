"""
End-to-End Integration Test: QRC + FB-IQFT
==========================================

Uses the FIXED QRCIntegratedPricer that actually injects
QRC sigma_p into the pricing pipeline.
"""

import numpy as np
import sys

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.integrated_pricer import QRCIntegratedPricer


def test_e2e_integration():
    """
    Full end-to-end test with fixed integration.
    """
    
    print("=" * 80)
    print("🧪 E2E INTEGRATION TEST: QRC + FB-IQFT (FIXED)")
    print("=" * 80)
    
    pricer = QRCIntegratedPricer(n_factors=4, M=16, beta=0.1)
    print("✅ Pricer initialized")
    
    n_assets = 5
    asset_prices = np.full(n_assets, 100.0)
    asset_vols = np.full(n_assets, 0.20)
    weights = np.ones(n_assets) / n_assets
    
    test_cases = [
        {'name': 'Low Corr (Calm)', 'corr': 0.3, 'stress': 0.2},
        {'name': 'Med Corr', 'corr': 0.5, 'stress': 0.4},
        {'name': 'High Corr (Stress)', 'corr': 0.8, 'stress': 0.8},
    ]
    
    print("\n" + "-" * 90)
    print(f"{'Scenario':<20} {'QRC Price':<12} {'PCA Price':<12} {'Diff %':<10} {'σ_p QRC':<10} {'σ_p PCA':<10}")
    print("-" * 90)
    
    results = []
    
    for case in test_cases:
        corr = np.eye(n_assets) + case['corr'] * (1 - np.eye(n_assets))
        market_data = {
            'prices': 100.0,
            'volatility': 0.20,
            'corr_change': case['stress'] * 0.5,
            'stress': case['stress']
        }
        
        comparison = pricer.compare_qrc_vs_pca(
            market_data, asset_prices, asset_vols,
            corr, weights, strike=100.0
        )
        
        qrc = comparison['qrc']
        pca = comparison['pca']
        diff_pct = comparison['price_diff_pct']
        
        print(f"{case['name']:<20} ${qrc['price_quantum']:<11.4f} ${pca['price_quantum']:<11.4f} "
              f"{diff_pct:<9.2f}% {qrc['sigma_p_qrc']:<10.4f} {pca['sigma_p_pca']:<10.4f}")
        
        results.append({
            'scenario': case['name'],
            'qrc_price': qrc['price_quantum'],
            'pca_price': pca['price_quantum'],
            'diff_pct': diff_pct,
            'sigma_change': qrc['sigma_p_change_pct']
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    avg_diff = np.mean([r['diff_pct'] for r in results])
    print(f"\nAverage Price Difference: {avg_diff:.2f}%")
    
    if avg_diff > 0.5:
        print("✅ QRC prices DIFFER from PCA (integration working!)")
    else:
        print("⚠️  QRC prices too similar to PCA")
    
    # Analyze regime impact
    calm = results[0]
    stressed = results[2]
    
    print(f"\nRegime Impact:")
    print(f"  Calm:     QRC ${calm['qrc_price']:.4f} vs PCA ${calm['pca_price']:.4f} (diff: {calm['diff_pct']:.2f}%)")
    print(f"  Stressed: QRC ${stressed['qrc_price']:.4f} vs PCA ${stressed['pca_price']:.4f} (diff: {stressed['diff_pct']:.2f}%)")
    
    print("\n" + "=" * 80)
    print("✅ E2E INTEGRATION TEST COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    test_e2e_integration()
