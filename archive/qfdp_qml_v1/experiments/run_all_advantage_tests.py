"""
Quantum Advantage Discovery - Master Orchestrator
===================================================
Runs all quantum advantage tests and generates a comprehensive report.

Tests:
1. Window Discovery (5 scenarios)
2. Noise Robustness
3. Sample Complexity
4. Correlation Learning
"""

import numpy as np
import pandas as pd
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from datetime import datetime
import matplotlib.pyplot as plt


def run_all_tests() -> dict:
    """
    Run all quantum advantage tests and compile results.
    
    Returns a dictionary with all results and summary statistics.
    """
    print("=" * 80)
    print("🚀 QUANTUM ADVANTAGE RECOVERY PROTOCOL")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("This will systematically test where quantum methods outperform classical.")
    print("Tests: Window Discovery, Noise Robustness, Sample Complexity, Correlation Learning")
    print()
    
    results = {}
    output_dir = '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP/qfdp_qml/results/advantage_discovery'
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # TEST 1: Window Discovery
    # ========================================================================
    print("\n" + "=" * 80)
    print("📋 TEST 1/4: QUANTUM ADVANTAGE WINDOW DISCOVERY")
    print("=" * 80)
    
    try:
        from qfdp_qml.experiments.quantum_advantage_discovery import find_quantum_advantage_windows
        df_windows = find_quantum_advantage_windows()
        results['windows'] = df_windows
        
        # Extract wins
        wins = df_windows[df_windows['advantage'] > 0.1]
        results['windows_wins'] = len(wins)
        results['windows_best'] = wins.nlargest(1, 'advantage').to_dict('records')[0] if len(wins) > 0 else None
    except Exception as e:
        print(f"❌ Window discovery failed: {e}")
        results['windows'] = None
        results['windows_wins'] = 0
    
    # ========================================================================
    # TEST 2: Noise Robustness
    # ========================================================================
    print("\n" + "=" * 80)
    print("📋 TEST 2/4: NOISE ROBUSTNESS")
    print("=" * 80)
    
    try:
        from qfdp_qml.experiments.noise_robustness_test import test_noise_robustness
        df_noise = test_noise_robustness(n_trials=3)
        results['noise'] = df_noise
        
        # Compute degradation
        degradation_c = df_noise[df_noise['noise_level'] == 0.20]['error_classical'].values[0] - \
                       df_noise[df_noise['noise_level'] == 0.0]['error_classical'].values[0]
        degradation_q = df_noise[df_noise['noise_level'] == 0.20]['error_quantum'].values[0] - \
                       df_noise[df_noise['noise_level'] == 0.0]['error_quantum'].values[0]
        
        results['noise_degradation_classical'] = degradation_c
        results['noise_degradation_quantum'] = degradation_q
        results['noise_quantum_more_robust'] = degradation_q < degradation_c
    except Exception as e:
        print(f"❌ Noise robustness test failed: {e}")
        results['noise'] = None
    
    # ========================================================================
    # TEST 3: Sample Complexity
    # ========================================================================
    print("\n" + "=" * 80)
    print("📋 TEST 3/4: SAMPLE COMPLEXITY ADVANTAGE")
    print("=" * 80)
    
    try:
        from qfdp_qml.experiments.sample_complexity_advantage import test_sample_complexity
        df_samples = test_sample_complexity()
        results['samples'] = df_samples
        
        # Extract speedup stats
        results['samples_avg_speedup'] = df_samples['speedup'].mean()
        results['samples_max_speedup'] = df_samples['speedup'].max()
    except Exception as e:
        print(f"❌ Sample complexity test failed: {e}")
        results['samples'] = None
    
    # ========================================================================
    # TEST 4: Correlation Complexity
    # ========================================================================
    print("\n" + "=" * 80)
    print("📋 TEST 4/4: CORRELATION COMPLEXITY LEARNING")
    print("=" * 80)
    
    try:
        from qfdp_qml.experiments.correlation_complexity_test import test_correlation_complexity
        df_corr = test_correlation_complexity()
        results['correlation'] = df_corr
        
        # Extract wins
        quantum_wins = df_corr[df_corr['advantage'] > 0.1]
        results['correlation_quantum_wins'] = len(quantum_wins)
        results['correlation_scenarios'] = df_corr['scenario'].tolist()
    except Exception as e:
        print(f"❌ Correlation complexity test failed: {e}")
        results['correlation'] = None
    
    # ========================================================================
    # GENERATE COMPREHENSIVE REPORT
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE QUANTUM ADVANTAGE REPORT")
    print("=" * 80)
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("QUANTUM ADVANTAGE RECOVERY PROTOCOL - FINAL REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Summary stats
    report_lines.append("SUMMARY")
    report_lines.append("-" * 70)
    
    total_tests = 0
    quantum_advantages = 0
    
    # Window discovery summary
    if results.get('windows') is not None:
        n_scenarios = len(results['windows'])
        n_wins = results['windows_wins']
        total_tests += n_scenarios
        quantum_advantages += n_wins
        report_lines.append(f"Window Discovery: {n_wins}/{n_scenarios} quantum wins")
        if results['windows_best']:
            best = results['windows_best']
            report_lines.append(f"  Best: {best['window']} (advantage: +{best['advantage']:.2f}%)")
    
    # Noise robustness summary
    if results.get('noise') is not None:
        if results.get('noise_quantum_more_robust', False):
            quantum_advantages += 1
            report_lines.append("Noise Robustness: ✅ Quantum is more robust")
            report_lines.append(f"  Degradation: Classical +{results['noise_degradation_classical']:.2f}% vs Quantum +{results['noise_degradation_quantum']:.2f}%")
        else:
            report_lines.append("Noise Robustness: Classical is more robust")
        total_tests += 1
    
    # Sample complexity summary
    if results.get('samples') is not None:
        avg_speedup = results['samples_avg_speedup']
        max_speedup = results['samples_max_speedup']
        if avg_speedup > 5:
            quantum_advantages += 1
            report_lines.append(f"Sample Complexity: ✅ Quantum has {avg_speedup:.0f}x average speedup")
        else:
            report_lines.append(f"Sample Complexity: {avg_speedup:.0f}x speedup (not significant)")
        report_lines.append(f"  Max speedup: {max_speedup:.0f}x")
        total_tests += 1
    
    # Correlation learning summary
    if results.get('correlation') is not None:
        n_wins = results['correlation_quantum_wins']
        n_scenarios = len(results['correlation_scenarios'])
        total_tests += n_scenarios
        quantum_advantages += n_wins
        report_lines.append(f"Correlation Learning: {n_wins}/{n_scenarios} quantum wins")
    
    # Overall conclusion
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("CONCLUSION")
    report_lines.append("=" * 70)
    
    if quantum_advantages > total_tests * 0.3:
        report_lines.append(f"✅ QUANTUM ADVANTAGE FOUND in {quantum_advantages} scenarios!")
        report_lines.append("")
        report_lines.append("Best opportunities for quantum advantage:")
        if results.get('samples_avg_speedup', 0) > 5:
            report_lines.append("  1. SAMPLE COMPLEXITY: Use fewer shots for same accuracy")
        if results.get('noise_quantum_more_robust', False):
            report_lines.append("  2. NOISE ROBUSTNESS: Better performance under market uncertainty")
        if results.get('windows_best'):
            report_lines.append(f"  3. {results['windows_best']['window'].upper()}: Specific market conditions")
    else:
        report_lines.append("⚠️ Limited quantum advantage found in current tests.")
        report_lines.append("")
        report_lines.append("Recommendations:")
        report_lines.append("  1. Focus on sample complexity (theoretical advantage confirmed)")
        report_lines.append("  2. Train QAE on more diverse data")
        report_lines.append("  3. Test on real quantum hardware")
        report_lines.append("  4. Consider hybrid approaches (QNN for specific tasks)")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)
    
    # Print and save report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open(f'{output_dir}/quantum_advantage_report.txt', 'w') as f:
        f.write(report_text)
    print(f"\n✅ Report saved to {output_dir}/quantum_advantage_report.txt")
    
    # Generate summary visualization
    try:
        generate_summary_figure(results, output_dir)
    except Exception as e:
        print(f"⚠️ Could not generate summary figure: {e}")
    
    print(f"\n✅ All tests completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


def generate_summary_figure(results: dict, output_dir: str):
    """Generate a comprehensive summary visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Window Discovery (Top Left)
    ax1 = axes[0, 0]
    if results.get('windows') is not None:
        df = results['windows']
        windows = df['window'].unique()
        x = np.arange(len(windows))
        
        # Mean advantage per window
        mean_advantage = [df[df['window'] == w]['advantage'].mean() for w in windows]
        colors = ['green' if a > 0 else 'red' for a in mean_advantage]
        
        ax1.bar(x, mean_advantage, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='gray', linestyle='--')
        ax1.set_xticks(x)
        ax1.set_xticklabels([w[:15] for w in windows], rotation=45, ha='right')
        ax1.set_ylabel('Mean Quantum Advantage (%)')
        ax1.set_title('Window Discovery: Mean Advantage by Scenario', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
    else:
        ax1.text(0.5, 0.5, 'Window Discovery\nTest Failed', ha='center', va='center', fontsize=14)
        ax1.set_title('Window Discovery', fontweight='bold')
    
    # Plot 2: Noise Robustness (Top Right)
    ax2 = axes[0, 1]
    if results.get('noise') is not None:
        df = results['noise']
        ax2.plot(df['noise_level'] * 100, df['error_classical'], 'b-o', label='Classical', linewidth=2)
        ax2.plot(df['noise_level'] * 100, df['error_quantum'], 'r-s', label='Quantum', linewidth=2)
        ax2.set_xlabel('Noise Level (%)')
        ax2.set_ylabel('Pricing Error (%)')
        ax2.set_title('Noise Robustness', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Noise Robustness\nTest Failed', ha='center', va='center', fontsize=14)
        ax2.set_title('Noise Robustness', fontweight='bold')
    
    # Plot 3: Sample Complexity (Bottom Left)
    ax3 = axes[1, 0]
    if results.get('samples') is not None:
        df = results['samples']
        ax3.semilogy(df['target_error'], df['samples_classical'], 'b-o', label='Classical MC', linewidth=2)
        ax3.semilogy(df['target_error'], df['shots_quantum'], 'r-s', label='Quantum AE', linewidth=2)
        ax3.set_xlabel('Target Error (%)')
        ax3.set_ylabel('Samples / Shots Needed (log)')
        ax3.set_title('Sample Complexity: Quantum Speedup', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.invert_xaxis()
    else:
        ax3.text(0.5, 0.5, 'Sample Complexity\nTest Failed', ha='center', va='center', fontsize=14)
        ax3.set_title('Sample Complexity', fontweight='bold')
    
    # Plot 4: Correlation Learning (Bottom Right)
    ax4 = axes[1, 1]
    if results.get('correlation') is not None:
        df = results['correlation']
        x = np.arange(len(df))
        width = 0.35
        
        ax4.bar(x - width/2, df['error_classical'], width, label='Classical', color='blue', alpha=0.7)
        ax4.bar(x + width/2, df['error_quantum'], width, label='Quantum', color='red', alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels([s[:12] for s in df['scenario']], rotation=45, ha='right')
        ax4.set_ylabel('Pricing Error (%)')
        ax4.set_title('Correlation Learning', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'Correlation Learning\nTest Failed', ha='center', va='center', fontsize=14)
        ax4.set_title('Correlation Learning', fontweight='bold')
    
    plt.suptitle('Quantum Advantage Recovery Protocol - Summary', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'{output_dir}/quantum_advantage_summary.png', dpi=150)
    plt.close()
    print(f"✅ Summary figure saved to {output_dir}/quantum_advantage_summary.png")


if __name__ == '__main__':
    results = run_all_tests()
