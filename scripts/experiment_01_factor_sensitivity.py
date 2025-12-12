#!/usr/bin/env python3
"""
Experiment 1: Factor Decomposition Sensitivity Analysis
========================================================

Sweeps K ∈ {1, 3, 5, 10} for N ∈ {5, 10, 20} with 25 random seeds per config.

Measures:
- Variance explained by top K factors
- Frobenius reconstruction error ||Σ - Σ_K||_F
- Portfolio variance error for equal-weight portfolio

Outputs:
- CSV results: outputs/experiments/experiment_01_results.csv
- Figures: Figure 1 (eigenvalue spectrum), Figure 2 (error vs K)
- Synthetic data: data/synthetic_correlations/*.npy (500 test matrices)

Runtime: ~10 minutes on standard workstation

Usage:
    python scripts/experiment_01_factor_sensitivity.py
    
    Optional flags:
    --n-seeds 25         (default: 25)
    --output-dir outputs/experiments
    --save-matrices      (save synthetic correlation matrices)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import time
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from qfdp_multiasset.sparse_copula import (
    FactorDecomposer,
    generate_synthetic_correlation_matrix,
    analyze_eigenvalue_decay
)


def run_single_decomposition(
    N: int,
    K: int,
    seed: int,
    noise_scale: float = 0.1
) -> Dict:
    """
    Run factor decomposition for single configuration.
    
    Returns dict with all metrics.
    """
    # Generate synthetic correlation matrix
    corr_matrix = generate_synthetic_correlation_matrix(N=N, K=K, noise_scale=noise_scale, seed=seed)
    
    # Decompose
    decomposer = FactorDecomposer()
    L, D, metrics = decomposer.fit(corr_matrix, K=K, validate=True)
    
    # Portfolio variance error (equal-weight)
    weights = np.ones(N) / N
    portfolio_results = decomposer.compute_portfolio_variance_error(L, D, weights)
    
    # Eigenvalue analysis
    eigenvalues = decomposer.get_eigenvalue_spectrum()
    
    return {
        'N': N,
        'K': K,
        'seed': seed,
        'variance_explained': metrics.variance_explained,
        'frobenius_error': metrics.frobenius_error,
        'max_element_error': metrics.max_element_error,
        'condition_number': metrics.condition_number,
        'eigenvalue_ratio': metrics.eigenvalue_ratio,
        'portfolio_variance_true': portfolio_results['variance_true'],
        'portfolio_variance_approx': portfolio_results['variance_approx'],
        'portfolio_absolute_error': portfolio_results['absolute_error'],
        'portfolio_relative_error': portfolio_results['relative_error'],
        'largest_eigenvalue': eigenvalues[0] if eigenvalues is not None else np.nan,
        'smallest_eigenvalue': eigenvalues[-1] if eigenvalues is not None else np.nan,
    }


def run_experiment(
    N_values: List[int] = [5, 10, 20],
    K_values: List[int] = [1, 3, 5, 10],
    n_seeds: int = 25,
    noise_scale: float = 0.1,
    output_dir: Path = Path("outputs/experiments"),
    save_matrices: bool = False
) -> pd.DataFrame:
    """
    Run full Experiment 1: factor decomposition sensitivity.
    
    Parameters
    ----------
    N_values : list of int
        Asset counts to test
    K_values : list of int
        Factor counts to test
    n_seeds : int
        Number of random seeds per configuration
    noise_scale : float
        Idiosyncratic noise scale for synthetic matrices
    output_dir : Path
        Output directory for results
    save_matrices : bool
        Whether to save generated correlation matrices
    
    Returns
    -------
    results_df : pd.DataFrame
        Complete results table
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("EXPERIMENT 1: FACTOR DECOMPOSITION SENSITIVITY")
    print("=" * 70)
    print(f"N values: {N_values}")
    print(f"K values: {K_values}")
    print(f"Seeds per config: {n_seeds}")
    print(f"Total runs: {len(N_values) * len(K_values) * n_seeds}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Generate all configurations
    configs = [
        (N, K, seed)
        for N in N_values
        for K in K_values
        for seed in range(n_seeds)
        if K <= N  # Skip invalid K > N
    ]
    
    print(f"\nRunning {len(configs)} decompositions...")
    
    # Run all decompositions
    results = []
    start_time = time.time()
    
    for N, K, seed in tqdm(configs, desc="Decompositions"):
        try:
            result = run_single_decomposition(N, K, seed, noise_scale)
            results.append(result)
            
            # Optionally save correlation matrix
            if save_matrices:
                matrix_dir = Path("data/synthetic_correlations")
                matrix_dir.mkdir(parents=True, exist_ok=True)
                corr = generate_synthetic_correlation_matrix(N, K, noise_scale, seed)
                np.save(matrix_dir / f"corr_N{N}_K{K}_seed{seed}.npy", corr)
        
        except Exception as e:
            print(f"\nWarning: Failed for N={N}, K={K}, seed={seed}: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = output_dir / "experiment_01_results.csv"
    results_df.to_csv(results_path, index=False)
    
    print(f"\n✓ Completed {len(results)} decompositions in {elapsed:.1f}s")
    print(f"✓ Results saved to: {results_path}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    for N in N_values:
        for K in [k for k in K_values if k <= N]:
            subset = results_df[(results_df['N'] == N) & (results_df['K'] == K)]
            if len(subset) > 0:
                print(f"\nN={N}, K={K}:")
                print(f"  Variance explained: {subset['variance_explained'].mean():.1%} ± {subset['variance_explained'].std():.1%}")
                print(f"  Frobenius error:    {subset['frobenius_error'].mean():.3f} ± {subset['frobenius_error'].std():.3f}")
                print(f"  Portfolio error:    {subset['portfolio_absolute_error'].mean():.4f} ± {subset['portfolio_absolute_error'].std():.4f}")
    
    return results_df


def generate_figures(results_df: pd.DataFrame, output_dir: Path):
    """
    Generate Figure 1 (eigenvalue spectrum) and Figure 2 (error vs K).
    
    Requires matplotlib. If not available, prints instructions.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("\nWarning: matplotlib not available. Skipping figure generation.")
        print("To generate figures, install: pip install matplotlib seaborn")
        return
    
    figures_dir = Path("paper/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Eigenvalue spectrum (for one example: N=20)
    print("\nGenerating Figure 1: Eigenvalue spectrum...")
    N_example = 20
    corr_example = generate_synthetic_correlation_matrix(N=N_example, K=3, seed=42)
    analysis = analyze_eigenvalue_decay(corr_example)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scree plot
    axes[0].plot(range(1, N_example + 1), analysis['eigenvalues'], 'o-', linewidth=2)
    axes[0].axvline(3, color='r', linestyle='--', alpha=0.7, label='K=3')
    axes[0].set_xlabel('Factor index', fontsize=12)
    axes[0].set_ylabel('Eigenvalue', fontsize=12)
    axes[0].set_title(f'Scree Plot (N={N_example})', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Variance explained
    axes[1].plot(range(1, N_example + 1), analysis['variance_explained'] * 100, 'o-', linewidth=2)
    axes[1].axhline(70, color='r', linestyle='--', alpha=0.7, label='70% threshold')
    axes[1].axvline(3, color='r', linestyle='--', alpha=0.7, label='K=3')
    axes[1].set_xlabel('Number of factors K', fontsize=12)
    axes[1].set_ylabel('Variance explained (%)', fontsize=12)
    axes[1].set_title(f'Cumulative Variance Explained (N={N_example})', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(figures_dir / "figure_01_eigenvalue_spectrum.png", dpi=300, bbox_inches='tight')
    fig.savefig(figures_dir / "figure_01_eigenvalue_spectrum.pdf", bbox_inches='tight')
    print(f"✓ Figure 1 saved: {figures_dir}/figure_01_eigenvalue_spectrum.*")
    plt.close()
    
    # Figure 2: Frobenius error vs K for N ∈ {5, 10, 20}
    print("Generating Figure 2: Frobenius error vs K...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Frobenius error
    for N in [5, 10, 20]:
        N_data = results_df[results_df['N'] == N]
        grouped = N_data.groupby('K')['frobenius_error'].agg(['mean', 'std'])
        axes[0].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                        marker='o', linewidth=2, capsize=5, label=f'N={N}')
    
    axes[0].set_xlabel('Number of factors K', fontsize=12)
    axes[0].set_ylabel('Frobenius error ||Σ - Σ_K||_F', fontsize=12)
    axes[0].set_title('Reconstruction Error vs K', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Panel B: Portfolio variance error
    for N in [5, 10, 20]:
        N_data = results_df[results_df['N'] == N]
        grouped = N_data.groupby('K')['portfolio_absolute_error'].agg(['mean', 'std'])
        axes[1].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                        marker='o', linewidth=2, capsize=5, label=f'N={N}')
    
    axes[1].set_xlabel('Number of factors K', fontsize=12)
    axes[1].set_ylabel('Portfolio variance error', fontsize=12)
    axes[1].set_title('Portfolio Error vs K (Equal-Weight)', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(figures_dir / "figure_02_error_vs_K.png", dpi=300, bbox_inches='tight')
    fig.savefig(figures_dir / "figure_02_error_vs_K.pdf", bbox_inches='tight')
    print(f"✓ Figure 2 saved: {figures_dir}/figure_02_error_vs_K.*")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Factor Decomposition Sensitivity")
    parser.add_argument('--n-seeds', type=int, default=25, help='Number of seeds per configuration')
    parser.add_argument('--output-dir', type=str, default='outputs/experiments', help='Output directory')
    parser.add_argument('--save-matrices', action='store_true', help='Save synthetic correlation matrices')
    parser.add_argument('--skip-figures', action='store_true', help='Skip figure generation')
    
    args = parser.parse_args()
    
    # Run experiment
    results_df = run_experiment(
        N_values=[5, 10, 20],
        K_values=[1, 3, 5, 10],
        n_seeds=args.n_seeds,
        output_dir=Path(args.output_dir),
        save_matrices=args.save_matrices
    )
    
    # Generate figures
    if not args.skip_figures:
        generate_figures(results_df, Path(args.output_dir))
    
    print("\n" + "=" * 70)
    print("✓ EXPERIMENT 1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
