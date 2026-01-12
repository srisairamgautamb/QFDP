#!/usr/bin/env python3
"""
IBM Quantum Live Demo: QRC + QTC + FB-IQFT Complete Pipeline
=============================================================

This script demonstrates the FULL quantum derivative pricing pipeline:
1. QRC (Quantum Recurrent Circuit) - Regime detection
2. QTC (Quantum Temporal Convolution) - Temporal patterns
3. FB-IQFT (Factor-Based Inverse QFT) - Shallow quantum pricing

Two modes:
- Phase 1: Synthetic data validation
- Phase 2: Real 5-year market data (yfinance)

Usage:
    python ibm_quantum_live_demo.py --mode synthetic
    python ibm_quantum_live_demo.py --mode real
    python ibm_quantum_live_demo.py --mode hardware  # After approval
"""

import numpy as np
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

print("=" * 80)
print("🔬 IBM QUANTUM LIVE DEMO: QRC + QTC + FB-IQFT")
print("=" * 80)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# =============================================================================
# IMPORTS
# =============================================================================

def import_modules():
    """Import all required modules with error handling."""
    modules = {}
    
    # Core Qiskit
    try:
        from qiskit import QuantumCircuit, transpile
        modules['qiskit'] = True
        print("✅ Qiskit core loaded")
    except ImportError as e:
        print(f"❌ Qiskit not found: {e}")
        return None
    
    # IBM Runtime
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        modules['ibm_runtime'] = True
        print("✅ IBM Runtime loaded")
    except ImportError:
        modules['ibm_runtime'] = False
        print("⚠️  IBM Runtime not installed (simulator only)")
    
    # QRC
    try:
        from qrc import QuantumRecurrentCircuit
        modules['qrc'] = QuantumRecurrentCircuit
        print("✅ QRC module loaded")
    except ImportError as e:
        print(f"❌ QRC not found: {e}")
        return None
    
    # QTC
    try:
        from qtc import QuantumTemporalConvolution
        modules['qtc'] = QuantumTemporalConvolution
        print("✅ QTC module loaded")
    except ImportError as e:
        print(f"❌ QTC not found: {e}")
        return None
    
    # FB-IQFT
    try:
        from qfdp.unified import FBIQFTPricing
        modules['fb_iqft'] = FBIQFTPricing
        print("✅ FB-IQFT module loaded")
    except ImportError as e:
        print(f"❌ FB-IQFT not found: {e}")
        return None
    
    # QRC Modulation
    try:
        from qfdp.unified.qrc_modulation import QRCModulation
        modules['modulation'] = QRCModulation
        print("✅ QRC Modulation loaded")
    except ImportError as e:
        print(f"⚠️  QRC Modulation not found: {e}")
        modules['modulation'] = None
    
    # Enhanced Factor Constructor
    try:
        from qfdp.unified.enhanced_factor_constructor import EnhancedFactorConstructor
        modules['factor_constructor'] = EnhancedFactorConstructor
        print("✅ Enhanced Factor Constructor loaded")
    except ImportError:
        modules['factor_constructor'] = None
        print("⚠️  Enhanced Factor Constructor not found")
    
    # Feature Fusion
    try:
        from qfdp.fusion.feature_fusion import FeatureFusion
        modules['fusion'] = FeatureFusion
        print("✅ Feature Fusion loaded")
    except ImportError:
        modules['fusion'] = None
        print("⚠️  Feature Fusion not found")
    
    return modules


# =============================================================================
# MONTE CARLO PRICER FOR COMPARISON
# =============================================================================

def monte_carlo_basket_option(
    asset_prices: np.ndarray,
    asset_vols: np.ndarray,
    correlation: np.ndarray,
    weights: np.ndarray,
    K: float,
    T: float,
    r: float = 0.05,
    n_paths: int = 100000,
    n_steps: int = 252
) -> Dict:
    """
    Price basket call option using Monte Carlo simulation.
    
    This provides a benchmark comparison for the quantum pricing.
    
    Args:
        asset_prices: Current asset prices
        asset_vols: Asset volatilities
        correlation: Correlation matrix
        weights: Portfolio weights
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        n_paths: Number of Monte Carlo paths
        n_steps: Time steps per path
    
    Returns:
        Dict with price, std_error, CI
    """
    import time
    start_time = time.time()
    
    n_assets = len(asset_prices)
    dt = T / n_steps
    
    # Cholesky decomposition for correlated paths
    L = np.linalg.cholesky(correlation)
    
    # Simulate paths
    np.random.seed(42)  # Reproducibility
    
    # Initialize asset paths
    S = np.tile(asset_prices, (n_paths, 1))  # (n_paths, n_assets)
    
    for _ in range(n_steps):
        # Independent normal random variables
        Z = np.random.randn(n_paths, n_assets)
        
        # Correlated random variables
        Z_corr = Z @ L.T
        
        # GBM update: S(t+dt) = S(t) * exp((r - 0.5*σ²)*dt + σ*√dt*Z)
        drift = (r - 0.5 * asset_vols**2) * dt
        diffusion = asset_vols * np.sqrt(dt) * Z_corr
        S = S * np.exp(drift + diffusion)
    
    # Basket value at maturity
    basket_T = S @ weights  # (n_paths,)
    
    # Call payoff
    payoffs = np.maximum(basket_T - K, 0)
    
    # Discounted expected payoff
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    elapsed = time.time() - start_time
    
    return {
        'price': float(price),
        'std_error': float(std_error),
        'ci_95': (float(price - 1.96 * std_error), float(price + 1.96 * std_error)),
        'n_paths': n_paths,
        'n_steps': n_steps,
        'elapsed_seconds': elapsed
    }


# =============================================================================
# PHASE 1: SYNTHETIC DATA TEST
# =============================================================================

def run_synthetic_demo(modules: Dict, backend='simulator') -> Dict:
    """
    Run complete QRC+QTC+FB-IQFT pipeline with synthetic data.
    
    This validates the entire quantum pipeline before using real data.
    """
    print("\n" + "=" * 80)
    print("📊 PHASE 1: SYNTHETIC DATA VALIDATION")
    print("=" * 80)
    
    # ====== STEP 1: Define Synthetic Portfolio ======
    print("\n🔹 STEP 1: Synthetic Portfolio Setup")
    print("-" * 40)
    
    n_assets = 4
    asset_names = ['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D']
    
    # Portfolio parameters
    asset_prices = np.array([100.0, 105.0, 98.0, 102.0])
    asset_vols = np.array([0.20, 0.25, 0.22, 0.23])
    weights = np.array([0.30, 0.25, 0.25, 0.20])
    
    # Correlation matrix (stressed regime: ρ ≈ 0.6)
    rho = 0.6
    correlation = np.eye(n_assets) + rho * (1 - np.eye(n_assets))
    
    # Price history for QTC (last 6 periods)
    price_history = np.array([99.0, 100.5, 99.8, 101.2, 100.0, 102.0])
    
    # Option parameters
    K = 100.0  # Strike
    T = 1.0    # Maturity (1 year)
    r = 0.05   # Risk-free rate
    
    print(f"  Assets: {asset_names}")
    print(f"  Prices: {asset_prices}")
    print(f"  Vols:   {asset_vols}")
    print(f"  Weights: {weights}")
    print(f"  Correlation (ρ): {rho}")
    print(f"  Strike K: ${K}")
    print(f"  Maturity T: {T} year")
    
    # ====== STEP 2: QRC - Regime Detection ======
    print("\n🔹 STEP 2: QRC (Quantum Recurrent Circuit) - Regime Detection")
    print("-" * 40)
    
    QRC = modules['qrc']
    qrc = QRC(n_factors=4)
    qrc.reset_hidden_state()
    
    # Prepare QRC input
    S_mean = np.mean(asset_prices)
    vol_mean = np.mean(asset_vols)
    stress = max(0, min(1, (rho - 0.3) * 2))  # 0 = calm, 1 = stressed
    
    qrc_input = {
        'prices': S_mean,
        'volatility': vol_mean,
        'corr_change': rho - 0.3,  # Change from baseline 0.3
        'stress': stress
    }
    
    print(f"  QRC Input: price={S_mean:.2f}, vol={vol_mean:.3f}, stress={stress:.2f}")
    
    # Run QRC
    qrc_result = qrc.forward(qrc_input)
    qrc_factors = qrc_result.factors
    
    print(f"\n  ✅ QRC Output:")
    print(f"     Factors: {np.round(qrc_factors, 4)}")
    print(f"     Sum: {np.sum(qrc_factors):.4f} (should be ≈1.0)")
    print(f"     Circuit depth: {qrc_result.circuit_depth}")
    print(f"     Hidden state norm: {np.linalg.norm(qrc_result.hidden_state):.4f}")
    
    # Interpret regime
    factor_concentration = np.max(qrc_factors) / np.min(qrc_factors)
    if factor_concentration > 3:
        regime = "STRESSED (high concentration)"
    elif factor_concentration > 2:
        regime = "TRANSITIONAL"
    else:
        regime = "CALM (uniform factors)"
    print(f"     Detected regime: {regime}")
    
    # ====== STEP 3: QTC - Temporal Patterns ======
    print("\n🔹 STEP 3: QTC (Quantum Temporal Convolution) - Temporal Patterns")
    print("-" * 40)
    
    QTC = modules['qtc']
    qtc = QTC(kernel_size=3, n_kernels=4, n_qubits=4, n_layers=3)
    
    print(f"  Price history: {price_history}")
    print(f"  Configuration: {qtc.n_kernels} kernels × {qtc.n_qubits} qubits")
    
    # Run QTC
    qtc_result = qtc.forward(price_history)
    qtc_patterns = qtc_result.patterns
    
    print(f"\n  ✅ QTC Output:")
    print(f"     Patterns: {np.round(qtc_patterns, 4)}")
    print(f"     Sum: {np.sum(qtc_patterns):.4f}")
    print(f"     Avg circuit depth: {qtc_result.circuit_depth}")
    print(f"     Kernel outputs: {len(qtc_result.kernel_outputs)}")
    
    # Interpret trend
    returns = np.diff(price_history) / price_history[:-1]
    momentum = np.mean(returns)
    if momentum > 0.01:
        trend = "BULLISH"
    elif momentum < -0.01:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"
    print(f"     Detected trend: {trend} (momentum={momentum:.4f})")
    
    # ====== STEP 4: Feature Fusion ======
    print("\n🔹 STEP 4: Feature Fusion (QRC + QTC)")
    print("-" * 40)
    
    if modules.get('fusion'):
        Fusion = modules['fusion']
        fusion = Fusion(method='weighted')
        fused = fusion.forward(qrc_factors, qtc_patterns)
    else:
        # Manual fusion: weighted average
        alpha = 0.6  # QRC weight
        fused = alpha * qrc_factors + (1 - alpha) * qtc_patterns
    
    print(f"  Fused features: {np.round(fused, 4)}")
    
    # ====== STEP 5: Enhanced Factor Construction ======
    print("\n🔹 STEP 5: Enhanced Factor Construction")
    print("-" * 40)
    
    # Compute base covariance
    vol_diag = np.diag(asset_vols)
    cov_base = vol_diag @ correlation @ vol_diag
    
    # PCA decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_base)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"  Base eigenvalues: {np.round(eigenvalues, 6)}")
    
    # Apply QRC+QTC modulation
    if modules.get('modulation'):
        Modulation = modules['modulation']
        modulator = Modulation(beta=0.5)
        modulated_eigenvalues, h_factors = modulator.apply_modulation(
            eigenvalues[:len(fused)], fused
        )
        print(f"  Modulated eigenvalues: {np.round(modulated_eigenvalues, 6)}")
        print(f"  Modulation factors h: {np.round(h_factors, 4)}")
    else:
        # Manual modulation
        beta = 0.5
        f_bar = 1.0 / len(fused)
        h_factors = 1.0 + beta * (fused / f_bar - 1.0)
        h_factors = np.clip(h_factors, 0.1, 2.0)
        modulated_eigenvalues = eigenvalues[:len(fused)] * h_factors
        print(f"  Modulation factors h: {np.round(h_factors, 4)}")
        print(f"  Modulated eigenvalues: {np.round(modulated_eigenvalues, 6)}")
    
    # ====== STEP 6: Portfolio Volatility ======
    print("\n🔹 STEP 6: Portfolio Volatility Computation")
    print("-" * 40)
    
    # Base σ_p (PCA)
    sigma_p_pca = np.sqrt(weights @ cov_base @ weights)
    
    # Enhanced σ_p (QRC+QTC modulated)
    Lambda_mod = np.diag(modulated_eigenvalues)
    Q_K = eigenvectors[:, :len(fused)]
    cov_enhanced = Q_K @ Lambda_mod @ Q_K.T
    sigma_p_enhanced = np.sqrt(weights @ cov_enhanced @ weights)
    
    print(f"  σ_p (PCA baseline):    {sigma_p_pca:.6f}")
    print(f"  σ_p (QRC+QTC enhanced): {sigma_p_enhanced:.6f}")
    print(f"  Change: {(sigma_p_enhanced - sigma_p_pca) / sigma_p_pca * 100:+.2f}%")
    
    # ====== STEP 7: FB-IQFT Quantum Pricing ======
    print("\n🔹 STEP 7: FB-IQFT Quantum Pricing Circuit")
    print("-" * 40)
    
    FBIQFT = modules['fb_iqft']
    pricer = FBIQFT(M=64, alpha=1.0, num_shots=8192)
    
    # Basket value
    B_0 = np.sum(weights * asset_prices)
    print(f"  Basket spot B₀: ${B_0:.2f}")
    print(f"  Grid size M: 64 (6 qubits)")
    print(f"  Running on: {backend}")
    
    # Price option
    try:
        result = pricer.price_option(
            asset_prices=asset_prices,
            asset_volatilities=asset_vols,
            correlation_matrix=correlation,
            portfolio_weights=weights,
            K=K,
            T=T,
            r=r,
            backend=backend
        )
        
        print(f"\n  ✅ FB-IQFT Results:")
        print(f"     Classical (BS) price: ${result['price_classical']:.4f}")
        print(f"     Quantum price:        ${result['price_quantum']:.4f}")
        print(f"     Error:                {result['error_percent']:.2f}%")
        print(f"     Circuit depth:        {result['circuit_depth']}")
        print(f"     Qubits used:          {result['num_qubits']}")
        print(f"     σ_p used:             {result['sigma_p']:.6f}")
        
        fb_iqft_success = True
        fb_iqft_result = result
        
    except Exception as e:
        print(f"\n  ❌ FB-IQFT Error: {e}")
        import traceback
        traceback.print_exc()
        fb_iqft_success = False
        fb_iqft_result = None
    
    # ====== SUMMARY ======
    print("\n" + "=" * 80)
    print("📋 SYNTHETIC DEMO SUMMARY")
    print("=" * 80)
    
    summary = {
        'phase': 'synthetic',
        'timestamp': datetime.now().isoformat(),
        'portfolio': {
            'n_assets': n_assets,
            'correlation': rho,
            'strike': K,
            'maturity': T
        },
        'qrc': {
            'factors': qrc_factors.tolist(),
            'circuit_depth': qrc_result.circuit_depth,
            'regime': regime
        },
        'qtc': {
            'patterns': qtc_patterns.tolist(),
            'circuit_depth': qtc_result.circuit_depth,
            'trend': trend
        },
        'volatility': {
            'sigma_p_pca': float(sigma_p_pca),
            'sigma_p_enhanced': float(sigma_p_enhanced),
            'change_pct': float((sigma_p_enhanced - sigma_p_pca) / sigma_p_pca * 100)
        },
        'fb_iqft': {
            'success': fb_iqft_success,
            'result': fb_iqft_result if fb_iqft_result else None
        },
        'backend': backend
    }
    
    if fb_iqft_success:
        print("✅ All quantum components executed successfully!")
        print(f"\n   QRC: {qrc_result.circuit_depth} depth | QTC: {qtc_result.circuit_depth} depth | FB-IQFT: {result['circuit_depth']} depth")
        print(f"   Error vs Black-Scholes: {result['error_percent']:.2f}%")
    else:
        print("⚠️  FB-IQFT had issues - check logs above")
    
    return summary


# =============================================================================
# PHASE 2: REAL MARKET DATA
# =============================================================================

def fetch_real_market_data(tickers: list, period: str = '5y') -> Optional[Dict]:
    """
    Fetch real market data from Yahoo Finance.
    
    Args:
        tickers: List of stock tickers
        period: Data period ('5y' for 5 years)
    
    Returns:
        Dict with prices, returns, volatilities, correlations
    """
    print("\n🌐 Fetching real market data from Yahoo Finance...")
    print(f"   Tickers: {tickers}")
    print(f"   Period: {period}")
    
    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed. Install with: pip install yfinance")
        return None
    
    try:
        # Download data
        data = yf.download(tickers, period=period, progress=False)
        
        if data.empty:
            print("❌ No data returned from Yahoo Finance")
            return None
        
        # Extract adjusted close prices
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            prices = data['Close']
        
        # Handle multi-level columns
        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices.droplevel(0, axis=1)
        
        # Drop any NaN rows
        prices = prices.dropna()
        
        print(f"   ✅ Downloaded {len(prices)} days of data")
        print(f"   Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
        
        # Compute returns
        returns = prices.pct_change().dropna()
        
        # Compute annualized volatilities
        volatilities = returns.std() * np.sqrt(252)
        
        # Compute correlation matrix
        correlation = returns.corr().values
        
        # Get last 6 days for QTC
        last_prices = prices.iloc[-6:].mean(axis=1).values
        
        # Current prices
        current_prices = prices.iloc[-1].values
        
        result = {
            'tickers': tickers,
            'current_prices': current_prices,
            'volatilities': volatilities.values,
            'correlation': correlation,
            'price_history': last_prices,
            'returns': returns.values,
            'n_days': len(prices),
            'date_range': (prices.index[0].strftime('%Y-%m-%d'), 
                          prices.index[-1].strftime('%Y-%m-%d'))
        }
        
        print(f"\n   Current prices: {np.round(current_prices, 2)}")
        print(f"   Annualized vols: {np.round(volatilities.values * 100, 2)}%")
        print(f"   Avg correlation: {np.mean(correlation[np.triu_indices(len(tickers), 1)]):.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_real_market_demo(modules: Dict, backend='simulator') -> Optional[Dict]:
    """
    Run complete pipeline with real 5-year market data.
    """
    print("\n" + "=" * 80)
    print("📈 PHASE 2: REAL MARKET DATA (5 Years)")
    print("=" * 80)
    
    # Import pandas
    global pd
    import pandas as pd
    
    # Define portfolio (major tech stocks)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    # Fetch data
    market_data = fetch_real_market_data(tickers, period='5y')
    
    if market_data is None:
        print("\n❌ Failed to fetch market data. Aborting real market demo.")
        return None
    
    # Portfolio setup
    n_assets = len(tickers)
    weights = np.ones(n_assets) / n_assets  # Equal weighted
    
    # Scale prices to ~100 for numerical stability
    scaling_factor = 100.0 / np.mean(market_data['current_prices'])
    asset_prices = market_data['current_prices'] * scaling_factor
    asset_vols = market_data['volatilities']
    correlation = market_data['correlation']
    
    # Price history for QTC
    price_history = market_data['price_history']
    price_history_scaled = price_history * scaling_factor
    
    # Option parameters
    K = 100.0  # Strike (ATM)
    T = 1.0    # 1 year maturity
    r = 0.05   # Risk-free rate
    
    print(f"\n📊 Portfolio Configuration:")
    print(f"   Tickers: {tickers}")
    print(f"   Scaled prices: {np.round(asset_prices, 2)}")
    print(f"   Volatilities: {np.round(asset_vols * 100, 2)}%")
    print(f"   Equal weights: {weights}")
    
    # ====== QRC ======
    print("\n🔹 Running QRC on real market data...")
    
    QRC = modules['qrc']
    qrc = QRC(n_factors=4)
    qrc.reset_hidden_state()
    
    # Compute stress from actual correlation
    avg_corr = np.mean(correlation[np.triu_indices(n_assets, 1)])
    stress = max(0, min(1, (avg_corr - 0.3) * 2))
    
    qrc_input = {
        'prices': np.mean(asset_prices),
        'volatility': np.mean(asset_vols),
        'corr_change': avg_corr - 0.3,
        'stress': stress
    }
    
    qrc_result = qrc.forward(qrc_input)
    qrc_factors = qrc_result.factors
    
    print(f"   QRC factors: {np.round(qrc_factors, 4)}")
    print(f"   Detected stress level: {stress:.2f}")
    
    # ====== QTC ======
    print("\n🔹 Running QTC on real price history...")
    
    QTC = modules['qtc']
    qtc = QTC(kernel_size=3, n_kernels=4, n_qubits=4, n_layers=3)
    
    qtc_result = qtc.forward(price_history_scaled)
    qtc_patterns = qtc_result.patterns
    
    print(f"   QTC patterns: {np.round(qtc_patterns, 4)}")
    
    # ====== Enhanced σ_p ======
    print("\n🔹 Computing enhanced portfolio volatility...")
    
    # Fusion
    alpha = 0.6
    fused = alpha * qrc_factors + (1 - alpha) * qtc_patterns
    
    # Base covariance
    vol_diag = np.diag(asset_vols)
    cov_base = vol_diag @ correlation @ vol_diag
    sigma_p_pca = np.sqrt(weights @ cov_base @ weights)
    
    # Modulated
    eigenvalues, eigenvectors = np.linalg.eigh(cov_base)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    beta = 0.5
    f_bar = 1.0 / len(fused)
    h_factors = np.clip(1.0 + beta * (fused / f_bar - 1.0), 0.1, 2.0)
    modulated_eigenvalues = eigenvalues[:len(fused)] * h_factors
    
    Lambda_mod = np.diag(modulated_eigenvalues)
    Q_K = eigenvectors[:, :len(fused)]
    cov_enhanced = Q_K @ Lambda_mod @ Q_K.T
    sigma_p_enhanced = np.sqrt(weights @ cov_enhanced @ weights)
    
    print(f"   σ_p (PCA):     {sigma_p_pca:.6f} ({sigma_p_pca*100:.2f}%)")
    print(f"   σ_p (enhanced): {sigma_p_enhanced:.6f} ({sigma_p_enhanced*100:.2f}%)")
    
    # ====== FB-IQFT ======
    print("\n🔹 Running FB-IQFT quantum circuit...")
    
    FBIQFT = modules['fb_iqft']
    pricer = FBIQFT(M=64, alpha=1.0, num_shots=8192)
    
    B_0 = np.sum(weights * asset_prices)
    print(f"   Basket value B₀: ${B_0:.2f}")
    
    try:
        result = pricer.price_option(
            asset_prices=asset_prices,
            asset_volatilities=asset_vols,
            correlation_matrix=correlation,
            portfolio_weights=weights,
            K=K,
            T=T,
            r=r,
            backend=backend
        )
        
        print(f"\n   ✅ FB-IQFT Results (Real Data):")
        print(f"      Classical price: ${result['price_classical']:.4f}")
        print(f"      Quantum price:   ${result['price_quantum']:.4f}")
        print(f"      Error:           {result['error_percent']:.2f}%")
        print(f"      Circuit depth:   {result['circuit_depth']}")
        
        success = True
        
    except Exception as e:
        print(f"\n   ❌ FB-IQFT Error: {e}")
        result = None
        success = False
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 REAL MARKET DATA SUMMARY")
    print("=" * 80)
    
    summary = {
        'phase': 'real_market',
        'timestamp': datetime.now().isoformat(),
        'tickers': tickers,
        'date_range': market_data['date_range'],
        'n_days': market_data['n_days'],
        'avg_correlation': float(avg_corr),
        'volatility': {
            'sigma_p_pca': float(sigma_p_pca),
            'sigma_p_enhanced': float(sigma_p_enhanced)
        },
        'qrc_factors': qrc_factors.tolist(),
        'qtc_patterns': qtc_patterns.tolist(),
        'fb_iqft_success': success,
        'fb_iqft_result': result,
        'backend': backend
    }
    
    if success:
        print(f"✅ Real market demo completed successfully!")
        print(f"   Portfolio: {tickers}")
        print(f"   Data: {market_data['n_days']} days ({market_data['date_range'][0]} to {market_data['date_range'][1]})")
        print(f"   Pricing error: {result['error_percent']:.2f}%")
    
    return summary


# =============================================================================
# PHASE 3: IBM QUANTUM HARDWARE
# =============================================================================

def setup_ibm_quantum(api_token: str) -> Optional[object]:
    """
    Set up IBM Quantum connection.
    """
    print("\n🔌 Setting up IBM Quantum connection...")
    
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        # Check if already saved
        try:
            service = QiskitRuntimeService()
            print("✅ Using saved IBM Quantum credentials")
        except:
            # Save new credentials
            print("📝 Saving new IBM Quantum credentials...")
            QiskitRuntimeService.save_account(
                channel="ibm_quantum",
                token=api_token,
                overwrite=True
            )
            service = QiskitRuntimeService()
            print("✅ Credentials saved and connected")
        
        # List available backends
        backends = service.backends(simulator=False, operational=True)
        print(f"\n📡 Available quantum backends:")
        for b in backends[:5]:
            print(f"   - {b.name}: {b.num_qubits} qubits")
        
        # Select best backend
        preferred = ['ibm_torino', 'ibm_kyiv', 'ibm_osaka', 'ibm_sherbrooke']
        backend = None
        for name in preferred:
            try:
                backend = service.backend(name)
                print(f"\n✅ Selected: {name}")
                break
            except:
                continue
        
        if backend is None and backends:
            backend = backends[0]
            print(f"\n✅ Selected: {backend.name} (first available)")
        
        return backend
        
    except ImportError:
        print("❌ qiskit-ibm-runtime not installed")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def run_hardware_demo(modules: Dict, api_token: str) -> Optional[Dict]:
    """
    Run on real IBM Quantum hardware.
    """
    print("\n" + "=" * 80)
    print("🖥️  PHASE 3: IBM QUANTUM HARDWARE")
    print("=" * 80)
    
    # Setup connection
    backend = setup_ibm_quantum(api_token)
    
    if backend is None:
        print("❌ Could not connect to IBM Quantum")
        return None
    
    print(f"\n⏱️  Hardware execution may take 5-20 minutes...")
    print("   (Queue time + execution + calibration)")
    
    # Run with real hardware
    # First synthetic to ensure circuit works
    print("\n📊 Running synthetic test on hardware...")
    result = run_synthetic_demo(modules, backend=backend)
    
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='IBM Quantum Live Demo')
    parser.add_argument('--mode', type=str, default='synthetic',
                       choices=['synthetic', 'real', 'hardware', 'all'],
                       help='Demo mode')
    parser.add_argument('--api-token', type=str, 
                       default='71ZGWcl3-sDX9RlhN9NCvhcGxg0FMRNF6eVhotgnxobr',
                       help='IBM Quantum API token')
    
    args = parser.parse_args()
    
    # Import modules
    modules = import_modules()
    if modules is None:
        print("\n❌ Failed to import required modules")
        sys.exit(1)
    
    results = {}
    
    # Phase 1: Synthetic
    if args.mode in ['synthetic', 'all']:
        results['synthetic'] = run_synthetic_demo(modules, backend='simulator')
    
    # Phase 2: Real Market
    if args.mode in ['real', 'all']:
        results['real'] = run_real_market_demo(modules, backend='simulator')
    
    # Phase 3: Hardware (requires approval)
    if args.mode == 'hardware':
        results['hardware'] = run_hardware_demo(modules, args.api_token)
    
    # Save results
    output_file = Path('/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP/results') / f'ibm_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_file.parent.mkdir(exist_ok=True)
    
    # Convert numpy arrays for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert(results), f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("🏁 DEMO COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
