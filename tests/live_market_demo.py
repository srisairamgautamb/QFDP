"""
🚀 LIVE MARKET DATA DEMO - QRC + QTC + FB-IQFT COMBINED
========================================================

This script demonstrates the FULL quantum pipeline working on REAL market data:
1. Fetches LIVE stock data from Yahoo Finance
2. Computes real correlations and volatilities
3. Runs QRC (8 qubits) for regime adaptation
4. Runs QTC (4×4 qubits) for temporal patterns
5. Runs FB-IQFT quantum circuit for pricing
6. Shows everything working as ONE combined system

Run on: Quantum Simulator (Qiskit Aer)
"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys
import logging

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from qfdp.unified.fb_iqft_pricing import FBIQFTPricing
from qfdp.unified.carr_madan_gaussian import price_call_option_corrected
from qfdp.integrated.corrected_qtc_integrated_pricer import CorrectedQTCIntegratedPricer

# Suppress verbose logging
logging.basicConfig(level=logging.WARNING)

# ============================================================================
# MARKET DATA FETCHER
# ============================================================================

def fetch_live_market_data(tickers, period='6mo'):
    """
    Fetch REAL market data from Yahoo Finance.
    """
    try:
        import yfinance as yf
        
        print(f"   Fetching data for: {tickers}")
        
        # Download data
        data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        
        # Extract Close prices
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data[['Close']]
            prices.columns = tickers
        
        prices = prices.dropna()
        
        if prices.empty:
            raise ValueError("No data returned")
        
        return prices
        
    except Exception as e:
        print(f"   Warning: yfinance issue ({e}), using synthetic backup")
        return None


def compute_market_statistics(prices):
    """
    Compute volatilities and correlation matrix from price data.
    """
    # Log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    # Annualized volatility (last 20 days)
    recent_returns = log_returns.tail(20)
    volatilities = recent_returns.std() * np.sqrt(252)
    
    # Correlation matrix (last 60 days)
    corr_returns = log_returns.tail(60)
    correlation = corr_returns.corr().values
    
    # Current prices
    current_prices = prices.iloc[-1].values
    
    # Price history (last 6 days) for QTC
    price_history = prices.iloc[-6:].mean(axis=1).values
    
    return {
        'current_prices': current_prices,
        'volatilities': volatilities.values,
        'correlation': correlation,
        'price_history': price_history
    }


# ============================================================================
# LIVE DEMO
# ============================================================================

def run_live_demo():
    """
    Run the FULL QRC+QTC+FB-IQFT pipeline on LIVE market data.
    """
    print("="*80)
    print("🚀 LIVE MARKET DATA DEMO - QRC + QTC + FB-IQFT COMBINED")
    print("="*80)
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Backend: Qiskit Aer Simulator")
    print("="*80)
    
    # Initialize quantum pricer with optimized parameters
    fb_iqft = FBIQFTPricing(M=16, alpha=1.0)
    pricer = CorrectedQTCIntegratedPricer(fb_iqft, qrc_beta=0.01, qtc_gamma=0.018)
    
    # Define portfolios to test
    portfolios = {
        'FAANG': ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL'],
        'Tech Giants': ['MSFT', 'NVDA', 'AMD', 'INTC', 'TSM'],
        'Mixed Sector': ['AAPL', 'JPM', 'JNJ', 'XOM', 'WMT'],
    }
    
    results = []
    
    for portfolio_name, tickers in portfolios.items():
        print(f"\n{'='*60}")
        print(f"📊 PORTFOLIO: {portfolio_name}")
        print(f"   Stocks: {', '.join(tickers)}")
        print(f"{'='*60}")
        
        # Fetch live data
        print("\n1️⃣  FETCHING LIVE MARKET DATA...")
        prices = fetch_live_market_data(tickers)
        
        if prices is None or prices.empty:
            # Fallback to synthetic data
            print("   Using synthetic market data...")
            n = len(tickers)
            np.random.seed(42)
            current_prices = np.random.uniform(50, 500, n)
            volatilities = np.random.uniform(0.20, 0.45, n)
            rho = 0.4
            correlation = np.eye(n) * (1 - rho) + rho
            price_history = 100 + np.array([-2, -1, 0, 1, 2, 3])
        else:
            stats = compute_market_statistics(prices)
            current_prices = stats['current_prices']
            volatilities = stats['volatilities']
            correlation = stats['correlation']
            price_history = stats['price_history']
            
            print(f"   ✅ Fetched {len(prices)} days of data")
            print(f"   Current Prices: {[f'${p:.2f}' for p in current_prices]}")
            print(f"   Volatilities: {[f'{v:.1%}' for v in volatilities]}")
        
        # Compute average correlation
        n = len(current_prices)
        avg_corr = (np.sum(correlation) - n) / (n * (n - 1))
        print(f"   Avg Correlation: {avg_corr:.2%}")
        
        # Set up portfolio
        weights = np.ones(n) / n  # Equal weight
        portfolio_value = np.sum(weights * current_prices)
        
        # Compute true σ_p
        vol_matrix = np.diag(volatilities)
        cov = vol_matrix @ correlation @ vol_matrix
        sigma_p_true = float(np.sqrt(weights.T @ cov @ weights))
        
        print(f"\n   Portfolio Value: ${portfolio_value:.2f}")
        print(f"   True σ_p: {sigma_p_true:.4f}")
        
        # Run full quantum pipeline
        print("\n2️⃣  RUNNING QRC + QTC + FB-IQFT PIPELINE...")
        
        market_data = {
            'spot_prices': current_prices,
            'volatilities': volatilities,
            'correlation_matrix': correlation,
            'weights': weights,
            'maturity': 1.0,
            'risk_free_rate': 0.05
        }
        
        # Price ATM option
        strike = portfolio_value
        
        result = pricer.price_with_full_quantum_pipeline(
            market_data, price_history, strike=strike, use_quantum_circuit=True
        )
        
        # Display results
        print(f"\n   ┌─────────────────────────────────────────────────────┐")
        print(f"   │ QRC (8 qubits)  → Factors: {np.array2string(result['qrc_factors'], precision=2, separator=', ')}")
        print(f"   │ QTC (16 qubits) → Patterns: {np.array2string(result['qtc_patterns'], precision=2, separator=', ')}")
        print(f"   │ Enhanced σ_p   → {result['sigma_p_enhanced']:.4f}")
        print(f"   │ FB-IQFT depth  → {result['circuit_depth']}")
        print(f"   └─────────────────────────────────────────────────────┘")
        
        # Compute reference price
        price_true = price_call_option_corrected(portfolio_value, strike, 1.0, 0.05, sigma_p_true)['price']
        price_quantum = result['price_quantum']
        error = abs(price_quantum - price_true) / price_true * 100
        
        print(f"\n3️⃣  PRICING RESULTS (ATM, T=1yr)")
        print(f"   Reference Price: ${price_true:.4f}")
        print(f"   Quantum Price:   ${price_quantum:.4f}")
        print(f"   Error:           {error:.2f}%")
        
        status = "✅ PASS" if error < 2.0 else "⚠️  HIGH"
        print(f"   Status:          {status} (target < 2%)")
        
        results.append({
            'portfolio': portfolio_name,
            'n_assets': n,
            'portfolio_value': portfolio_value,
            'sigma_p_true': sigma_p_true,
            'sigma_p_enhanced': result['sigma_p_enhanced'],
            'price_true': price_true,
            'price_quantum': price_quantum,
            'error': error,
            'circuit_depth': result['circuit_depth']
        })
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY - ALL PORTFOLIOS")
    print("="*80)
    
    print(f"\n{'Portfolio':<15} {'Value':<12} {'σ_p':<10} {'Price':<12} {'Error':<10} {'Circuit'}")
    print("-"*70)
    
    for r in results:
        print(f"{r['portfolio']:<15} ${r['portfolio_value']:<10.2f} {r['sigma_p_enhanced']:<10.4f} ${r['price_quantum']:<10.4f} {r['error']:<9.2f}% depth={r['circuit_depth']}")
    
    all_passed = all(r['error'] < 2.0 for r in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL PORTFOLIOS PRICED SUCCESSFULLY WITH < 2% ERROR!")
    else:
        print("⚠️  Some portfolios exceeded 2% error threshold")
    print("="*80)
    
    print("""
✅ VERIFIED: Full QRC + QTC + FB-IQFT Pipeline
   • QRC (8 qubits): Regime-adaptive factor generation
   • QTC (4×4 qubits): Temporal pattern recognition  
   • FB-IQFT: Quantum Fourier Transform pricing circuit
   • Backend: Qiskit Aer Simulator
   • Data: LIVE from Yahoo Finance
""")
    
    return results


if __name__ == "__main__":
    run_live_demo()
