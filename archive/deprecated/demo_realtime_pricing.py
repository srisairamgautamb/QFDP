#!/usr/bin/env python3
"""
Real-Time Quantum Option Pricing
=================================

Fetches live market data and prices options using quantum MLQAE.

Usage:
    python demo_realtime_pricing.py AAPL
    python demo_realtime_pricing.py TSLA --strike 250
    
Environment:
    ALPHAVANTAGE_API_KEY: Your Alpha Vantage API key
"""

import argparse
from qiskit import QuantumRegister

from qfdp_multiasset.market_data import AlphaVantageConnector
from qfdp_multiasset.state_prep import prepare_lognormal_asset
from qfdp_multiasset.oracles import apply_call_payoff_rotation
from qfdp_multiasset.mlqae import run_mlqae


def price_call_option_realtime(
    api_key: str,
    symbol: str,
    strike: float = None,
    maturity: float = 1.0,
    n_qubits: int = 8,
    shots: int = 2000,
    seed: int = 42
):
    """
    Price call option using real-time market data.
    
    Args:
        api_key: Alpha Vantage API key
        symbol: Stock ticker (e.g., 'AAPL')
        strike: Strike price (default: 105% of spot)
        maturity: Time to maturity in years
        n_qubits: Number of qubits for state prep
        shots: MLQAE measurement shots
        seed: Random seed
        
    Returns:
        dict with pricing results
    """
    print("=" * 70)
    print(f"QUANTUM OPTION PRICING: {symbol}")
    print("=" * 70)
    
    # Step 1: Fetch real-time market data
    print(f"\n[1/4] Fetching live market data...")
    connector = AlphaVantageConnector(api_key)
    
    S0, r, sigma = connector.estimate_parameters(symbol, risk_free_rate=0.045)
    
    # Default strike: 5% OTM
    if strike is None:
        strike = S0 * 1.05
    
    print(f"\n  Market Parameters:")
    print(f"    Spot price (S₀): ${S0:.2f}")
    print(f"    Strike (K): ${strike:.2f} ({(strike/S0-1)*100:+.1f}% moneyness)")
    print(f"    Volatility (σ): {sigma*100:.1f}% (historical 1Y)")
    print(f"    Risk-free rate (r): {r*100:.1f}%")
    print(f"    Maturity (T): {maturity:.1f} year")
    
    # Step 2: Prepare quantum state
    print(f"\n[2/4] Preparing quantum state...")
    circuit, prices = prepare_lognormal_asset(S0, r, sigma, maturity, n_qubits=n_qubits)
    print(f"  ✓ {n_qubits} qubits → {2**n_qubits} price points")
    print(f"  ✓ Price range: [${prices.min():.2f}, ${prices.max():.2f}]")
    
    # Step 3: Encode call payoff
    print(f"\n[3/4] Encoding call payoff oracle...")
    anc = QuantumRegister(1, 'ancilla')
    circuit.add_register(anc)
    scale = apply_call_payoff_rotation(circuit, circuit.qregs[0], anc[0], prices, strike)
    print(f"  ✓ Max payoff: ${scale:.2f}")
    print(f"  ✓ Circuit: {circuit.num_qubits} qubits, depth {circuit.depth()}")
    
    # Step 4: MLQAE pricing
    print(f"\n[4/4] Running MLQAE ({shots} shots)...")
    result = run_mlqae(
        circuit, anc[0], scale,
        grover_powers=[0],
        shots_per_power=shots,
        seed=seed
    )
    
    # Results
    print(f"\n{'=' * 70}")
    print(f"QUANTUM PRICING RESULTS")
    print(f"{'=' * 70}")
    print(f"  Call Option Price: ${result.price_estimate:.2f}")
    print(f"  95% Confidence Interval: [${result.confidence_interval[0]:.2f}, ${result.confidence_interval[1]:.2f}]")
    print(f"  Amplitude: {result.amplitude_estimate:.4f}")
    print(f"  Measurements: {result.total_shots:,}")
    
    # Classical comparison
    from qiskit.quantum_info import Statevector
    sv = Statevector(circuit)
    ancilla_idx = circuit.qubits.index(anc[0])
    prob_1 = sum(
        float((amp.conjugate() * amp).real)
        for i, amp in enumerate(sv.data)
        if (i >> ancilla_idx) & 1
    )
    classical_price = prob_1 * scale
    error = abs(result.price_estimate - classical_price)
    
    print(f"\n  Classical (exact): ${classical_price:.2f}")
    print(f"  Quantum error: ${error:.2f} ({error/classical_price*100:.1f}%)")
    
    # Intrinsic value
    intrinsic = max(S0 - strike, 0)
    time_value = result.price_estimate - intrinsic
    
    print(f"\n  Option Greeks (approx):")
    print(f"    Intrinsic value: ${intrinsic:.2f}")
    print(f"    Time value: ${time_value:.2f}")
    
    moneyness = "ITM" if S0 > strike else "OTM" if S0 < strike else "ATM"
    print(f"    Moneyness: {moneyness}")
    
    print(f"\n{'=' * 70}")
    
    return {
        'symbol': symbol,
        'spot': S0,
        'strike': strike,
        'volatility': sigma,
        'quantum_price': result.price_estimate,
        'classical_price': classical_price,
        'error': error,
        'ci': result.confidence_interval
    }


def main():
    parser = argparse.ArgumentParser(description='Real-time quantum option pricing')
    parser.add_argument('symbol', type=str, help='Stock ticker (e.g., AAPL, TSLA)')
    parser.add_argument('--strike', type=float, default=None, help='Strike price (default: 105%% spot)')
    parser.add_argument('--maturity', type=float, default=1.0, help='Time to maturity (years)')
    parser.add_argument('--qubits', type=int, default=8, help='Number of qubits (default: 8)')
    parser.add_argument('--shots', type=int, default=2000, help='MLQAE shots (default: 2000)')
    parser.add_argument('--api-key', type=str, default='V2MR7V040MOVAGC0', help='Alpha Vantage API key')
    
    args = parser.parse_args()
    
    result = price_call_option_realtime(
        api_key=args.api_key,
        symbol=args.symbol.upper(),
        strike=args.strike,
        maturity=args.maturity,
        n_qubits=args.qubits,
        shots=args.shots
    )
    
    print(f"\n✓ Pricing complete for {result['symbol']}")
    print(f"  Quantum: ${result['quantum_price']:.2f}")
    print(f"  Classical: ${result['classical_price']:.2f}\n")


if __name__ == "__main__":
    main()
