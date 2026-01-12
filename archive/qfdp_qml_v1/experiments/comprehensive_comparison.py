"""
Comprehensive QML Validation Suite
===================================
Runs all 3 QML strategies and compares against PCA baseline.
"""

import numpy as np
import pandas as pd
import time
import sys
sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class StrategyResult:
    """Result from one strategy."""
    name: str
    price: float
    error: float
    training_time: float


def monte_carlo_reference(S0, sigma, corr, weights, K, T, r, n_sims=200000):
    """Correct Monte Carlo reference with correlated assets."""
    np.random.seed(42)
    N = len(S0)
    cov = np.outer(sigma, sigma) * corr
    L = np.linalg.cholesky(cov)
    Z = np.random.standard_normal((n_sims, N))
    Z_corr = Z @ L.T
    drift = (r - 0.5 * sigma**2) * T
    diffusion = np.sqrt(T) * Z_corr
    ST = S0 * np.exp(drift + diffusion)
    portfolio_T = ST @ weights
    payoffs = np.maximum(portfolio_T - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    stderr = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    return price, stderr


def strategy_pca_baseline(returns, S0, sigma, corr, weights, K, T, r):
    """Strategy 0: Classical PCA baseline."""
    start = time.time()
    
    eigenvalues = np.linalg.eigvalsh(corr)[::-1]
    n_factors = 3
    explained = eigenvalues[:n_factors].sum() / eigenvalues.sum()
    
    portfolio_value = np.sum(S0 * weights)
    cov = np.outer(sigma, sigma) * corr
    portfolio_vol = np.sqrt(weights @ cov @ weights)
    
    d1 = (np.log(portfolio_value/K) + (r + 0.5*portfolio_vol**2)*T) / (portfolio_vol*np.sqrt(T))
    d2 = d1 - portfolio_vol*np.sqrt(T)
    price = portfolio_value*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    return price, time.time() - start


def strategy_a_qae(returns, S0, sigma, corr, weights, K, T, r):
    """Strategy A: Quantum Autoencoder for factor extraction."""
    from qfdp_qml.hybrid_integration import QAE_FB_IQFT_Pricer
    
    start = time.time()
    pricer = QAE_FB_IQFT_Pricer(n_factors=min(3, len(S0)-1), n_layers=2)
    pricer.train(returns, max_iter=30)
    result = pricer.price_option(S0, sigma, corr, weights, K, T, r)
    
    return result.price_qae, time.time() - start


def strategy_b_volatility_qnn(returns, S0, sigma, corr, weights, K, T, r):
    """Strategy B: Volatility Surface QNN."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Statevector
    from qiskit_algorithms.optimizers import COBYLA
    
    start = time.time()
    
    portfolio_value = np.sum(S0 * weights)
    cov = np.outer(sigma, sigma) * corr
    base_vol = np.sqrt(weights @ cov @ weights)
    
    n_qubits = 3
    n_layers = 2
    n_params = n_qubits * n_layers * 2
    
    input_params = ParameterVector('x', 2)
    weight_params = ParameterVector('w', n_params)
    
    qc = QuantumCircuit(n_qubits)
    
    for i in range(n_qubits):
        qc.ry(input_params[0] * (i + 1), i)
        qc.rz(input_params[1] * (i + 1), i)
    
    for layer in range(n_layers):
        for i in range(n_qubits):
            idx = layer * n_qubits * 2 + i * 2
            qc.ry(weight_params[idx], i)
            qc.rz(weight_params[idx + 1], i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
    
    strikes = np.linspace(0.9, 1.1, 10) * portfolio_value
    maturities = np.array([0.5, 1.0, 1.5])
    
    training_data = []
    for k in strikes:
        for t in maturities:
            moneyness = k / portfolio_value
            vol = base_vol * (1 + 0.05 * (moneyness - 1)**2 - 0.02 * (moneyness - 1))
            training_data.append([moneyness, t, vol])
    
    training_data = np.array(training_data)
    X = training_data[:, :2]
    y = training_data[:, 2]
    
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y_norm = (y - y.mean()) / (y.std() + 1e-8)
    
    def forward(x, params):
        param_dict = {input_params[i]: x[i] for i in range(2)}
        param_dict.update({weight_params[i]: params[i] for i in range(n_params)})
        bound = qc.assign_parameters(param_dict)
        sv = Statevector(bound)
        probs = np.abs(sv.data)**2
        return probs[0] - probs[-1]
    
    def loss_fn(params):
        total = 0
        for i in range(len(X)):
            pred = forward(X_norm[i], params)
            total += (pred - y_norm[i])**2
        return total / len(X)
    
    init_params = np.random.randn(n_params) * 0.1
    optimizer = COBYLA(maxiter=30)
    result = optimizer.minimize(loss_fn, init_params)
    trained_params = result.x
    
    moneyness_test = K / portfolio_value
    x_test = np.array([(moneyness_test - X[:, 0].mean()) / (X[:, 0].std() + 1e-8),
                       (T - X[:, 1].mean()) / (X[:, 1].std() + 1e-8)])
    pred_norm = forward(x_test, trained_params)
    predicted_vol = pred_norm * y.std() + y.mean()
    predicted_vol = max(0.05, min(1.0, predicted_vol))
    
    d1 = (np.log(portfolio_value/K) + (r + 0.5*predicted_vol**2)*T) / (predicted_vol*np.sqrt(T))
    d2 = d1 - predicted_vol*np.sqrt(T)
    price = portfolio_value*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    return price, time.time() - start


def strategy_c_error_calibration(returns, S0, sigma, corr, weights, K, T, r, mc_ref):
    """Strategy C: Error Calibration QNN."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Statevector
    from qiskit_algorithms.optimizers import COBYLA
    
    start = time.time()
    
    portfolio_value = np.sum(S0 * weights)
    cov = np.outer(sigma, sigma) * corr
    portfolio_vol = np.sqrt(weights @ cov @ weights)
    
    d1 = (np.log(portfolio_value/K) + (r + 0.5*portfolio_vol**2)*T) / (portfolio_vol*np.sqrt(T))
    d2 = d1 - portfolio_vol*np.sqrt(T)
    base_price = portfolio_value*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    n_qubits = 4
    n_layers = 2
    n_params = n_qubits * n_layers
    
    input_params = ParameterVector('x', 4)
    weight_params = ParameterVector('w', n_params)
    
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(input_params[i], i)
    
    for layer in range(n_layers):
        for i in range(n_qubits):
            idx = layer * n_qubits + i
            qc.ry(weight_params[idx], i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
    
    training_scenarios = []
    for k_mult in [0.9, 0.95, 1.0, 1.05, 1.1]:
        for t in [0.5, 1.0, 1.5]:
            k_train = portfolio_value * k_mult
            mc_train, _ = monte_carlo_reference(S0, sigma, corr, weights, k_train, t, r, n_sims=50000)
            
            d1_t = (np.log(portfolio_value/k_train) + (r + 0.5*portfolio_vol**2)*t) / (portfolio_vol*np.sqrt(t))
            d2_t = d1_t - portfolio_vol*np.sqrt(t)
            bs_train = portfolio_value*norm.cdf(d1_t) - k_train*np.exp(-r*t)*norm.cdf(d2_t)
            
            error = mc_train - bs_train
            training_scenarios.append([k_mult, t, portfolio_vol, len(S0), error])
    
    training_scenarios = np.array(training_scenarios)
    X = training_scenarios[:, :4]
    y = training_scenarios[:, 4]
    
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y_norm = y / (np.abs(y).max() + 1e-8)
    
    def forward(x, params):
        param_dict = {input_params[i]: x[i] for i in range(4)}
        param_dict.update({weight_params[i]: params[i] for i in range(n_params)})
        bound = qc.assign_parameters(param_dict)
        sv = Statevector(bound)
        probs = np.abs(sv.data)**2
        return (probs[0] - probs[-1]) * 2
    
    def loss_fn(params):
        total = 0
        for i in range(len(X)):
            pred = forward(X_norm[i], params)
            total += (pred - y_norm[i])**2
        return total / len(X)
    
    init_params = np.random.randn(n_params) * 0.1
    optimizer = COBYLA(maxiter=30)
    result = optimizer.minimize(loss_fn, init_params)
    trained_params = result.x
    
    x_test = np.array([K/portfolio_value, T, portfolio_vol, len(S0)])
    x_test_norm = (x_test - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    pred_correction_norm = forward(x_test_norm, trained_params)
    pred_correction = pred_correction_norm * np.abs(y).max()
    
    corrected_price = base_price + pred_correction
    
    return corrected_price, time.time() - start


def run_comprehensive_comparison():
    """Run all 4 strategies and compare."""
    print("=" * 70)
    print("COMPREHENSIVE QML STRATEGY COMPARISON")
    print("=" * 70)
    
    returns = pd.read_csv('data/qml/log_returns.csv', index_col=0)
    prices = pd.read_csv('data/qml/raw_prices.csv', index_col=0)
    
    S0 = prices.iloc[-1].values
    weights = np.ones(5) / 5
    sigma = returns.std().values * np.sqrt(252)
    corr = returns.corr().values
    K = np.sum(S0 * weights)
    T = 1.0
    r = 0.05
    
    print(f"\nPortfolio: {list(prices.columns)}")
    print(f"Value: ${np.sum(S0 * weights):.2f}")
    print(f"Strike (ATM): ${K:.2f}")
    print(f"Volatility: {np.sqrt(weights @ (np.outer(sigma, sigma) * corr) @ weights):.1%}")
    
    print("\n" + "-" * 70)
    print("Computing Monte Carlo reference (200K paths)...")
    mc_price, mc_stderr = monte_carlo_reference(S0, sigma, corr, weights, K, T, r)
    print(f"MC Reference: ${mc_price:.4f} ± ${2*mc_stderr:.4f}")
    
    print("\n" + "-" * 70)
    print("Running Strategy 0: PCA Baseline...")
    pca_price, pca_time = strategy_pca_baseline(returns.values, S0, sigma, corr, weights, K, T, r)
    pca_error = abs(pca_price - mc_price) / mc_price * 100
    print(f"PCA Price: ${pca_price:.4f}, Error: {pca_error:.2f}%, Time: {pca_time:.2f}s")
    
    print("\n" + "-" * 70)
    print("Running Strategy A: Quantum Autoencoder...")
    qae_price, qae_time = strategy_a_qae(returns.values, S0, sigma, corr, weights, K, T, r)
    qae_error = abs(qae_price - mc_price) / mc_price * 100
    print(f"QAE Price: ${qae_price:.4f}, Error: {qae_error:.2f}%, Time: {qae_time:.2f}s")
    
    print("\n" + "-" * 70)
    print("Running Strategy B: Volatility QNN...")
    vol_price, vol_time = strategy_b_volatility_qnn(returns.values, S0, sigma, corr, weights, K, T, r)
    vol_error = abs(vol_price - mc_price) / mc_price * 100
    print(f"Vol QNN Price: ${vol_price:.4f}, Error: {vol_error:.2f}%, Time: {vol_time:.2f}s")
    
    print("\n" + "-" * 70)
    print("Running Strategy C: Error Calibration QNN...")
    err_price, err_time = strategy_c_error_calibration(returns.values, S0, sigma, corr, weights, K, T, r, mc_price)
    err_error = abs(err_price - mc_price) / mc_price * 100
    print(f"Error QNN Price: ${err_price:.4f}, Error: {err_error:.2f}%, Time: {err_time:.2f}s")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Price':<12} {'Error':<10} {'Time':<10}")
    print("-" * 57)
    print(f"{'Monte Carlo (ref)':<25} ${mc_price:<11.4f} {'-':<10} {'-':<10}")
    print(f"{'PCA Baseline':<25} ${pca_price:<11.4f} {pca_error:<9.2f}% {pca_time:<9.2f}s")
    print(f"{'A: QAE':<25} ${qae_price:<11.4f} {qae_error:<9.2f}% {qae_time:<9.2f}s")
    print(f"{'B: Volatility QNN':<25} ${vol_price:<11.4f} {vol_error:<9.2f}% {vol_time:<9.2f}s")
    print(f"{'C: Error Calibration':<25} ${err_price:<11.4f} {err_error:<9.2f}% {err_time:<9.2f}s")
    
    errors = {'PCA': pca_error, 'QAE': qae_error, 'Vol QNN': vol_error, 'Error QNN': err_error}
    best = min(errors, key=errors.get)
    
    print("\n" + "=" * 70)
    print(f"🏆 BEST STRATEGY: {best} with {errors[best]:.2f}% error")
    print("=" * 70)
    
    results = pd.DataFrame({
        'Strategy': ['MC Reference', 'PCA Baseline', 'QAE (A)', 'Vol QNN (B)', 'Error QNN (C)'],
        'Price': [mc_price, pca_price, qae_price, vol_price, err_price],
        'Error (%)': [0, pca_error, qae_error, vol_error, err_error],
        'Time (s)': [0, pca_time, qae_time, vol_time, err_time]
    })
    results.to_csv('qfdp_qml/results/strategy_comparison.csv', index=False)
    print("\n✅ Results saved to qfdp_qml/results/strategy_comparison.csv")
    
    return results


if __name__ == '__main__':
    run_comprehensive_comparison()
