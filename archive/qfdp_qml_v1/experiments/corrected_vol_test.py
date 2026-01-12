"""Corrected Vol QNN validation with independent seeds."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')
from scipy.stats import norm
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA

returns = pd.read_csv('data/qml/log_returns.csv', index_col=0)
prices = pd.read_csv('data/qml/raw_prices.csv', index_col=0)

S0 = prices.iloc[-1].values
weights = np.ones(5) / 5
sigma = returns.std().values * np.sqrt(252)
corr = returns.corr().values
K = np.sum(S0 * weights)
T = 1.0
r = 0.05

portfolio_value = np.sum(S0 * weights)
cov = np.outer(sigma, sigma) * corr
base_vol = np.sqrt(weights @ cov @ weights)

print("=" * 70)
print("CORRECTED VOL QNN TEST (Independent Seeds)")
print("=" * 70)

vol_errors = []
pca_errors = []

for seed in range(10):
    np.random.seed(seed)
    
    n_qubits = 3
    n_params = n_qubits * 2 * 2
    
    input_params = ParameterVector('x', 2)
    weight_params = ParameterVector('w', n_params)
    
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(input_params[0] * (i + 1), i)
        qc.rz(input_params[1] * (i + 1), i)
    for layer in range(2):
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
    vol_price = portfolio_value*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    d1_pca = (np.log(portfolio_value/K) + (r + 0.5*base_vol**2)*T) / (base_vol*np.sqrt(T))
    d2_pca = d1_pca - base_vol*np.sqrt(T)
    pca_price = portfolio_value*norm.cdf(d1_pca) - K*np.exp(-r*T)*norm.cdf(d2_pca)
    
    np.random.seed(seed + 1000)
    L = np.linalg.cholesky(cov)
    n_sims = 100000
    Z = np.random.standard_normal((n_sims, 5))
    Z_corr = Z @ L.T
    drift = (r - 0.5 * sigma**2) * T
    diffusion = np.sqrt(T) * Z_corr
    ST = S0 * np.exp(drift + diffusion)
    portfolio_T = ST @ weights
    payoffs = np.maximum(portfolio_T - K, 0)
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    
    vol_error = abs(vol_price - mc_price) / mc_price * 100
    pca_error = abs(pca_price - mc_price) / mc_price * 100
    
    vol_errors.append(vol_error)
    pca_errors.append(pca_error)
    
    print(f'Seed {seed}: Vol QNN={vol_error:.2f}%, PCA={pca_error:.2f}%')

print()
print("=" * 70)
print("CORRECTED RESULTS")
print("=" * 70)
print(f'Vol QNN: Mean={np.mean(vol_errors):.3f}% +/- Std={np.std(vol_errors):.3f}%')
print(f'PCA:     Mean={np.mean(pca_errors):.3f}% +/- Std={np.std(pca_errors):.3f}%')
print()

vol_wins = sum(1 for v, p in zip(vol_errors, pca_errors) if v < p)
print(f'Vol QNN wins: {vol_wins}/10')

if np.mean(vol_errors) < np.mean(pca_errors):
    improvement = (np.mean(pca_errors) - np.mean(vol_errors)) / np.mean(pca_errors) * 100
    print(f'Vol QNN beats PCA by {improvement:.1f}%')
else:
    print('PCA beats Vol QNN')
