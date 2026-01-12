#!/usr/bin/env python3
"""
QADP - REAL MARKET DATA ON IBM HARDWARE
========================================
"""

import numpy as np
import sys
import time
from datetime import datetime

sys.path.insert(0, '/Volumes/Hippocampus/Quant_Finance/QFDP/QFDP')

print("=" * 90)
print("  QADP - REAL MARKET DATA ON IBM QUANTUM HARDWARE")
print("=" * 90)
print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)

# Fetch real market data
import yfinance as yf
import pandas as pd

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
print(f"\n📈 Fetching maximum data for {tickers}...")

data = yf.download(tickers, period='10y', progress=False)
prices = data['Close'].dropna()

print(f"   Downloaded: {len(prices)} trading days")
print(f"   Period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")

# Compute statistics
returns = prices.pct_change().dropna()
vols = returns.std() * np.sqrt(252)
corr = returns.corr().values
current_prices = prices.iloc[-1].values

# Scale prices
scale = 100.0 / np.mean(current_prices)
asset_prices = current_prices * scale
asset_vols = vols.values
weights = np.ones(4) / 4
K, T, r = 100.0, 1.0, 0.05

# Price history for QTC
price_history = prices.iloc[-6:].mean(axis=1).values * scale

avg_corr = np.mean(corr[np.triu_indices(4, 1)])
regime = 'CALM' if avg_corr < 0.35 else ('MODERATE' if avg_corr < 0.55 else ('ELEVATED' if avg_corr < 0.70 else 'STRESSED'))

print(f"\n📊 Portfolio Statistics:")
print(f"   Current prices: ${np.round(current_prices, 2)}")
print(f"   Annualized vols: {np.round(vols.values * 100, 2)}%")
print(f"   Avg correlation: {avg_corr:.3f} ({regime})")

# Import components
from qrc import QuantumRecurrentCircuit
from qtc import QuantumTemporalConvolution
from qfdp.unified import FBIQFTPricing
from qfdp.unified.qrc_modulation import QRCModulation
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# Connect to IBM
print("\n🔌 Connecting to IBM Quantum...")
service = QiskitRuntimeService()
hw_backend = service.backend('ibm_torino')
print(f"   ✅ Connected: {hw_backend.name}")

# ===== RUN QADP =====

print("\n" + "=" * 90)
print("  RUNNING QADP ON IBM HARDWARE")
print("=" * 90)

# ----- QRC -----
print("\n🔹 QRC on IBM Hardware...")
qrc = QuantumRecurrentCircuit(n_factors=4)
qrc.reset_hidden_state()

stress = max(0, min(1, (avg_corr - 0.3) * 2))
qrc_input = {'prices': np.mean(asset_prices), 'volatility': np.mean(asset_vols),
             'corr_change': avg_corr - 0.3, 'stress': stress}

# Simulator baseline
qrc_result_sim = qrc.forward(qrc_input)
qrc_factors_sim = qrc_result_sim.factors
qrc_circuit = qrc.h_quantum

# Hardware
qrc_transpiled = transpile(qrc_circuit, hw_backend, optimization_level=3)
sampler = SamplerV2(hw_backend)
job = sampler.run([qrc_transpiled], shots=4096)
print(f"   Job: {job.job_id()}")
result = job.result()
qrc_counts = result[0].data.meas.get_counts()
qrc_factors_hw = qrc._extract_factors(qrc_counts)
print(f"   Simulator: {np.round(qrc_factors_sim, 4)}")
print(f"   Hardware:  {np.round(qrc_factors_hw, 4)}")

# ----- QTC -----
print("\n🔹 QTC on IBM Hardware...")
qtc = QuantumTemporalConvolution(kernel_size=3, n_kernels=4, n_qubits=4, n_layers=3)

# Simulator baseline
qtc_result_sim = qtc.forward(price_history)
qtc_patterns_sim = qtc_result_sim.patterns

# Hardware
qtc_circuits = qtc.build_circuits(price_history)
qtc_kernel_counts = []

for i, circ in enumerate(qtc_circuits):
    circ_t = transpile(circ, hw_backend, optimization_level=3)
    sampler = SamplerV2(hw_backend)
    job = sampler.run([circ_t], shots=2048)
    result = job.result()
    counts = result[0].data.meas.get_counts()
    qtc_kernel_counts.append(counts)
    print(f"   Kernel {i}: {job.job_id()}")

qtc_result_hw = qtc.forward_with_counts(qtc_kernel_counts)
qtc_patterns_hw = qtc_result_hw.patterns
print(f"   Simulator: {np.round(qtc_patterns_sim, 4)}")
print(f"   Hardware:  {np.round(qtc_patterns_hw, 4)}")

# ----- Feature Fusion -----
print("\n🔹 Feature Fusion...")
alpha = 0.6
fused_sim = alpha * qrc_factors_sim + (1 - alpha) * qtc_patterns_sim
fused_hw = alpha * qrc_factors_hw + (1 - alpha) * qtc_patterns_hw
print(f"   Simulator: {np.round(fused_sim, 4)}")
print(f"   Hardware:  {np.round(fused_hw, 4)}")

# ----- Enhanced σ_p -----
print("\n🔹 Enhanced σ_p...")
vol_diag = np.diag(asset_vols)
cov_base = vol_diag @ corr @ vol_diag
sigma_p_base = np.sqrt(weights @ cov_base @ weights)

eigenvalues, eigenvectors = np.linalg.eigh(cov_base)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

modulator = QRCModulation(beta=0.5)
n_f = min(len(fused_hw), len(eigenvalues))
mod_eigen, h_factors = modulator.apply_modulation(eigenvalues[:n_f], fused_hw[:n_f])
Lambda_mod = np.diag(mod_eigen)
Q_K = eigenvectors[:, :n_f]
cov_enhanced = Q_K @ Lambda_mod @ Q_K.T
sigma_p_hw = np.sqrt(weights @ cov_enhanced @ weights)

print(f"   σ_p (base): {sigma_p_base:.4f} ({sigma_p_base*100:.2f}%)")
print(f"   σ_p (HW):   {sigma_p_hw:.4f} ({sigma_p_hw*100:.2f}%)")
print(f"   Change: {(sigma_p_hw - sigma_p_base) / sigma_p_base * 100:+.2f}%")

# ----- FB-IQFT -----
print("\n🔹 FB-IQFT on IBM Hardware...")
pricer = FBIQFTPricing(M=64, alpha=1.0, num_shots=8192)

# Simulator
result_sim = pricer.price_option(
    asset_prices=asset_prices, asset_volatilities=asset_vols,
    correlation_matrix=corr, portfolio_weights=weights,
    K=K, T=T, r=r, backend='simulator'
)
print(f"   Simulator: ${result_sim['price_quantum']:.4f} (Error: {result_sim['error_percent']:.2f}%)")

# Hardware
result_hw = pricer.price_option(
    asset_prices=asset_prices, asset_volatilities=asset_vols,
    correlation_matrix=corr, portfolio_weights=weights,
    K=K, T=T, r=r, backend=hw_backend
)
print(f"   Hardware:  ${result_hw['price_quantum']:.4f} (Error: {result_hw['error_percent']:.2f}%)")

# ===== FINAL RESULTS =====
print("\n" + "=" * 90)
print("  REAL MARKET DATA - IBM HARDWARE RESULTS")
print("=" * 90)

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│  DATA SOURCE                                                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│  Tickers:    {str(tickers):<61} │
│  Period:     {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')} ({len(prices)} days)                     │
│  Regime:     {regime} (ρ = {avg_corr:.3f})                                                  │
│  Backend:    {hw_backend.name}                                                          │
├────────────────────────────────────────────────────────────────────────────────┤
│  COMPONENT          │ SIMULATOR         │ HARDWARE                            │
├────────────────────────────────────────────────────────────────────────────────┤
│  QRC Factors        │ {str(np.round(qrc_factors_sim, 3)):<17} │ {str(np.round(qrc_factors_hw, 3)):<17}  ✅             │
│  QTC Patterns       │ {str(np.round(qtc_patterns_sim, 3)):<17} │ {str(np.round(qtc_patterns_hw, 3)):<17}  ✅             │
│  Fused Features     │ {str(np.round(fused_sim, 3)):<17} │ {str(np.round(fused_hw, 3)):<17}  ✅             │
│  σ_p Enhanced       │ {sigma_p_base*100:.2f}%             │ {sigma_p_hw*100:.2f}%               ✅             │
│  FB-IQFT Price      │ ${result_sim['price_quantum']:.4f}           │ ${result_hw['price_quantum']:.4f}              ✅             │
│  Error vs BS        │ {result_sim['error_percent']:.2f}%              │ {result_hw['error_percent']:.2f}%                              │
├────────────────────────────────────────────────────────────────────────────────┤
│  HARDWARE COMPONENTS: 3/3 (QRC ✓, QTC ✓, FB-IQFT ✓)                           │
│  NOISE CONTRIBUTION: {result_hw['error_percent'] - result_sim['error_percent']:+.2f}%                                                      │
└────────────────────────────────────────────────────────────────────────────────┘
""")

print("✅ QADP REAL MARKET DATA ON IBM HARDWARE - COMPLETE")
print("=" * 90)
