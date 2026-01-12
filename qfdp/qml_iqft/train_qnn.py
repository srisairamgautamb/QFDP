"""
QNN Training and Proper Validation
===================================

This script implements the ACTUAL quantum machine learning:
1. Trains QNN to learn characteristic function from real market data
2. Compares QNN vs Classical NN vs Analytical Gaussian
3. Uses PROPER Monte Carlo reference (correlated multi-asset)
4. Validates no-arbitrage constraints

THIS IS THE CORE QUANTUM WORK!

Author: QFDP Research Team
Date: January 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional
import time

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_algorithms.optimizers import SPSA, COBYLA

# PyTorch for classical baseline
import torch
import torch.nn as nn

# Our modules
from qfdp.qml_iqft.factor_model import PCAFactorModel
from qfdp.qml_iqft.characteristic_function import compute_empirical_cf


@dataclass
class QNNTrainingResult:
    """Result from QNN training."""
    trained_params: np.ndarray
    final_loss: float
    cf_mse: float
    constraint_loss: float  # φ(0) = 1 penalty
    training_time: float
    n_iterations: int
    convergence_history: list


@dataclass 
class CFComparisonResult:
    """Comparison of CF learning methods."""
    # Errors (MAE against empirical CF)
    error_gaussian: float
    error_classical_nn: float
    error_quantum_nn: float
    
    # No-arbitrage check: φ(0) = 1
    phi0_gaussian: complex
    phi0_classical: complex
    phi0_quantum: complex
    
    # Is QNN better?
    qnn_wins: bool


class RealQNNTrainer:
    """
    Trains a REAL Quantum Neural Network on market data.
    
    This is the core QML component that learns the characteristic
    function from empirical market data.
    """
    
    def __init__(
        self,
        n_qubits: int = 3,
        n_layers: int = 2,
        entanglement: str = 'linear'
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.trained_params = None
        self._qnn = None
        
        self._build_circuit()
    
    def _build_circuit(self):
        """Build ZZFeatureMap + RealAmplitudes circuit."""
        # Feature map for encoding frequency (Eq 5.1 from PDF)
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=1,
            entanglement=self.entanglement
        )
        
        # Variational ansatz (Eq 5.1 from PDF)
        self.ansatz = RealAmplitudes(
            num_qubits=self.n_qubits,
            reps=self.n_layers,
            entanglement=self.entanglement
        )
        
        # Full circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.compose(self.ansatz, inplace=True)
        
        self.n_params = self.ansatz.num_parameters
        print(f"🔮 QNN Circuit Built:")
        print(f"   Qubits: {self.n_qubits}")
        print(f"   Layers: {self.n_layers}")
        print(f"   Parameters: {self.n_params}")
        print(f"   Depth: {self.circuit.depth()}")
    
    def _build_qnn(self):
        """Build EstimatorQNN with observables for CF."""
        # Observable for real part: Z on qubit 0
        obs_real = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1.0)])
        
        # Observable for imaginary part: Z on qubit 1
        if self.n_qubits >= 2:
            obs_imag = SparsePauliOp.from_list([("I" + "Z" + "I" * (self.n_qubits - 2), 1.0)])
        else:
            obs_imag = obs_real
        
        self._qnn = EstimatorQNN(
            circuit=self.circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            observables=[obs_real, obs_imag]
        )
        return self._qnn
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        max_iterations: int = 100,
        lambda_constraint: float = 0.1
    ) -> QNNTrainingResult:
        """
        Train QNN to learn characteristic function.
        
        Parameters
        ----------
        X_train : np.ndarray
            Input frequencies (N_samples, 1) - will be tiled to n_qubits
        y_train : np.ndarray
            Target CF [real, imag] (N_samples, 2)
        max_iterations : int
            Maximum SPSA iterations
        lambda_constraint : float
            Weight for φ(0)=1 constraint
            
        Returns
        -------
        QNNTrainingResult
            Training results with optimized parameters
        """
        if self._qnn is None:
            self._build_qnn()
        
        # Tile input to match n_qubits
        X = np.tile(X_train, (1, self.n_qubits))[:, :self.n_qubits]
        
        # Initial parameters
        init_params = np.random.randn(self.n_params) * 0.1
        
        # Find index of u=0 for constraint
        zero_idx = np.argmin(np.abs(X_train[:, 0]))
        X_zero = X[zero_idx:zero_idx+1, :]
        
        # Training tracking
        history = []
        best_loss = float('inf')
        best_params = init_params.copy()
        
        print(f"\n🚀 Training QNN (max {max_iterations} iterations)...")
        start_time = time.time()
        
        def objective(params):
            nonlocal best_loss, best_params
            
            # Forward pass
            pred = self._qnn.forward(X, params)
            
            # MSE loss
            mse = np.mean((pred - y_train) ** 2)
            
            # Constraint: φ(0) = 1 (real=1, imag=0)
            pred_zero = self._qnn.forward(X_zero, params)
            constraint = (pred_zero[0, 0] - 1.0) ** 2 + pred_zero[0, 1] ** 2
            
            # Total loss
            loss = mse + lambda_constraint * constraint
            
            history.append(loss)
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
            
            return loss
        
        # SPSA optimizer
        optimizer = SPSA(maxiter=max_iterations)
        
        try:
            result = optimizer.minimize(objective, init_params)
            final_params = result.x
        except Exception as e:
            print(f"⚠️ Optimizer warning: {e}")
            final_params = best_params
        
        training_time = time.time() - start_time
        
        # Evaluate final model
        self.trained_params = final_params
        final_pred = self._qnn.forward(X, final_params)
        cf_mse = np.mean((final_pred - y_train) ** 2)
        
        # Check constraint
        pred_zero = self._qnn.forward(X_zero, final_params)
        constraint_loss = (pred_zero[0, 0] - 1.0) ** 2 + pred_zero[0, 1] ** 2
        
        print(f"\n✅ QNN Training Complete!")
        print(f"   Time: {training_time:.1f}s")
        print(f"   Final MSE: {cf_mse:.6f}")
        print(f"   φ(0) constraint: {constraint_loss:.6f}")
        print(f"   φ(0) = {pred_zero[0, 0]:.4f} + {pred_zero[0, 1]:.4f}i (target: 1.0)")
        
        return QNNTrainingResult(
            trained_params=final_params,
            final_loss=best_loss,
            cf_mse=cf_mse,
            constraint_loss=constraint_loss,
            training_time=training_time,
            n_iterations=len(history),
            convergence_history=history
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CF values at given frequencies."""
        if self.trained_params is None:
            raise RuntimeError("Model not trained")
        
        X_tiled = np.tile(X.reshape(-1, 1), (1, self.n_qubits))[:, :self.n_qubits]
        return self._qnn.forward(X_tiled, self.trained_params)


class ClassicalCFNet(nn.Module):
    """Classical NN baseline for fair comparison."""
    
    def __init__(self, hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def train_classical_baseline(X_train, y_train, epochs=200):
    """Train classical NN for comparison."""
    model = ClassicalCFNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model


def gaussian_cf(u: np.ndarray, mu: float, sigma: float, T: float = 1.0) -> np.ndarray:
    """Analytical Gaussian characteristic function."""
    # φ(u) = exp(i·u·μ·T - 0.5·σ²·T·u²)
    real_part = np.exp(-0.5 * sigma**2 * T * u**2) * np.cos(u * mu * T)
    imag_part = np.exp(-0.5 * sigma**2 * T * u**2) * np.sin(u * mu * T)
    return np.column_stack([real_part, imag_part])


def monte_carlo_portfolio_reference(
    S0: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    weights: np.ndarray,
    K: float,
    T: float,
    r: float,
    n_sims: int = 500000,
    seed: int = 42
) -> Tuple[float, float]:
    """
    CORRECT Monte Carlo reference for multi-asset portfolio.
    
    This is the proper reference - NOT Black-Scholes!
    """
    np.random.seed(seed)
    N = len(S0)
    
    # Cholesky decomposition for correlated paths
    cov = np.outer(sigma, sigma) * corr
    L = np.linalg.cholesky(cov)
    
    # Generate correlated standard normals
    Z = np.random.standard_normal((n_sims, N))
    Z_corr = Z @ L.T
    
    # GBM simulation
    drift = (r - 0.5 * sigma**2) * T
    diffusion = np.sqrt(T) * Z_corr
    ST = S0 * np.exp(drift + diffusion)
    
    # Portfolio value at maturity
    portfolio_T = ST @ weights
    
    # Option payoff
    payoffs = np.maximum(portfolio_T - K, 0)
    
    # Discounted expectation
    price = np.exp(-r * T) * np.mean(payoffs)
    stderr = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    
    return price, stderr


def run_complete_qnn_experiment():
    """
    Run the COMPLETE QNN experiment with proper validation.
    
    This is what we should have done from the start!
    """
    print("=" * 70)
    print("QUANTUM NEURAL NETWORK TRAINING & VALIDATION")
    print("=" * 70)
    
    # Load real market data
    print("\n📂 Loading real market data...")
    returns = pd.read_csv('data/qml/log_returns.csv', index_col=0)
    prices = pd.read_csv('data/qml/raw_prices.csv', index_col=0)
    
    latest_prices = prices.iloc[-1].values
    weights = np.ones(5) / 5
    corr = returns.corr().values
    vols = returns.std().values * np.sqrt(252)
    
    print(f"   Assets: {list(returns.columns)}")
    print(f"   Observations: {len(returns)}")
    
    # Fit factor model
    print("\n📊 Fitting PCA factor model...")
    factor_model = PCAFactorModel(n_factors=3)
    factor_model.fit(returns.values, risk_free_rate=0.05)
    
    # Compute portfolio returns
    portfolio_returns = returns.values @ weights
    
    # Create frequency grid
    u_grid = np.linspace(-5, 5, 50)
    
    # Compute EMPIRICAL characteristic function (this is the TRUTH)
    print("\n🌊 Computing empirical characteristic function...")
    cf_real, cf_imag = compute_empirical_cf(portfolio_returns, u_grid)
    y_empirical = np.column_stack([cf_real, cf_imag])
    X_train = u_grid.reshape(-1, 1)
    
    print(f"   Frequency range: [{u_grid.min():.1f}, {u_grid.max():.1f}]")
    print(f"   φ(0) empirical = {cf_real[25]:.4f} + {cf_imag[25]:.4f}i")
    
    # ===== METHOD 1: Analytical Gaussian =====
    print("\n📐 Method 1: Analytical Gaussian CF...")
    mu_p = portfolio_returns.mean() * 252
    sigma_p = portfolio_returns.std() * np.sqrt(252)
    cf_gaussian = gaussian_cf(u_grid, mu_p, sigma_p, T=1.0)
    error_gaussian = np.mean(np.abs(cf_gaussian - y_empirical))
    print(f"   Portfolio μ = {mu_p:.4f}, σ = {sigma_p:.4f}")
    print(f"   MAE vs empirical: {error_gaussian:.6f}")
    
    # ===== METHOD 2: Classical Neural Network =====
    print("\n🧠 Method 2: Classical Neural Network...")
    classical_model = train_classical_baseline(X_train, y_empirical, epochs=200)
    with torch.no_grad():
        cf_classical = classical_model(torch.FloatTensor(X_train)).numpy()
    error_classical = np.mean(np.abs(cf_classical - y_empirical))
    print(f"   MAE vs empirical: {error_classical:.6f}")
    
    # ===== METHOD 3: Quantum Neural Network =====
    print("\n🔮 Method 3: Quantum Neural Network...")
    qnn_trainer = RealQNNTrainer(n_qubits=3, n_layers=2)
    qnn_result = qnn_trainer.train(X_train, y_empirical, max_iterations=50)
    
    cf_quantum = qnn_trainer.predict(X_train)
    error_quantum = np.mean(np.abs(cf_quantum - y_empirical))
    print(f"   MAE vs empirical: {error_quantum:.6f}")
    
    # ===== COMPARISON =====
    print("\n" + "=" * 70)
    print("CF LEARNING COMPARISON")
    print("=" * 70)
    print(f"{'Method':<20} {'MAE':<12} {'φ(0) Real':<12} {'φ(0) Imag':<12}")
    print("-" * 56)
    
    # Check φ(0) for each method
    idx_zero = 25  # u=0 is at index 25
    phi0_gaussian = cf_gaussian[idx_zero]
    phi0_classical = cf_classical[idx_zero]
    phi0_quantum = cf_quantum[idx_zero]
    
    print(f"{'Analytical Gaussian':<20} {error_gaussian:<12.6f} {phi0_gaussian[0]:<12.4f} {phi0_gaussian[1]:<12.4f}")
    print(f"{'Classical NN':<20} {error_classical:<12.6f} {phi0_classical[0]:<12.4f} {phi0_classical[1]:<12.4f}")
    print(f"{'Quantum NN':<20} {error_quantum:<12.6f} {phi0_quantum[0]:<12.4f} {phi0_quantum[1]:<12.4f}")
    print("-" * 56)
    print(f"{'Target (φ(0)=1)':<20} {'':<12} {'1.0000':<12} {'0.0000':<12}")
    
    # Determine winner
    qnn_wins = error_quantum < error_gaussian
    
    print("\n" + "=" * 70)
    if qnn_wins:
        improvement = (error_gaussian - error_quantum) / error_gaussian * 100
        print(f"✅ QNN WINS! {improvement:.1f}% better than Gaussian assumption")
    else:
        print(f"⚠️ QNN did not beat Gaussian (need more training)")
    print("=" * 70)
    
    # ===== PROPER PORTFOLIO PRICING =====
    print("\n" + "=" * 70)
    print("PORTFOLIO OPTION PRICING (CORRECT MC REFERENCE)")
    print("=" * 70)
    
    S0 = latest_prices
    portfolio_value = np.sum(S0 * weights)
    K = portfolio_value  # ATM
    T = 1.0
    r = 0.05
    
    print(f"\nPortfolio: ${portfolio_value:.2f}")
    print(f"Strike (ATM): ${K:.2f}")
    
    # CORRECT reference: Multi-asset Monte Carlo
    print("\n💰 Computing Monte Carlo reference (500K paths)...")
    mc_price, mc_stderr = monte_carlo_portfolio_reference(
        S0, vols, corr, weights, K, T, r, n_sims=500000
    )
    print(f"   MC Price: ${mc_price:.4f} ± ${2*mc_stderr:.4f} (95% CI)")
    
    # Save results
    results = {
        'error_gaussian': error_gaussian,
        'error_classical': error_classical,
        'error_quantum': error_quantum,
        'qnn_wins': qnn_wins,
        'qnn_training_time': qnn_result.training_time,
        'qnn_iterations': qnn_result.n_iterations,
        'mc_price': mc_price,
        'mc_stderr': mc_stderr
    }
    
    np.save('data/qml/qnn_results.npy', results)
    np.save('data/qml/qnn_trained_params.npy', qnn_result.trained_params)
    
    print("\n✅ Results saved to data/qml/qnn_results.npy")
    
    return results


if __name__ == '__main__':
    results = run_complete_qnn_experiment()
