"""
Classical Neural Network Baseline for QML-IQFT
===============================================

1-layer neural network for factor→price mapping.
Provides baseline for comparison with quantum neural network.

Author: QFDP Research Team
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ClassicalNNResult:
    """
    Result from classical neural network training.
    
    Attributes
    ----------
    final_loss : float
        Final training loss
    epochs : int
        Number of training epochs
    train_losses : list
        Loss history during training
    model_params : int
        Number of model parameters
    """
    final_loss: float
    epochs: int
    train_losses: list
    model_params: int


if TORCH_AVAILABLE:
    class ClassicalFactorPricingNN(nn.Module):
        """
        Classical neural network: Factor → Price mapping.
        
        Simple 1-layer network for baseline comparison with QNN.
        
        Architecture:
        - Input: K factors
        - Hidden: hidden_dim neurons with ReLU
        - Output: 1 (option price)
        - Softplus to ensure positive prices
        
        Parameters
        ----------
        n_factors : int
            Number of input factors (K)
        hidden_dim : int
            Hidden layer dimension
            
        Examples
        --------
        >>> model = ClassicalFactorPricingNN(n_factors=3, hidden_dim=16)
        >>> factors = torch.randn(32, 3)  # batch of 32
        >>> prices = model(factors)
        >>> print(prices.shape)  # (32, 1)
        """
        
        def __init__(self, n_factors: int = 3, hidden_dim: int = 16):
            super().__init__()
            self.n_factors = n_factors
            self.hidden_dim = hidden_dim
            
            self.net = nn.Sequential(
                nn.Linear(n_factors, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()  # Ensure positive prices
            )
        
        def forward(self, factors: torch.Tensor) -> torch.Tensor:
            """
            Forward pass: factors → price.
            
            Parameters
            ----------
            factors : torch.Tensor
                Factor values (batch_size × n_factors)
                
            Returns
            -------
            prices : torch.Tensor
                Predicted option prices (batch_size × 1)
            """
            return self.net(factors)
        
        def count_parameters(self) -> int:
            """Count total trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    class ClassicalCFLearner(nn.Module):
        """
        Classical neural network for characteristic function learning.
        
        Learns the mapping: frequency u → CF value (real, imag)
        For direct comparison with QuantumCharacteristicFunctionLearner.
        
        Parameters
        ----------
        input_dim : int
            Frequency dimension (typically 1 for scalar input)
        hidden_dim : int
            Hidden layer dimension
        n_layers : int
            Number of hidden layers
        """
        
        def __init__(
            self,
            input_dim: int = 1,
            hidden_dim: int = 32,
            n_layers: int = 2
        ):
            super().__init__()
            
            layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
            for _ in range(n_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
            layers.append(nn.Linear(hidden_dim, 2))  # Output: [real, imag]
            
            self.net = nn.Sequential(*layers)
        
        def forward(self, u: torch.Tensor) -> torch.Tensor:
            """
            Forward pass: frequency → CF value.
            
            Parameters
            ----------
            u : torch.Tensor
                Frequency values (batch_size × input_dim)
                
            Returns
            -------
            cf : torch.Tensor
                CF values [real, imag] (batch_size × 2)
            """
            return self.net(u)


def train_classical_baseline(
    factors_train: np.ndarray,
    prices_train: np.ndarray,
    epochs: int = 100,
    hidden_dim: int = 16,
    learning_rate: float = 0.01,
    verbose: bool = True
) -> Tuple['ClassicalFactorPricingNN', ClassicalNNResult]:
    """
    Train classical neural network baseline.
    
    Parameters
    ----------
    factors_train : np.ndarray
        Training factor values (N_samples × K)
    prices_train : np.ndarray
        Training prices (N_samples,) or (N_samples × 1)
    epochs : int
        Number of training epochs
    hidden_dim : int
        Hidden layer dimension
    learning_rate : float
        Learning rate for Adam optimizer
    verbose : bool
        Print training progress
        
    Returns
    -------
    model : ClassicalFactorPricingNN
        Trained model
    result : ClassicalNNResult
        Training results
        
    Examples
    --------
    >>> model, result = train_classical_baseline(factors, prices)
    >>> print(f"Final loss: {result.final_loss:.6f}")
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed. Run: pip install torch")
    
    n_factors = factors_train.shape[1]
    
    # Convert to tensors
    X = torch.FloatTensor(factors_train)
    y = torch.FloatTensor(prices_train).reshape(-1, 1)
    
    # Create model
    model = ClassicalFactorPricingNN(n_factors=n_factors, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    
    if verbose:
        print(f"🧠 Training Classical NN Baseline:")
        print(f"   Input: {n_factors} factors")
        print(f"   Hidden: {hidden_dim} neurons")
        print(f"   Parameters: {model.count_parameters()}")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        pred_prices = model(X)
        loss = criterion(pred_prices, y)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if verbose and (epoch + 1) % (epochs // 5) == 0:
            print(f"   Epoch {epoch+1:4d}: Loss = {loss.item():.6f}")
    
    model.eval()
    
    result = ClassicalNNResult(
        final_loss=train_losses[-1],
        epochs=epochs,
        train_losses=train_losses,
        model_params=model.count_parameters()
    )
    
    if verbose:
        print(f"✅ Training complete. Final loss: {result.final_loss:.6f}")
    
    return model, result


def train_cf_learner(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 200,
    hidden_dim: int = 32,
    n_layers: int = 2,
    learning_rate: float = 0.01,
    verbose: bool = True
) -> Tuple['ClassicalCFLearner', ClassicalNNResult]:
    """
    Train classical CF learner (baseline for QNN comparison).
    
    Parameters
    ----------
    X_train : np.ndarray
        Frequency values (N_samples × 1)
    y_train : np.ndarray
        CF values [real, imag] (N_samples × 2)
    epochs : int
        Number of training epochs
    hidden_dim : int
        Hidden layer dimension
    n_layers : int
        Number of hidden layers
    learning_rate : float
        Learning rate
    verbose : bool
        Print progress
        
    Returns
    -------
    model : ClassicalCFLearner
        Trained model
    result : ClassicalNNResult
        Training results
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed. Run: pip install torch")
    
    # Convert to tensors
    X = torch.FloatTensor(X_train)
    y = torch.FloatTensor(y_train)
    
    # Create model
    model = ClassicalCFLearner(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        n_layers=n_layers
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    
    if verbose:
        print(f"🧠 Training Classical CF Learner:")
        print(f"   Input: frequency (dim={X_train.shape[1]})")
        print(f"   Hidden: {hidden_dim} × {n_layers} layers")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        pred_cf = model(X)
        loss = criterion(pred_cf, y)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if verbose and (epoch + 1) % (epochs // 5) == 0:
            print(f"   Epoch {epoch+1:4d}: Loss = {loss.item():.6f}")
    
    model.eval()
    
    result = ClassicalNNResult(
        final_loss=train_losses[-1],
        epochs=epochs,
        train_losses=train_losses,
        model_params=sum(p.numel() for p in model.parameters())
    )
    
    if verbose:
        print(f"✅ Training complete. Final loss: {result.final_loss:.6f}")
    
    return model, result


if __name__ == '__main__':
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Install with: pip install torch")
    else:
        # Test with synthetic data
        np.random.seed(42)
        N_samples = 100
        n_factors = 3
        
        # Synthetic factor → price relationship
        factors = np.random.randn(N_samples, n_factors)
        prices = 10 + 2 * factors[:, 0] + np.random.randn(N_samples) * 0.1
        
        model, result = train_classical_baseline(factors, prices, epochs=50)
        
        print(f"\n✅ Classical NN test complete")
        print(f"   Parameters: {result.model_params}")
