"""
QRC Training Utilities
======================

Training loop and utilities for optimizing QRC parameters.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import logging

from .quantum_recurrent_circuit import QuantumRecurrentCircuit, QRCResult

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Training result summary."""
    final_loss: float
    losses: List[float]
    n_epochs: int
    best_parameters: Dict[str, np.ndarray]


class QRCTrainer:
    """
    Trainer for Quantum Recurrent Circuit parameters.
    
    Uses numerical gradient estimation for parameter optimization
    since quantum gradients are not directly available in simulation.
    
    Example:
        >>> qrc = QuantumRecurrentCircuit()
        >>> trainer = QRCTrainer(qrc, learning_rate=0.01)
        >>> result = trainer.train(train_data, n_epochs=50)
    """
    
    def __init__(
        self,
        qrc: QuantumRecurrentCircuit,
        learning_rate: float = 0.01,
        epsilon: float = 1e-4
    ):
        """
        Initialize trainer.
        
        Args:
            qrc: The QRC to train
            learning_rate: Step size for parameter updates
            epsilon: Small value for numerical gradient estimation
        """
        self.qrc = qrc
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.best_loss = float('inf')
        self.best_params = None
    
    def train(
        self,
        train_data: List[Tuple[Dict, np.ndarray]],
        n_epochs: int = 50,
        verbose: bool = True
    ) -> TrainingResult:
        """
        Train QRC on market data samples.
        
        Args:
            train_data: List of (market_data, target_factors) tuples
            n_epochs: Number of training epochs
            verbose: Print progress
        
        Returns:
            TrainingResult with loss history and best parameters
        """
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            self.qrc.reset_hidden_state()
            
            for market_data, target_factors in train_data:
                # Forward pass
                result = self.qrc.forward(market_data)
                predicted = result.factors
                
                # Compute MSE loss
                loss = np.mean((predicted - target_factors) ** 2)
                epoch_loss += loss
                
                # Compute gradients and update parameters
                self._gradient_update(market_data, target_factors)
            
            avg_loss = epoch_loss / len(train_data)
            losses.append(avg_loss)
            
            # Track best
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_params = self.qrc.get_parameters()
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.6f}")
        
        return TrainingResult(
            final_loss=losses[-1],
            losses=losses,
            n_epochs=n_epochs,
            best_parameters=self.best_params
        )
    
    def _gradient_update(self, market_data: Dict, target: np.ndarray) -> None:
        """
        Update parameters using numerical gradient estimation.
        
        Uses central difference: ∂L/∂θ ≈ (L(θ+ε) - L(θ-ε)) / 2ε
        """
        # Update theta parameters
        for l in range(self.qrc.n_deep_layers):
            for q in range(self.qrc.n_qubits):
                # Compute gradient for theta[l, q]
                original = self.qrc.theta[l, q]
                
                # Forward pass with θ + ε
                self.qrc.theta[l, q] = original + self.epsilon
                result_plus = self.qrc.forward(market_data)
                loss_plus = np.mean((result_plus.factors - target) ** 2)
                
                # Forward pass with θ - ε
                self.qrc.theta[l, q] = original - self.epsilon
                result_minus = self.qrc.forward(market_data)
                loss_minus = np.mean((result_minus.factors - target) ** 2)
                
                # Restore and update
                self.qrc.theta[l, q] = original
                gradient = (loss_plus - loss_minus) / (2 * self.epsilon)
                self.qrc.theta[l, q] -= self.learning_rate * gradient
        
        # Update phi parameters (similar process, abbreviated)
        for l in range(self.qrc.n_deep_layers):
            for q in range(self.qrc.n_qubits):
                original = self.qrc.phi[l, q]
                
                self.qrc.phi[l, q] = original + self.epsilon
                loss_plus = np.mean((self.qrc.forward(market_data).factors - target) ** 2)
                
                self.qrc.phi[l, q] = original - self.epsilon
                loss_minus = np.mean((self.qrc.forward(market_data).factors - target) ** 2)
                
                self.qrc.phi[l, q] = original
                gradient = (loss_plus - loss_minus) / (2 * self.epsilon)
                self.qrc.phi[l, q] -= self.learning_rate * gradient


def train_qrc(
    qrc: QuantumRecurrentCircuit,
    train_data: List[Tuple[Dict, np.ndarray]],
    learning_rate: float = 0.01,
    n_epochs: int = 50,
    verbose: bool = True
) -> TrainingResult:
    """
    Convenience function to train a QRC.
    
    Args:
        qrc: QRC instance to train
        train_data: Training data as (market_data, target_factors) tuples
        learning_rate: Optimization step size
        n_epochs: Number of training epochs
        verbose: Print progress
    
    Returns:
        TrainingResult with loss history
    """
    trainer = QRCTrainer(qrc, learning_rate=learning_rate)
    return trainer.train(train_data, n_epochs=n_epochs, verbose=verbose)
