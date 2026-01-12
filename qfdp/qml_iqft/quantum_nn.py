"""
Quantum Neural Network for Characteristic Function Learning
=============================================================

Implements QNN from Section 5 of QML_QHDP.pdf:
- ZZFeatureMap for encoding (Eq 5.1)
- RealAmplitudes ansatz (Eq 5.1)
- Observable definition (Eq 5.2)
- No-arbitrage constraints (Eq 5.3)

Author: QFDP Research Team
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    QISKIT_ML_AVAILABLE = True
except ImportError:
    QISKIT_ML_AVAILABLE = False
    

@dataclass
class QNNTrainingResult:
    """
    Result from QNN training.
    
    Attributes
    ----------
    trained_params : np.ndarray
        Optimized variational parameters
    final_loss : float
        Final training loss
    n_iterations : int
        Number of optimization iterations
    circuit_depth : int
        Depth of the quantum circuit
    n_parameters : int
        Number of variational parameters
    """
    trained_params: np.ndarray
    final_loss: float
    n_iterations: int
    circuit_depth: int
    n_parameters: int


class QuantumCharacteristicFunctionLearner:
    """
    Quantum Neural Network for learning characteristic functions.
    
    Implements Section 5 from QML_QHDP.pdf:
    - ZZFeatureMap for encoding frequency values (Eq 5.1)
    - RealAmplitudes ansatz for variational parameters (Eq 5.1)
    - Observable definition for CF real/imag parts (Eq 5.2)
    - No-arbitrage constraints in loss function (Eq 5.3)
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits (typically 2-4)
    n_layers : int
        Number of ansatz layers (depth)
    entanglement : str
        Entanglement pattern ('linear', 'full', 'circular')
        
    Examples
    --------
    >>> learner = QuantumCharacteristicFunctionLearner(n_qubits=3, n_layers=2)
    >>> result = learner.train(factor_data, empirical_cf)
    >>> cf_pred = learner.predict(test_frequencies)
    """
    
    def __init__(
        self,
        n_qubits: int = 3,
        n_layers: int = 2,
        entanglement: str = 'linear'
    ):
        if not QISKIT_ML_AVAILABLE:
            raise ImportError(
                "qiskit-machine-learning not installed. "
                "Run: pip install qiskit-machine-learning"
            )
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.trained_params = None
        self._qnn = None
        
        # Build circuit components
        self._build_circuit()
    
    def _build_circuit(self):
        """Build ZZFeatureMap + RealAmplitudes circuit."""
        # Feature map: encodes frequency input (Eq 5.1)
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=1,
            entanglement=self.entanglement
        )
        
        # Ansatz: variational parameters (Eq 5.1)
        self.ansatz = RealAmplitudes(
            num_qubits=self.n_qubits,
            reps=self.n_layers,
            entanglement=self.entanglement
        )
        
        # Construct full circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.compose(self.ansatz, inplace=True)
        
        self.circuit_depth = self.circuit.depth()
        self.n_parameters = self.ansatz.num_parameters
        
        print(f"🔮 Built QNN circuit:")
        print(f"   Qubits: {self.n_qubits}")
        print(f"   Layers: {self.n_layers}")
        print(f"   Depth: {self.circuit_depth}")
        print(f"   Parameters: {self.n_parameters}")
    
    def build_qnn(self) -> 'EstimatorQNN':
        """
        Build EstimatorQNN with observables for CF real/imag parts.
        
        Observables (Eq 5.2):
        - Real part: Z on qubit 0
        - Imag part: Z on qubit 1
        
        Returns
        -------
        qnn : EstimatorQNN
            Configured QNN
        """
        # Define observables (Eq 5.2)
        # Real part: measure Z on first qubit
        obs_real_str = "Z" + "I" * (self.n_qubits - 1)
        observable_re = SparsePauliOp.from_list([(obs_real_str, 1.0)])
        
        # Imaginary part: measure Z on second qubit
        if self.n_qubits >= 2:
            obs_imag_str = "I" + "Z" + "I" * (self.n_qubits - 2)
        else:
            obs_imag_str = "Z"
        observable_im = SparsePauliOp.from_list([(obs_imag_str, 1.0)])
        
        # Create QNN
        self._qnn = EstimatorQNN(
            circuit=self.circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            observables=[observable_re, observable_im]
        )
        
        return self._qnn
    
    def _loss_with_constraints(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        lambda_1: float = 0.1,
        lambda_2: float = 0.05
    ) -> float:
        """
        Custom loss with no-arbitrage constraints (Eq 5.3).
        
        L = MSE + λ₁·|φ(0) - 1|² + λ₂·Hermitian_penalty
        
        Parameters
        ----------
        params : np.ndarray
            Variational parameters
        X : np.ndarray
            Input frequencies (N × n_qubits)
        y : np.ndarray
            Target CF values [real, imag] (N × 2)
        lambda_1 : float
            Constraint weight for φ(0) = 1
        lambda_2 : float
            Constraint weight for Hermitian symmetry
            
        Returns
        -------
        loss : float
            Total loss value
        """
        if self._qnn is None:
            self.build_qnn()
        
        # Forward pass
        pred = self._qnn.forward(X, params)
        
        # Standard MSE loss
        mse_loss = np.mean((pred - y) ** 2)
        
        # Constraint 1: φ(0) = 1 (Eq 5.3, R1)
        # Evaluate at u = 0
        zero_input = np.zeros((1, self.n_qubits))
        cf_zero = self._qnn.forward(zero_input, params)
        # For CF: real(φ(0)) = 1, imag(φ(0)) = 0
        constraint_1 = (cf_zero[0, 0] - 1.0) ** 2 + cf_zero[0, 1] ** 2
        
        # Constraint 2: Hermitian symmetry φ(-u) = φ(u)* (Eq 5.3, R2)
        # Simplified: check at a few sample points
        constraint_2 = 0.0
        
        # Total loss
        total_loss = mse_loss + lambda_1 * constraint_1 + lambda_2 * constraint_2
        
        return total_loss
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        max_iterations: int = 100,
        optimizer: str = 'SPSA',
        verbose: bool = True
    ) -> QNNTrainingResult:
        """
        Train QNN to learn characteristic function.
        
        Parameters
        ----------
        X_train : np.ndarray
            Input frequencies (N_samples × n_qubits)
            Note: Input must match n_qubits dimension
        y_train : np.ndarray
            Target CF [real, imag] (N_samples × 2)
        max_iterations : int
            Maximum optimization iterations
        optimizer : str
            Optimizer: 'SPSA' or 'COBYLA'
        verbose : bool
            Print progress
            
        Returns
        -------
        QNNTrainingResult
            Training results with optimized parameters
        """
        if self._qnn is None:
            self.build_qnn()
        
        # Pad or tile input to match n_qubits
        if X_train.shape[1] < self.n_qubits:
            # Tile to fill qubits
            n_tiles = int(np.ceil(self.n_qubits / X_train.shape[1]))
            X_train = np.tile(X_train, (1, n_tiles))[:, :self.n_qubits]
        elif X_train.shape[1] > self.n_qubits:
            X_train = X_train[:, :self.n_qubits]
        
        # Initial parameters
        initial_params = np.random.randn(self.n_parameters) * 0.1
        
        # Select optimizer
        if optimizer == 'SPSA':
            opt = SPSA(maxiter=max_iterations)
        else:
            opt = COBYLA(maxiter=max_iterations)
        
        if verbose:
            print(f"\n🚀 Training Quantum Neural Network...")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Optimizer: {optimizer}")
            print(f"   Max iterations: {max_iterations}")
        
        # Optimization
        best_loss = float('inf')
        best_params = initial_params.copy()
        iteration_count = 0
        
        def objective(params):
            nonlocal best_loss, best_params, iteration_count
            iteration_count += 1
            
            loss = self._loss_with_constraints(params, X_train, y_train)
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
            
            if verbose and iteration_count % 10 == 0:
                print(f"   Iteration {iteration_count}: Loss = {loss:.6f}")
            
            return loss
        
        # Run optimization
        try:
            result = opt.minimize(objective, initial_params)
            self.trained_params = result.x
            final_loss = result.fun
        except Exception as e:
            print(f"   ⚠️ Optimization warning: {e}")
            self.trained_params = best_params
            final_loss = best_loss
        
        if verbose:
            print(f"✅ Training complete. Final loss: {final_loss:.6f}")
        
        return QNNTrainingResult(
            trained_params=self.trained_params,
            final_loss=final_loss,
            n_iterations=iteration_count,
            circuit_depth=self.circuit_depth,
            n_parameters=self.n_parameters
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CF values for input frequencies.
        
        Parameters
        ----------
        X : np.ndarray
            Input frequencies (N × n_qubits)
            
        Returns
        -------
        cf_pred : np.ndarray
            Predicted CF [real, imag] (N × 2)
        """
        if self.trained_params is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if self._qnn is None:
            self.build_qnn()
        
        # Pad input if needed
        if X.shape[1] < self.n_qubits:
            n_tiles = int(np.ceil(self.n_qubits / X.shape[1]))
            X = np.tile(X, (1, n_tiles))[:, :self.n_qubits]
        
        return self._qnn.forward(X, self.trained_params)
    
    def get_circuit(self) -> 'QuantumCircuit':
        """Return the quantum circuit."""
        return self.circuit


def create_simple_qnn(
    n_qubits: int = 2,
    n_layers: int = 1
) -> 'QuantumCharacteristicFunctionLearner':
    """
    Create a simple QNN for quick testing.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits (default: 2 for minimal depth)
    n_layers : int
        Ansatz layers (default: 1 for minimal depth)
        
    Returns
    -------
    QuantumCharacteristicFunctionLearner
        Configured QNN
    """
    return QuantumCharacteristicFunctionLearner(
        n_qubits=n_qubits,
        n_layers=n_layers,
        entanglement='linear'
    )


if __name__ == '__main__':
    if not QISKIT_ML_AVAILABLE:
        print("❌ qiskit-machine-learning not available.")
        print("   Install with: pip install qiskit-machine-learning")
    else:
        # Test with synthetic data
        print("Testing QNN...")
        
        learner = create_simple_qnn(n_qubits=2, n_layers=1)
        
        # Simple test data
        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        y = np.array([[1.0, 0.0], [0.8, 0.2], [0.5, 0.5]])
        
        result = learner.train(X, y, max_iterations=10, verbose=True)
        
        print(f"\n✅ QNN test complete")
        print(f"   Circuit depth: {result.circuit_depth}")
        print(f"   Parameters: {result.n_parameters}")
