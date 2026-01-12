"""
Quantum Factor Autoencoder
===========================
Replaces classical PCA with quantum dimensionality reduction.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA


@dataclass
class QAEResult:
    """Result from QAE training or factor extraction."""
    factors: np.ndarray
    reconstruction_error: float
    trained_params: Optional[np.ndarray] = None
    n_iterations: int = 0
    training_time: float = 0.0


class QuantumFactorAutoencoder:
    """
    Quantum Autoencoder for correlation matrix factor extraction.
    Replaces classical PCA in FB-IQFT pipeline.
    """
    
    def __init__(self, n_assets: int = 5, n_factors: int = 3, n_layers: int = 2):
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.n_layers = n_layers
        self.n_qubits = max(3, int(np.ceil(np.log2(n_assets * (n_assets + 1) // 2))))
        self.trained_params = None
        self._build_circuit()
    
    def _build_variational_layer(self, qc: QuantumCircuit, params: ParameterVector, offset: int):
        """Add one variational layer with RY and CX gates."""
        n = self.n_qubits
        for i in range(n):
            qc.ry(params[offset + i], i)
        for i in range(n - 1):
            qc.cx(i, i + 1)
    
    def _build_circuit(self):
        """Build encoder-decoder circuit with separate parameters."""
        n_params_per_layer = self.n_qubits
        self.n_encoder_params = n_params_per_layer * self.n_layers
        self.n_decoder_params = n_params_per_layer * self.n_layers
        self.n_total_params = self.n_encoder_params + self.n_decoder_params
        
        enc_params = ParameterVector('enc', self.n_encoder_params)
        dec_params = ParameterVector('dec', self.n_decoder_params)
        
        self.full_circuit = QuantumCircuit(self.n_qubits)
        
        for layer in range(self.n_layers):
            offset = layer * n_params_per_layer
            self._build_variational_layer(self.full_circuit, enc_params, offset)
        
        self.full_circuit.barrier()
        
        for layer in range(self.n_layers):
            offset = layer * n_params_per_layer
            self._build_variational_layer(self.full_circuit, dec_params, offset)
        
        self.enc_params = enc_params
        self.dec_params = dec_params
    
    def encode_correlation(self, corr: np.ndarray) -> np.ndarray:
        """Encode correlation matrix as amplitude vector."""
        n = corr.shape[0]
        upper_tri = corr[np.triu_indices(n)]
        amplitudes = upper_tri / (np.linalg.norm(upper_tri) + 1e-10)
        padded = np.zeros(2 ** self.n_qubits)
        padded[:len(amplitudes)] = amplitudes
        padded = padded / (np.linalg.norm(padded) + 1e-10)
        return padded
    
    def decode_to_correlation(self, amplitudes: np.ndarray) -> np.ndarray:
        """Decode amplitude vector back to correlation matrix."""
        n = self.n_assets
        n_elem = n * (n + 1) // 2
        upper_tri = amplitudes[:n_elem]
        corr = np.zeros((n, n))
        corr[np.triu_indices(n)] = upper_tri
        corr = corr + corr.T - np.diag(np.diag(corr))
        np.fill_diagonal(corr, 1.0)
        return corr
    
    def _forward(self, input_amps: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Forward pass through autoencoder."""
        param_dict = {}
        for i, p in enumerate(self.enc_params):
            param_dict[p] = params[i]
        for i, p in enumerate(self.dec_params):
            param_dict[p] = params[self.n_encoder_params + i]
        
        qc = QuantumCircuit(self.n_qubits)
        qc.initialize(input_amps, range(self.n_qubits))
        qc.compose(self.full_circuit.assign_parameters(param_dict), inplace=True)
        sv = Statevector(qc)
        return np.abs(sv.data)
    
    def _compute_loss(self, params: np.ndarray, corr_matrices: List[np.ndarray]) -> float:
        """Compute reconstruction loss."""
        total_loss = 0.0
        for corr in corr_matrices:
            input_amps = self.encode_correlation(corr)
            output_amps = self._forward(input_amps, params)
            recon_corr = self.decode_to_correlation(output_amps)
            total_loss += np.mean((corr - recon_corr) ** 2)
        return total_loss / len(corr_matrices)
    
    def train(self, correlation_matrices: List[np.ndarray], max_iter: int = 100) -> QAEResult:
        """Train autoencoder on correlation data."""
        import time
        start = time.time()
        
        init_params = np.random.randn(self.n_total_params) * 0.1
        
        def objective(p):
            return self._compute_loss(p, correlation_matrices)
        
        optimizer = COBYLA(maxiter=max_iter)
        result = optimizer.minimize(objective, init_params)
        
        self.trained_params = result.x
        
        return QAEResult(
            factors=np.zeros(self.n_factors),
            reconstruction_error=result.fun,
            trained_params=result.x,
            n_iterations=getattr(result, 'nit', max_iter),
            training_time=time.time() - start
        )
    
    def extract_factors(self, corr: np.ndarray) -> np.ndarray:
        """Extract quantum factors from correlation matrix."""
        if self.trained_params is None:
            raise ValueError("Model not trained")
        
        input_amps = self.encode_correlation(corr)
        
        param_dict = {}
        for i, p in enumerate(self.enc_params):
            param_dict[p] = self.trained_params[i]
        
        qc = QuantumCircuit(self.n_qubits)
        qc.initialize(input_amps, range(self.n_qubits))
        
        enc_circuit = QuantumCircuit(self.n_qubits)
        for layer in range(self.n_layers):
            offset = layer * self.n_qubits
            for i in range(self.n_qubits):
                enc_circuit.ry(self.enc_params[offset + i], i)
            for i in range(self.n_qubits - 1):
                enc_circuit.cx(i, i + 1)
        
        qc.compose(enc_circuit.assign_parameters(param_dict), inplace=True)
        sv = Statevector(qc)
        probs = np.abs(sv.data) ** 2
        
        factors = np.zeros(self.n_factors)
        for i in range(min(self.n_factors, len(probs))):
            factors[i] = probs[i] - 0.5
        
        return factors
    
    def compare_with_pca(self, corr: np.ndarray) -> dict:
        """Compare QAE factors with classical PCA."""
        qae_factors = self.extract_factors(corr)
        
        eigenvalues = np.linalg.eigvalsh(corr)[::-1]
        pca_factors = eigenvalues[:self.n_factors]
        pca_explained = pca_factors.sum() / eigenvalues.sum()
        
        return {
            'qae_factors': qae_factors,
            'pca_factors': pca_factors,
            'pca_explained_variance': pca_explained
        }
