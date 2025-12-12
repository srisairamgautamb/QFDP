"""
Unit Tests for Quantum State Preparation
=========================================

Tests Grover-Rudolph amplitude encoding for:
- Marginal distribution fidelity (F ≥ 0.95 for n=8 qubits)
- Gaussian factor fidelity (F ≥ 0.90 for m=6 qubits)
- Log-normal asset price distributions
- Resource cost estimation
- Edge cases (uniform, delta, zero probability)

Run: python3 -m pytest tests/unit/test_state_prep.py -v
"""

import pytest
import numpy as np
from scipy.stats import norm
from qiskit.quantum_info import Statevector

from qfdp_multiasset.state_prep import (
    prepare_marginal_distribution,
    prepare_lognormal_asset,
    prepare_gaussian_factor,
    compute_fidelity,
    estimate_resource_cost
)


class TestMarginalDistribution:
    """Test generic marginal distribution preparation."""
    
    def test_uniform_distribution(self):
        """Uniform distribution should have exact fidelity."""
        n_qubits = 6
        N = 2**n_qubits
        
        # Uniform probabilities
        probabilities = np.ones(N) / N
        prices = np.linspace(0, 1, N)
        
        # Prepare state
        circuit = prepare_marginal_distribution(prices, probabilities, n_qubits)
        
        # Compute fidelity
        fidelity = compute_fidelity(circuit, probabilities)
        
        assert fidelity >= 0.999, f"Uniform fidelity {fidelity:.4f} < 0.999"
        
    def test_gaussian_marginal_fidelity(self):
        """Gaussian marginal should achieve F ≥ 0.95 for n=8 qubits."""
        n_qubits = 8
        N = 2**n_qubits
        
        # Discretized Gaussian
        x_grid = np.linspace(-4, 4, N)
        pdf = norm.pdf(x_grid, loc=0, scale=1)
        probabilities = pdf / pdf.sum()
        
        # Prepare state
        circuit = prepare_marginal_distribution(x_grid, probabilities, n_qubits)
        
        # Validate fidelity threshold
        fidelity = compute_fidelity(circuit, probabilities)
        
        assert fidelity >= 0.95, f"Gaussian marginal fidelity {fidelity:.4f} < 0.95 (Phase 2 threshold)"
        
    def test_automatic_normalization(self):
        """Non-normalized probabilities should be auto-normalized."""
        n_qubits = 4
        N = 2**n_qubits
        
        # Unnormalized probabilities
        probabilities = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        prices = np.arange(N)
        
        # Should not raise error
        circuit = prepare_marginal_distribution(prices, probabilities, n_qubits)
        
        # Verify statevector is normalized
        statevector = Statevector(circuit)
        assert np.isclose(np.sum(np.abs(statevector.data)**2), 1.0, atol=1e-10)
        
    def test_negative_probability_raises_error(self):
        """Negative probabilities should raise ValueError."""
        n_qubits = 3
        N = 2**n_qubits
        
        probabilities = np.array([0.1, 0.2, -0.1, 0.3, 0.2, 0.1, 0.1, 0.1])
        prices = np.arange(N)
        
        with pytest.raises(ValueError, match="Probabilities must be non-negative"):
            prepare_marginal_distribution(prices, probabilities, n_qubits)
            
    def test_delta_distribution(self):
        """Delta function (all mass at one point) should work."""
        n_qubits = 5
        N = 2**n_qubits
        
        # All mass at index 10
        probabilities = np.zeros(N)
        probabilities[10] = 1.0
        prices = np.arange(N)
        
        circuit = prepare_marginal_distribution(prices, probabilities, n_qubits)
        
        # Check statevector has amplitude only at |10⟩
        statevector = Statevector(circuit)
        probs = np.abs(statevector.data)**2
        
        assert probs[10] > 0.99, f"Delta peak at index 10: {probs[10]:.4f}"
        assert np.sum(probs) > 0.99, "Total probability should be ~1"


class TestGaussianFactor:
    """Test Gaussian factor state preparation for copula encoding."""
    
    def test_standard_normal_fidelity(self):
        """Standard normal N(0,1) should achieve F ≥ 0.90 for m=6 qubits."""
        n_qubits = 6
        
        # Prepare factor
        circuit = prepare_gaussian_factor(n_qubits=n_qubits, mean=0, std=1)
        
        # Target distribution
        N = 2**n_qubits
        x_grid = np.linspace(-4, 4, N)
        target_probs = norm.pdf(x_grid, loc=0, scale=1)
        target_probs /= target_probs.sum()
        
        # Validate fidelity threshold
        fidelity = compute_fidelity(circuit, target_probs)
        
        assert fidelity >= 0.90, f"Gaussian factor fidelity {fidelity:.4f} < 0.90 (Phase 2 threshold)"
        
    def test_nonzero_mean_gaussian(self):
        """Gaussian with mean=1, std=0.5 should have high fidelity."""
        n_qubits = 6
        mean, std = 1.0, 0.5
        
        circuit = prepare_gaussian_factor(n_qubits=n_qubits, mean=mean, std=std)
        
        # Target distribution
        N = 2**n_qubits
        x_grid = np.linspace(mean - 4*std, mean + 4*std, N)
        target_probs = norm.pdf(x_grid, loc=mean, scale=std)
        target_probs /= target_probs.sum()
        
        fidelity = compute_fidelity(circuit, target_probs)
        
        assert fidelity >= 0.85, f"Shifted Gaussian fidelity {fidelity:.4f} < 0.85"
        
    def test_gaussian_statistics(self):
        """Prepared Gaussian should have correct mean and std."""
        n_qubits = 7
        target_mean, target_std = 0.0, 1.0
        
        circuit = prepare_gaussian_factor(n_qubits=n_qubits, mean=target_mean, std=target_std)
        
        # Extract probabilities
        statevector = Statevector(circuit)
        probs = np.abs(statevector.data)**2
        
        # Grid points
        N = 2**n_qubits
        x_grid = np.linspace(-4, 4, N)
        
        # Compute empirical statistics
        empirical_mean = np.sum(x_grid * probs)
        empirical_var = np.sum((x_grid - empirical_mean)**2 * probs)
        empirical_std = np.sqrt(empirical_var)
        
        assert abs(empirical_mean - target_mean) < 0.1, f"Mean error: {empirical_mean:.3f} vs {target_mean}"
        assert abs(empirical_std - target_std) < 0.15, f"Std error: {empirical_std:.3f} vs {target_std}"


class TestLognormalAsset:
    """Test log-normal asset price distribution preparation."""
    
    def test_lognormal_preparation(self):
        """Log-normal asset should prepare successfully."""
        S0, r, sigma, T = 100.0, 0.05, 0.2, 1.0
        n_qubits = 8
        
        circuit, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits=n_qubits)
        
        # Check circuit exists
        assert circuit.num_qubits == n_qubits
        assert len(prices) == 2**n_qubits
        
        # Check price range is reasonable
        assert prices[0] > 0, "Prices must be positive"
        assert prices[-1] > prices[0], "Prices must be increasing"
        
    def test_lognormal_fidelity(self):
        """Log-normal asset should have high fidelity vs theoretical."""
        S0, r, sigma, T = 150.0, 0.03, 0.25, 1.0
        n_qubits = 8
        
        circuit, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits=n_qubits)
        
        # Target log-normal distribution
        mu = (r - 0.5*sigma**2) * T
        sigma_r = sigma * np.sqrt(T)
        log_returns = np.log(prices / S0)
        pdf_log = norm.pdf(log_returns, loc=mu, scale=sigma_r)
        pdf_price = pdf_log / prices
        target_probs = pdf_price / pdf_price.sum()
        
        # Compute fidelity
        fidelity = compute_fidelity(circuit, target_probs)
        
        assert fidelity >= 0.95, f"Log-normal fidelity {fidelity:.4f} < 0.95"
        
    def test_lognormal_expected_value(self):
        """Prepared log-normal should match theoretical expected price."""
        S0, r, sigma, T = 100.0, 0.05, 0.3, 2.0
        n_qubits = 8
        
        circuit, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits=n_qubits)
        
        # Get probabilities
        statevector = Statevector(circuit)
        probs = np.abs(statevector.data)**2
        
        # Empirical expected price
        empirical_E = np.sum(prices * probs)
        
        # Theoretical: E[S_T] = S0 * exp(r*T)
        theoretical_E = S0 * np.exp(r * T)
        
        # Relative error should be < 20% (discretization and truncation effects)
        rel_error = abs(empirical_E - theoretical_E) / theoretical_E
        
        assert rel_error < 0.20, f"Expected price error: {rel_error:.2%} (empirical={empirical_E:.2f}, theoretical={theoretical_E:.2f})"
        
    def test_zero_volatility_degeneracy(self):
        """Zero volatility should prepare successfully (note: discretization prevents true delta)."""
        S0, r, sigma, T = 100.0, 0.05, 0.001, 1.0  # Near-zero vol
        n_qubits = 6
        
        # Should not crash with very low volatility
        circuit, prices = prepare_lognormal_asset(S0, r, sigma, T, n_qubits=n_qubits)
        
        # Get probabilities
        statevector = Statevector(circuit)
        probs = np.abs(statevector.data)**2
        
        # Verify statevector is valid
        assert np.isclose(np.sum(probs), 1.0, atol=1e-10)
        assert len(prices) == 2**n_qubits


class TestResourceCosts:
    """Test T-count and depth estimation."""
    
    def test_resource_scaling(self):
        """T-count should scale as 3×(2^n - 1)."""
        test_cases = [
            (4, 45),    # 3 × (16-1) = 45
            (6, 189),   # 3 × (64-1) = 189
            (8, 765),   # 3 × (256-1) = 765
        ]
        
        for n_qubits, expected_t_count in test_cases:
            resources = estimate_resource_cost(n_qubits)
            
            assert resources['t_count_estimate'] == expected_t_count, \
                f"n={n_qubits}: T-count {resources['t_count_estimate']} != {expected_t_count}"
                
    def test_resource_multiasset(self):
        """Multi-asset resource cost should be N × single-asset cost."""
        n_qubits = 8
        N_assets = 5
        
        single = estimate_resource_cost(n_qubits)
        total_t_count = N_assets * single['t_count_estimate']
        
        # For 5 assets with 8 qubits each:
        # Single: 3 × 255 = 765
        # Total: 5 × 765 = 3,825
        assert total_t_count == 3825, f"Multi-asset T-count: {total_t_count}"
        
    def test_resource_dict_structure(self):
        """Resource dict should contain required keys."""
        resources = estimate_resource_cost(6)
        
        required_keys = {'n_qubits', 'ry_gates', 't_count_estimate', 
                        't_count_per_qubit', 'depth_estimate', 'formula'}
        
        assert set(resources.keys()) == required_keys, f"Missing keys: {required_keys - set(resources.keys())}"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_qubit_preparation(self):
        """Single qubit should work (n=1)."""
        n_qubits = 1
        probabilities = np.array([0.7, 0.3])
        prices = np.array([0, 1])
        
        circuit = prepare_marginal_distribution(prices, probabilities, n_qubits)
        
        # Validate
        statevector = Statevector(circuit)
        probs = np.abs(statevector.data)**2
        
        assert np.isclose(probs[0], 0.7, atol=0.01)
        assert np.isclose(probs[1], 0.3, atol=0.01)
        
    def test_large_n_qubits(self):
        """Test with n=10 qubits (1024 points)."""
        n_qubits = 10
        N = 2**n_qubits
        
        # Gaussian
        x_grid = np.linspace(-4, 4, N)
        pdf = norm.pdf(x_grid)
        probabilities = pdf / pdf.sum()
        
        # Should not crash
        circuit = prepare_marginal_distribution(x_grid, probabilities, n_qubits)
        
        assert circuit.num_qubits == n_qubits
        
    def test_probability_padding(self):
        """Probabilities with length < 2^n should be padded."""
        n_qubits = 4  # Expects 16 elements
        probabilities = np.array([0.2, 0.3, 0.5])  # Only 3 elements
        prices = np.arange(3)
        
        circuit = prepare_marginal_distribution(prices, probabilities, n_qubits)
        
        # Should pad with zeros
        statevector = Statevector(circuit)
        probs = np.abs(statevector.data)**2
        
        # First 3 should have mass, rest should be near zero
        assert np.sum(probs[:3]) > 0.95
        assert np.sum(probs[3:]) < 0.05
        
    def test_probability_truncation(self):
        """Probabilities with length > 2^n should be truncated."""
        n_qubits = 3  # Expects 8 elements
        probabilities = np.ones(20) / 20  # 20 elements
        prices = np.arange(20)
        
        circuit = prepare_marginal_distribution(prices, probabilities, n_qubits)
        
        # Should truncate to 8
        assert circuit.num_qubits == n_qubits


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
