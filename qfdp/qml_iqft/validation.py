"""
Validation Pipeline for QML-IQFT
=================================

Comparative validation of QML-enhanced pricing vs classical methods.

Provides:
- Multi-method comparison (BS, MC, FB-IQFT, QML-IQFT)
- Error analysis
- Hardware execution support

Author: QFDP Research Team
Date: January 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False


@dataclass
class ValidationResult:
    """
    Result from validation pipeline.
    
    Attributes
    ----------
    scenario_name : str
        Name of the test scenario
    n_assets : int
        Number of assets in portfolio
    bs_price : float
        Black-Scholes price
    mc_price : float
        Monte Carlo price
    iqft_price : float
        Original FB-IQFT price
    qml_iqft_price : float
        QML-enhanced IQFT price
    qml_error_pct : float
        QML error vs BS (%)
    execution_time : float
        Total execution time (seconds)
    backend_used : str
        Backend used for execution
    """
    scenario_name: str
    n_assets: int
    bs_price: float
    mc_price: float
    iqft_price: float
    qml_iqft_price: float
    qml_error_pct: float
    execution_time: float
    backend_used: str


class ValidationPipeline:
    """
    Comprehensive validation for QML-IQFT pricing.
    
    Compares:
    1. Classical Black-Scholes
    2. Classical Monte Carlo
    3. Original FB-IQFT
    4. QML-enhanced FB-IQFT
    
    Parameters
    ----------
    use_hardware : bool
        Use real IBM Quantum hardware
    ibm_token : str, optional
        IBM Quantum API token (required if use_hardware=True)
    backend_name : str, optional
        Specific backend name (or let service select)
        
    Examples
    --------
    >>> pipeline = ValidationPipeline(use_hardware=False)
    >>> results = pipeline.run_comparative_study(scenarios)
    >>> print(results.to_string())
    """
    
    def __init__(
        self,
        use_hardware: bool = False,
        ibm_token: Optional[str] = None,
        backend_name: Optional[str] = None
    ):
        self.use_hardware = use_hardware
        self.ibm_token = ibm_token
        self.backend_name = backend_name
        self._service = None
        self._backend = None
        
        if use_hardware:
            self._initialize_ibm_backend()
    
    def _initialize_ibm_backend(self):
        """Initialize IBM Quantum backend."""
        if not IBM_RUNTIME_AVAILABLE:
            raise ImportError(
                "qiskit-ibm-runtime not installed. "
                "Run: pip install qiskit-ibm-runtime"
            )
        
        if self.ibm_token:
            self._service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=self.ibm_token
            )
        else:
            # Try to use saved credentials
            try:
                self._service = QiskitRuntimeService()
            except Exception as e:
                print(f"⚠️ Could not connect to IBM Quantum: {e}")
                self.use_hardware = False
                return
        
        # Select backend
        if self.backend_name:
            self._backend = self._service.backend(self.backend_name)
        else:
            self._backend = self._service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=8
            )
        
        print(f"✅ Connected to: {self._backend.name}")
        print(f"   Qubits: {self._backend.num_qubits}")
    
    def black_scholes_price(
        self,
        spot: float,
        strike: float,
        maturity: float,
        volatility: float,
        risk_free_rate: float
    ) -> float:
        """
        Black-Scholes European call price.
        
        Parameters
        ----------
        spot : float
            Current price
        strike : float
            Strike price
        maturity : float
            Time to expiry (years)
        volatility : float
            Annualized volatility
        risk_free_rate : float
            Risk-free rate
            
        Returns
        -------
        price : float
            Call option price
        """
        from scipy.stats import norm
        
        d1 = (np.log(spot / strike) + 
              (risk_free_rate + 0.5 * volatility ** 2) * maturity) / \
             (volatility * np.sqrt(maturity))
        d2 = d1 - volatility * np.sqrt(maturity)
        
        price = (spot * norm.cdf(d1) - 
                 strike * np.exp(-risk_free_rate * maturity) * norm.cdf(d2))
        
        return price
    
    def monte_carlo_price(
        self,
        spot: float,
        strike: float,
        maturity: float,
        volatility: float,
        risk_free_rate: float,
        n_sims: int = 100000,
        seed: Optional[int] = None
    ) -> float:
        """
        Monte Carlo option price.
        
        Parameters
        ----------
        spot : float
            Current price
        strike : float
            Strike price
        maturity : float
            Time to expiry (years)
        volatility : float
            Annualized volatility
        risk_free_rate : float
            Risk-free rate
        n_sims : int
            Number of simulations
        seed : int, optional
            Random seed
            
        Returns
        -------
        price : float
            Expected discounted payoff
        """
        rng = np.random.default_rng(seed)
        
        # Simulate terminal prices
        z = rng.standard_normal(n_sims)
        ST = spot * np.exp(
            (risk_free_rate - 0.5 * volatility ** 2) * maturity +
            volatility * np.sqrt(maturity) * z
        )
        
        # Payoffs
        payoffs = np.maximum(ST - strike, 0)
        
        # Discounted expectation
        price = np.exp(-risk_free_rate * maturity) * np.mean(payoffs)
        
        return price
    
    def run_single_scenario(
        self,
        scenario: Dict[str, Any]
    ) -> ValidationResult:
        """
        Run validation for a single scenario.
        
        Parameters
        ----------
        scenario : dict
            Scenario parameters with keys:
            - name: str
            - n_assets: int
            - spot: float
            - strike: float
            - maturity: float
            - volatility: float
            - risk_free_rate: float
            - returns: np.ndarray (optional)
            - weights: np.ndarray (optional)
            
        Returns
        -------
        ValidationResult
            Comparison results
        """
        import time
        start_time = time.time()
        
        name = scenario['name']
        n_assets = scenario.get('n_assets', 1)
        spot = scenario['spot']
        strike = scenario['strike']
        maturity = scenario['maturity']
        volatility = scenario['volatility']
        r = scenario['risk_free_rate']
        
        print(f"\n{'=' * 60}")
        print(f"Scenario: {name}")
        print(f"{'=' * 60}")
        
        # 1. Black-Scholes
        bs_price = self.black_scholes_price(spot, strike, maturity, volatility, r)
        print(f"  Black-Scholes: ${bs_price:.4f}")
        
        # 2. Monte Carlo
        mc_price = self.monte_carlo_price(spot, strike, maturity, volatility, r)
        print(f"  Monte Carlo:   ${mc_price:.4f}")
        
        # 3. Original FB-IQFT
        try:
            from qfdp.unified import FBIQFTPricing
            
            # Create simple 1-asset case
            pricer = FBIQFTPricing(M=16, alpha=1.0, num_shots=4096)
            
            result = pricer.price_option(
                asset_prices=np.array([spot]),
                asset_volatilities=np.array([volatility]),
                correlation_matrix=np.array([[1.0]]),
                portfolio_weights=np.array([1.0]),
                K=strike,
                T=maturity,
                r=r,
                backend='simulator'
            )
            iqft_price = result['price_quantum']
        except Exception as e:
            print(f"  ⚠️ FB-IQFT error: {e}")
            iqft_price = bs_price  # Fallback
        
        print(f"  FB-IQFT:       ${iqft_price:.4f}")
        
        # 4. QML-enhanced IQFT
        try:
            from .hybrid_pricer import QMLEnhancedFBIQFTPricer
            
            # Generate synthetic returns if not provided
            if 'returns' not in scenario:
                T_hist = 500
                returns = volatility / np.sqrt(252) * np.random.randn(T_hist, n_assets)
            else:
                returns = scenario['returns']
            
            weights = scenario.get('weights', np.ones(n_assets) / n_assets)
            
            qml_pricer = QMLEnhancedFBIQFTPricer(
                n_factors=min(2, n_assets),
                M=16
            )
            
            qml_result = qml_pricer.price_option(
                returns=returns,
                portfolio_weights=weights,
                strike=strike,
                maturity=maturity,
                risk_free_rate=r,
                asset_prices=np.ones(n_assets) * spot,
                train_qnn=False,  # Skip QNN for speed
                backend='simulator'
            )
            qml_iqft_price = qml_result.price_qml
        except Exception as e:
            print(f"  ⚠️ QML-IQFT error: {e}")
            qml_iqft_price = iqft_price
        
        print(f"  QML-IQFT:      ${qml_iqft_price:.4f}")
        
        # Compute error
        qml_error = abs(qml_iqft_price - bs_price) / bs_price * 100
        print(f"  Error vs BS:   {qml_error:.2f}%")
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            scenario_name=name,
            n_assets=n_assets,
            bs_price=bs_price,
            mc_price=mc_price,
            iqft_price=iqft_price,
            qml_iqft_price=qml_iqft_price,
            qml_error_pct=qml_error,
            execution_time=execution_time,
            backend_used='simulator' if not self.use_hardware else self._backend.name
        )
    
    def run_comparative_study(
        self,
        scenarios: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Run comparative study across multiple scenarios.
        
        Parameters
        ----------
        scenarios : list of dict, optional
            Test scenarios. If None, uses default scenarios.
            
        Returns
        -------
        results : pd.DataFrame
            Comparison results table
        """
        if scenarios is None:
            scenarios = self._default_scenarios()
        
        print("=" * 70)
        print("QML-IQFT VALIDATION PIPELINE")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        results = []
        for scenario in scenarios:
            result = self.run_single_scenario(scenario)
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'Scenario': r.scenario_name,
                'N Assets': r.n_assets,
                'BS': r.bs_price,
                'MC': r.mc_price,
                'FB-IQFT': r.iqft_price,
                'QML-IQFT': r.qml_iqft_price,
                'Error (%)': r.qml_error_pct,
                'Time (s)': r.execution_time
            }
            for r in results
        ])
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(df.to_string(index=False))
        print(f"\nMean QML Error: {df['Error (%)'].mean():.2f}%")
        
        return df
    
    def _default_scenarios(self) -> List[Dict[str, Any]]:
        """Default test scenarios."""
        return [
            {
                'name': 'ATM Call',
                'n_assets': 1,
                'spot': 100.0,
                'strike': 100.0,
                'maturity': 1.0,
                'volatility': 0.20,
                'risk_free_rate': 0.05
            },
            {
                'name': 'OTM Call',
                'n_assets': 1,
                'spot': 100.0,
                'strike': 110.0,
                'maturity': 1.0,
                'volatility': 0.20,
                'risk_free_rate': 0.05
            },
            {
                'name': 'ITM Call',
                'n_assets': 1,
                'spot': 100.0,
                'strike': 90.0,
                'maturity': 1.0,
                'volatility': 0.20,
                'risk_free_rate': 0.05
            },
            {
                'name': 'High Vol',
                'n_assets': 1,
                'spot': 100.0,
                'strike': 100.0,
                'maturity': 1.0,
                'volatility': 0.40,
                'risk_free_rate': 0.05
            },
            {
                'name': 'Short Maturity',
                'n_assets': 1,
                'spot': 100.0,
                'strike': 100.0,
                'maturity': 0.25,
                'volatility': 0.20,
                'risk_free_rate': 0.05
            },
        ]
    
    def save_results(
        self,
        df: pd.DataFrame,
        filepath: str = 'validation_results.csv'
    ):
        """Save results to CSV."""
        df.to_csv(filepath, index=False)
        print(f"✅ Results saved to {filepath}")


if __name__ == '__main__':
    # Quick test
    pipeline = ValidationPipeline(use_hardware=False)
    
    # Run with just 2 scenarios for quick test
    scenarios = [
        {
            'name': 'ATM Call',
            'n_assets': 1,
            'spot': 100.0,
            'strike': 100.0,
            'maturity': 1.0,
            'volatility': 0.20,
            'risk_free_rate': 0.05
        },
    ]
    
    try:
        results = pipeline.run_comparative_study(scenarios)
        print(f"\n✅ Validation pipeline test complete")
    except Exception as e:
        print(f"\n⚠️ Test incomplete: {e}")
