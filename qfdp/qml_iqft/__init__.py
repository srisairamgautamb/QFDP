"""
QML-IQFT: Quantum Machine Learning Enhanced FB-IQFT Pricing
============================================================

This module implements QML-enhanced Fourier-based quantum derivative pricing,
combining neural network learning of characteristic functions with the
proven FB-IQFT algorithm.

Key Components:
- Data Pipeline: Stock data collection and preprocessing
- Factor Model: PCA decomposition with risk-neutral adjustment
- Classical NN: Baseline neural network for comparison
- Quantum NN: QNN for learning characteristic functions
- Hybrid Pricer: Integration of QNN with FB-IQFT
- Accurate Pricer: Production-quality pricing with <1% error

Mathematical Framework (from QML_QHDP.pdf):
-------------------------------------------
1. Classical PCA: Reduce N assets to K factors
2. QNN learns market-implied characteristic function φ_QNN(u; θ)
3. Enhanced FB-IQFT uses φ_QNN instead of analytical φ
4. NISQ-feasible pricing with learned market dynamics

Author: QFDP Research Team
Date: January 2026
"""

__version__ = "0.1.0"

# Data Pipeline
from .data_pipeline import (
    collect_stock_data,
    compute_log_returns,
    analyze_correlations,
    StockData,
)

# Factor Model
from .factor_model import (
    PCAFactorModel,
    FactorModelResult,
)

# Characteristic Function
from .characteristic_function import (
    compute_empirical_cf,
    prepare_qnn_training_data,
    EmpiricalCFResult,
)

# Classical Neural Network
from .classical_nn import (
    ClassicalFactorPricingNN,
    train_classical_baseline,
)

# Quantum Neural Network
from .quantum_nn import (
    QuantumCharacteristicFunctionLearner,
    QNNTrainingResult,
)

# Hybrid Pricer
from .hybrid_pricer import (
    QMLEnhancedFBIQFTPricer,
    QMLPricingResult,
)

# Validation
from .validation import (
    ValidationPipeline,
    ValidationResult,
)

# Accurate Pricer (Production-quality with <1% error)
from .accurate_pricer import (
    AccurateQMLPricer,
    AccuratePricingResult,
    run_accuracy_test,
)

__all__ = [
    # Data Pipeline
    'collect_stock_data',
    'compute_log_returns',
    'analyze_correlations',
    'StockData',
    # Factor Model
    'PCAFactorModel',
    'FactorModelResult',
    # Characteristic Function
    'compute_empirical_cf',
    'prepare_qnn_training_data',
    'EmpiricalCFResult',
    # Classical NN
    'ClassicalFactorPricingNN',
    'train_classical_baseline',
    # Quantum NN
    'QuantumCharacteristicFunctionLearner',
    'QNNTrainingResult',
    # Hybrid Pricer
    'QMLEnhancedFBIQFTPricer',
    'QMLPricingResult',
    # Validation
    'ValidationPipeline',
    'ValidationResult',
    # Accurate Pricer
    'AccurateQMLPricer',
    'AccuratePricingResult',
    'run_accuracy_test',
]
