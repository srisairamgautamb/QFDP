# QML-IQFT: Quantum Machine Learning Enhanced Pricing

**Version**: 0.1.0  
**Date**: January 2026  
**Status**: Scaffolding Complete

---

## Overview

This module implements QML-enhanced Fourier-based quantum derivative pricing, combining neural network learning of characteristic functions with the proven FB-IQFT algorithm.

### Key Innovation

```
Classical PCA → QNN learns market CF → Quantum IQFT pricing
    ↓                    ↓                        ↓
 Factor Model        QML Magic            Proven Algorithm
```

---

## Module Structure

```
qfdp/qml_iqft/
├── __init__.py              # Package exports
├── data_pipeline.py         # Stock data collection via yfinance
├── factor_model.py          # PCA with risk-neutral adjustment
├── characteristic_function.py # Empirical CF computation
├── classical_nn.py          # PyTorch baseline (1-layer NN)
├── quantum_nn.py            # QNN: ZZFeatureMap + RealAmplitudes
├── hybrid_pricer.py         # Integration with FB-IQFT
└── validation.py            # Comparative testing pipeline
```

---

## Quick Start

### 1. Data Collection (Phase 1)

```python
from qfdp.qml_iqft import collect_stock_data, prepare_full_dataset

# Download 5 years of data for 5 stocks
data = prepare_full_dataset(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
    years=5,
    save_to_disk=True
)

print(f"Downloaded {len(data.prices)} days")
print(f"Correlation:\n{data.correlation_matrix}")
```

### 2. Factor Model (Phase 1)

```python
from qfdp.qml_iqft import PCAFactorModel

model = PCAFactorModel(n_factors=3)
result = model.fit(data.returns, risk_free_rate=0.05)

print(f"Variance explained: {result.total_variance_explained:.1%}")
```

### 3. QML-Enhanced Pricing (Phase 3-4)

```python
from qfdp.qml_iqft import QMLEnhancedFBIQFTPricer

pricer = QMLEnhancedFBIQFTPricer(n_factors=3, M=16)

result = pricer.price_option(
    returns=data.returns,
    portfolio_weights=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    strike=105,
    maturity=1.0,
    risk_free_rate=0.05
)

print(f"QML Price: ${result.price_qml:.2f}")
print(f"Error: {result.error_vs_classical:.2f}%")
```

### 4. Validation (Phase 5)

```python
from qfdp.qml_iqft import ValidationPipeline

pipeline = ValidationPipeline(use_hardware=False)
results = pipeline.run_comparative_study()

print(results.to_string())
```

---

## Dependencies

Add to `requirements.txt`:

```
yfinance>=0.2.0
qiskit-machine-learning>=0.7.0
torch>=2.0.0
```

---

## Implementation Phases

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Data Pipeline | ✅ Complete |
| 1 | Factor Model | ✅ Complete |
| 2 | Classical NN | ✅ Complete |
| 3 | Quantum NN | ✅ Complete |
| 4 | Hybrid Pricer | ✅ Complete |
| 5 | Validation | ✅ Complete |
| - | IBM Q Testing | ⏳ Pending API |

---

## IBM Quantum Integration

When ready for hardware testing, initialize with API token:

```python
from qfdp.qml_iqft import ValidationPipeline

pipeline = ValidationPipeline(
    use_hardware=True,
    ibm_token="YOUR_API_TOKEN"
)

results = pipeline.run_comparative_study()
```

---

## References

- **Theoretical Foundation**: QML_QHDP.pdf (Sections 5-6)
- **Proven Baseline**: FB-IQFT (`qfdp.fb_iqft.pricing`)
- **Original QFDP**: `qfdp.unified`
