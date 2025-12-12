# QFDP Project Restructuring Plan

## Current State (Messy)
- 25 Python files in root directory
- Multiple demo/test files scattered
- Unclear structure
- QFDP_base_model as separate folder

## Target Structure (Clean)

```
QFDP/
├── README.md                    # Main project README
├── requirements.txt             # Dependencies
├── setup.py                     # Package installation
│
├── qfdp/                        # Main package (consolidated)
│   ├── __init__.py
│   │
│   ├── core/                    # Core implementations
│   │   ├── __init__.py
│   │   ├── sparse_copula.py    # From qfdp_multiasset
│   │   ├── state_prep.py       # Quantum state preparation
│   │   ├── mlqae.py            # MLQAE implementation
│   │   ├── oracles.py          # Payoff oracles
│   │   ├── iqft.py             # Enhanced IQFT
│   │   └── hardware.py         # IBM Quantum integration
│   │
│   ├── fb_iqft/                # FB-IQFT breakthrough (NEW)
│   │   ├── __init__.py
│   │   ├── factor_char_func.py
│   │   ├── circuit.py
│   │   └── pricing.py
│   │
│   ├── portfolio/              # Portfolio management
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   └── risk.py             # VaR/CVaR
│   │
│   ├── market_data/            # Data connectors
│   │   ├── __init__.py
│   │   └── alphavantage.py
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       └── reproducibility.py
│
├── examples/                   # Demo scripts (organized)
│   ├── basic/
│   │   ├── single_asset.py
│   │   └── simple_portfolio.py
│   ├── advanced/
│   │   ├── multiasset_pricing.py
│   │   └── var_cvar_demo.py
│   └── breakthrough/
│       └── fb_iqft_demo.py     # The breakthrough
│
├── tests/                      # All tests
│   ├── test_sparse_copula.py
│   ├── test_mlqae.py
│   ├── test_iqft.py
│   ├── test_fb_iqft.py
│   ├── test_hardware.py
│   └── test_system.py
│
├── docs/                       # Documentation
│   ├── FB_IQFT_BREAKTHROUGH.md
│   ├── CONSOLIDATION_COMPLETE.md
│   ├── HONEST_STATUS.md
│   └── architecture.md
│
├── notebooks/                  # Jupyter notebooks (if any)
│
├── archive/                    # Old/backup code
│   ├── QFDP_base_model/       # Move here
│   └── deprecated/
│
└── .gitignore
```

## Files to Keep (Essential)

### Core Implementation
- qfdp_multiasset/* → qfdp/core/
- unified_qfdp/* → qfdp/fb_iqft/

### Tests (Keep Best Ones)
- test_complete_system.py → tests/test_system.py
- test_ibm_hardware.py → tests/test_hardware.py
- test_enhanced_iqft.py → tests/test_iqft.py
- test_mlqae_k_greater_than_zero.py → tests/test_mlqae.py

### Demos (Keep Key Ones)
- demo_fb_iqft_breakthrough.py → examples/breakthrough/
- demo_real_var_cvar.py → examples/advanced/
- demo_single_asset.py → examples/basic/

### Documentation
- FB_IQFT_BREAKTHROUGH.md → docs/
- CONSOLIDATION_COMPLETE.md → docs/
- HONEST_STATUS.md → docs/
- README.md (keep and update)

## Files to Archive

### Move to archive/
- QFDP_base_model/ (entire folder - reference only)
- All old demo files except key ones
- deprecated test files
- Old documentation (PHASE_*.md, etc.)

## Files to Delete

### Temporary/Generated
- __pycache__/
- *.pyc
- .DS_Store
- ._* (macOS resource forks)

### Duplicate/Obsolete
- Multiple similar demo files
- Old test files that are superseded
- Redundant documentation

## Implementation Steps

1. Create new structure
2. Copy/move essential files to new locations
3. Update imports in all files
4. Archive old QFDP_base_model
5. Clean up root directory
6. Test everything works
7. Update README with new structure
