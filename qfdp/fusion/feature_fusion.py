"""
Feature Fusion Layer
====================

Combines QRC adaptive factors with QTC temporal patterns.

Methods:
- concat: Simple concatenation [QRC, QTC] → 8 features
- weighted: α*QRC + β*QTC → 4 features
- gating: Learned gating mechanism
"""

import numpy as np
from typing import Literal
import logging

logger = logging.getLogger(__name__)


class FeatureFusion:
    """
    Combine QRC adaptive factors with QTC pattern features.
    
    Input:
    - QRC factors: [F₁, F₂, F₃, F₄] (adaptive correlation factors)
    - QTC patterns: [p₁, p₂, p₃, p₄] (temporal patterns)
    
    Output:
    - Fused features: Combined representation
    """
    
    def __init__(self, method: Literal['concat', 'weighted', 'gating'] = 'weighted'):
        """
        Args:
            method: 'concat' | 'weighted' | 'gating'
        """
        self.method = method
        
        if method == 'weighted':
            self.alpha = 0.6  # Weight for QRC (correlation regime)
            self.beta = 0.4   # Weight for QTC (temporal patterns)
        elif method == 'gating':
            # Learnable gating weights
            self.gate_qrc = 0.5
            self.gate_qtc = 0.5
        
        logger.info(f"FeatureFusion initialized with method={method}")
    
    def forward(self, qrc_factors: np.ndarray, qtc_patterns: np.ndarray) -> np.ndarray:
        """
        Fuse QRC and QTC features.
        
        Args:
            qrc_factors: Shape (4,) - from QRC
            qtc_patterns: Shape (4,) - from QTC
        
        Returns:
            fused: Shape (8,) for concat, (4,) for weighted/gating
        """
        # Normalize both
        qrc_norm = qrc_factors / (np.linalg.norm(qrc_factors) + 1e-8)
        qtc_norm = qtc_patterns / (np.linalg.norm(qtc_patterns) + 1e-8)
        
        if self.method == 'concat':
            # Simple concatenation → 8 features
            fused = np.concatenate([qrc_norm, qtc_norm])
        
        elif self.method == 'weighted':
            # Weighted average → 4 features
            fused = self.alpha * qrc_norm + self.beta * qtc_norm
            # Re-normalize
            fused = fused / (np.sum(fused) + 1e-8)
        
        elif self.method == 'gating':
            # Element-wise gating
            gate = self._compute_gate(qrc_factors, qtc_patterns)
            fused = gate * qrc_norm + (1 - gate) * qtc_norm
            fused = fused / (np.sum(fused) + 1e-8)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return fused
    
    def _compute_gate(self, qrc: np.ndarray, qtc: np.ndarray) -> np.ndarray:
        """Compute adaptive gate based on feature magnitudes."""
        qrc_energy = np.sum(qrc**2)
        qtc_energy = np.sum(qtc**2)
        
        # Dynamic gating: higher energy features get more weight
        total_energy = qrc_energy + qtc_energy + 1e-8
        gate = qrc_energy / total_energy
        
        return gate
    
    def compute_volatility_adjustment(
        self, 
        qrc_factors: np.ndarray, 
        qtc_patterns: np.ndarray,
        sigma_p_base: float
    ) -> float:
        """
        Compute volatility adjustment based on fused features.
        
        Args:
            qrc_factors: QRC adaptive factors
            qtc_patterns: QTC temporal patterns
            sigma_p_base: Base portfolio volatility
        
        Returns:
            sigma_p_adjusted: Adjusted volatility
        """
        # Fuse features
        fused = self.forward(qrc_factors, qtc_patterns)
        
        # Compute adjustment factor
        # Higher first feature → trend detected → lower vol adjustment
        # Higher last feature → uncertainty → higher vol adjustment
        
        if self.method == 'concat':
            # Use first 4 (QRC) for regime, last 4 (QTC) for patterns
            regime_factor = 1 + 0.1 * (fused[0] - 0.25)  # QRC dominant factor
            pattern_factor = 1 + 0.05 * (fused[-1] - 0.25)  # QTC recent pattern
            adjustment = regime_factor * pattern_factor
        else:
            # Weighted/gating: use fused features directly
            concentration = np.max(fused) - np.mean(fused)
            adjustment = 1 + 0.1 * concentration
        
        sigma_p_adjusted = sigma_p_base * adjustment
        
        return sigma_p_adjusted


# ============================================================================
# VALIDATION
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("FEATURE FUSION VALIDATION")
    print("=" * 60)
    
    # Test data
    qrc = np.array([0.3, 0.3, 0.2, 0.2])  # QRC factors
    qtc = np.array([0.4, 0.3, 0.2, 0.1])  # QTC patterns
    
    # Test concat
    print("\n--- Concat Method ---")
    fusion_concat = FeatureFusion(method='concat')
    fused_concat = fusion_concat.forward(qrc, qtc)
    print(f"QRC: {qrc}")
    print(f"QTC: {qtc}")
    print(f"Fused (concat): {fused_concat} (len={len(fused_concat)})")
    assert len(fused_concat) == 8, "Concat should give 8 features"
    print("✅ Concat OK")
    
    # Test weighted
    print("\n--- Weighted Method ---")
    fusion_weighted = FeatureFusion(method='weighted')
    fused_weighted = fusion_weighted.forward(qrc, qtc)
    print(f"Fused (weighted): {fused_weighted} (len={len(fused_weighted)})")
    assert len(fused_weighted) == 4, "Weighted should give 4 features"
    print("✅ Weighted OK")
    
    # Test gating
    print("\n--- Gating Method ---")
    fusion_gating = FeatureFusion(method='gating')
    fused_gating = fusion_gating.forward(qrc, qtc)
    print(f"Fused (gating): {fused_gating} (len={len(fused_gating)})")
    assert len(fused_gating) == 4, "Gating should give 4 features"
    print("✅ Gating OK")
    
    # Test volatility adjustment
    print("\n--- Volatility Adjustment ---")
    sigma_base = 0.20
    sigma_adjusted = fusion_weighted.compute_volatility_adjustment(qrc, qtc, sigma_base)
    print(f"σ_base: {sigma_base}")
    print(f"σ_adjusted: {sigma_adjusted:.4f}")
    print(f"Adjustment: {(sigma_adjusted/sigma_base - 1)*100:.2f}%")
    print("✅ Volatility adjustment OK")
    
    print("\n✅ FEATURE FUSION VALIDATED")
