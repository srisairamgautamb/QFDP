"""
Feature Fusion Layer
====================

Combines QRC adaptive factors with QTC pattern features for enhanced pricing.

Fusion Methods:
    1. Concatenation: Simple stacking [F_qrc, P_qtc]
    2. Weighted: Learned importance weights
    3. Gating: Adaptive mixture based on market conditions
"""

import numpy as np
from typing import Optional, Dict, Literal
from dataclasses import dataclass


@dataclass
class FusionResult:
    """Result from feature fusion."""
    fused_features: np.ndarray
    qrc_weight: float
    qtc_weight: float
    fusion_method: str


class FeatureFusion:
    """
    Combine QRC factors and QTC patterns for enhanced pricing.
    
    The fusion layer learns to optimally combine regime-aware factors
    (from QRC) with temporal patterns (from QTC) for each market condition.
    
    Example:
        >>> fusion = FeatureFusion(method='weighted')
        >>> qrc_factors = np.array([0.3, 0.25, 0.25, 0.2])
        >>> qtc_patterns = np.array([0.4, 0.3, 0.2, 0.1])
        >>> result = fusion.forward(qrc_factors, qtc_patterns)
    """
    
    def __init__(
        self,
        n_qrc_factors: int = 4,
        n_qtc_patterns: int = 4,
        method: Literal['concat', 'weighted', 'gating'] = 'weighted'
    ):
        """
        Initialize fusion layer.
        
        Args:
            n_qrc_factors: Number of QRC factors
            n_qtc_patterns: Number of QTC pattern features
            method: Fusion strategy
                - 'concat': Simple concatenation
                - 'weighted': Learned importance weights
                - 'gating': Adaptive gating mechanism
        """
        self.n_qrc = n_qrc_factors
        self.n_qtc = n_qtc_patterns
        self.method = method
        
        # Learnable parameters
        if method == 'weighted':
            # Importance weights for each source
            self.w_qrc = np.ones(n_qrc_factors) / n_qrc_factors
            self.w_qtc = np.ones(n_qtc_patterns) / n_qtc_patterns
            self.alpha = 0.5  # Balance between QRC and QTC
        elif method == 'gating':
            # Gating network weights
            total_dim = n_qrc_factors + n_qtc_patterns
            self.gate_weights = np.random.randn(total_dim) * 0.1
    
    def forward(
        self,
        qrc_factors: np.ndarray,
        qtc_patterns: np.ndarray,
        market_context: Optional[Dict] = None
    ) -> FusionResult:
        """
        Fuse QRC and QTC features.
        
        Args:
            qrc_factors: Adaptive factors from QRC [F1, F2, F3, F4]
            qtc_patterns: Pattern features from QTC [p1, p2, ...]
            market_context: Optional market info for context-aware fusion
        
        Returns:
            FusionResult with fused features and weights
        """
        # Normalize both inputs
        qrc_norm = self._normalize(qrc_factors)
        qtc_norm = self._normalize(qtc_patterns)
        
        if self.method == 'concat':
            fused = np.concatenate([qrc_norm, qtc_norm])
            qrc_weight = 0.5
            qtc_weight = 0.5
            
        elif self.method == 'weighted':
            # Apply learned weights
            qrc_weighted = qrc_norm * self.w_qrc
            qtc_weighted = qtc_norm * self.w_qtc
            
            # Combine with learned balance
            fused_qrc = self.alpha * qrc_weighted
            fused_qtc = (1 - self.alpha) * qtc_weighted
            fused = np.concatenate([fused_qrc, fused_qtc])
            
            qrc_weight = self.alpha
            qtc_weight = 1 - self.alpha
            
        elif self.method == 'gating':
            # Compute gate values from combined input
            combined = np.concatenate([qrc_norm, qtc_norm])
            gate = self._sigmoid(self.gate_weights @ combined)
            
            # Apply gates
            fused_qrc = qrc_norm * gate
            fused_qtc = qtc_norm * (1 - gate)
            fused = np.concatenate([fused_qrc, fused_qtc])
            
            qrc_weight = gate
            qtc_weight = 1 - gate
        
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
        
        return FusionResult(
            fused_features=fused,
            qrc_weight=float(qrc_weight),
            qtc_weight=float(qtc_weight),
            fusion_method=self.method
        )
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """L2 normalize vector."""
        norm = np.linalg.norm(x)
        if norm > 1e-8:
            return x / norm
        return x
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def update_alpha(self, new_alpha: float) -> None:
        """Update QRC/QTC balance weight."""
        self.alpha = np.clip(new_alpha, 0, 1)
    
    def get_parameters(self) -> Dict:
        """Get learnable parameters."""
        if self.method == 'weighted':
            return {
                'w_qrc': self.w_qrc.copy(),
                'w_qtc': self.w_qtc.copy(),
                'alpha': self.alpha
            }
        elif self.method == 'gating':
            return {'gate_weights': self.gate_weights.copy()}
        return {}
    
    def set_parameters(self, params: Dict) -> None:
        """Set learnable parameters."""
        if self.method == 'weighted':
            if 'w_qrc' in params:
                self.w_qrc = params['w_qrc'].copy()
            if 'w_qtc' in params:
                self.w_qtc = params['w_qtc'].copy()
            if 'alpha' in params:
                self.alpha = params['alpha']
        elif self.method == 'gating':
            if 'gate_weights' in params:
                self.gate_weights = params['gate_weights'].copy()
