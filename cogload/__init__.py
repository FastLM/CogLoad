"""
Cognitive Load Traces (CLT) Framework
=====================================

A mid-level interpretability framework for deep models inspired by Cognitive Load Theory.

This package implements:
- CLT computation for transformer models
- Load-guided decoding interventions
- Visualization tools for interpretability
"""

from .clt import CognitiveLoadTraces
from .visualization import CLTVisualizer
from .interventions import LoadGuidedDecoding

__version__ = "0.1.0"
__all__ = ["CognitiveLoadTraces", "CLTVisualizer", "LoadGuidedDecoding"]

