"""Slow-Mode Workspace (SMW) — reference implementation."""

from .projector import SlowModeProjector
from .metacognitive import MetacognitiveController
from .episodic import EpisodicLedger
from .model import SMWModel, SMWSite, Block
from .configs import SMWConfig, CONDITIONS

__version__ = "0.0.1"
__all__ = [
    "SlowModeProjector",
    "MetacognitiveController",
    "EpisodicLedger",
    "SMWModel",
    "SMWSite",
    "Block",
    "SMWConfig",
    "CONDITIONS",
]
