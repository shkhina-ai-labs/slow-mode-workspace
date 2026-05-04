"""
MetacognitiveController M — continuous per-token spectral-band selector.

Outputs a soft mask over the d_w eigenbands of the Koopman propagator K.
This is the SMW analog of AMOR's binary entropy gate, but continuous and
per-eigenband rather than binary architecture-switching.

P3 (cognition) prediction: the entropy of this mask correlates monotonically
with item difficulty, without saturation.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetacognitiveController(nn.Module):
    """
    Per-token controller emitting a soft mask over the workspace eigenbands.

    Args:
        d_model: residual-stream width (input)
        d_w:     workspace width / number of eigenbands (output)
        hidden:  controller MLP hidden width (default d_model // 4)
        gate:    'sigmoid' (independent per-band, default) or 'softmax' (top-k attention)

    For 'sigmoid' gate, mask entries are independent in [0, 1] — a continuous
    bandwidth dial.  Mean(mask) is the expected fraction of active bands.
    """

    def __init__(self, d_model: int, d_w: int, hidden: int | None = None, gate: str = "sigmoid"):
        super().__init__()
        if hidden is None:
            hidden = max(d_w * 2, d_model // 4)
        self.d_model = d_model
        self.d_w = d_w
        self.gate = gate

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_w),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> mask: (B, T, d_w)."""
        logits = self.head(x)
        if self.gate == "sigmoid":
            return torch.sigmoid(logits)
        if self.gate == "softmax":
            return F.softmax(logits, dim=-1) * self.d_w  # rescale so mean ~ 1
        raise ValueError(f"unknown gate {self.gate!r}")

    @staticmethod
    def mask_entropy(mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Bernoulli entropy of each mask entry, summed across bands, averaged across tokens.
        Diagnostic for P3.

        mask: (B, T, d_w) in [0, 1]
        returns scalar.
        """
        m = mask.clamp(eps, 1 - eps)
        H = -(m * m.log() + (1 - m) * (1 - m).log())  # (B, T, d_w)
        return H.sum(-1).mean()

    @staticmethod
    def expected_active_dims(mask: torch.Tensor) -> torch.Tensor:
        """Expected number of active bands per token (for FLOP accounting)."""
        return mask.sum(-1).mean()
