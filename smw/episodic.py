"""
EpisodicLedger E — Larimar-style sparse-distributed memory in workspace space.

Operates entirely in the d_w slow-mode subspace, NOT in the full d_model
residual stream.  This is the structural commitment that distinguishes SMW
from prior episodic-memory work: consolidation lives INSIDE the Koopman
slow-mode subspace (commitment 2 of synthesis 04b).

Reads: top-k content-addressable attention over slots.
Writes: novelty-gated one-shot writes to least-recently-used slots.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicLedger(nn.Module):
    """
    A bank of n_slots key-value pairs, each of dimension d_w.

    Args:
        n_slots: number of memory slots
        d_w:     workspace dimension (== projector output width)
        top_k:   how many slots to read per query
    """

    def __init__(self, n_slots: int = 256, d_w: int = 64, top_k: int = 8):
        super().__init__()
        self.n_slots = n_slots
        self.d_w = d_w
        self.top_k = top_k

        # Persistent (non-trainable) memory state.
        self.register_buffer("keys", torch.zeros(n_slots, d_w))
        self.register_buffer("values", torch.zeros(n_slots, d_w))
        # Recency timestamps (lower = older).
        self.register_buffer("last_used", torch.zeros(n_slots, dtype=torch.long))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def reset(self):
        self.keys.zero_()
        self.values.zero_()
        self.last_used.zero_()
        self.step.zero_()

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Top-k content-addressable read.

        query: (B, T, d_w) -> retrieved: (B, T, d_w)
        """
        # Cosine attention.
        q = F.normalize(query, dim=-1)         # (B, T, d_w)
        k = F.normalize(self.keys, dim=-1)     # (n_slots, d_w)
        scores = q @ k.T                       # (B, T, n_slots)

        topk = min(self.top_k, self.n_slots)
        top_scores, top_idx = scores.topk(topk, dim=-1)  # (B, T, topk)
        weights = F.softmax(top_scores, dim=-1)          # (B, T, topk)

        # Gather the matching values.
        # values: (n_slots, d_w) -> (B, T, topk, d_w)
        gathered = self.values[top_idx]
        retrieved = (weights.unsqueeze(-1) * gathered).sum(dim=-2)  # (B, T, d_w)

        # Update recency for read slots.
        with torch.no_grad():
            self.step += 1
            flat_idx = top_idx.reshape(-1)
            self.last_used[flat_idx] = self.step

        return retrieved

    @torch.no_grad()
    def write(self, w: torch.Tensor, novelty_gate: torch.Tensor, novelty_threshold: float = 0.5):
        """
        Novelty-gated one-shot writes.

        w:            (B, T, d_w)        -- workspace activations to potentially store
        novelty_gate: (B, T)              -- per-token scalar in [0, 1] (e.g. surprise)
        novelty_threshold: only writes for tokens with gate >= threshold
        """
        B, T, _ = w.shape
        flat_w = w.reshape(B * T, self.d_w)
        flat_gate = novelty_gate.reshape(B * T)

        novel_mask = flat_gate >= novelty_threshold
        if not novel_mask.any():
            return

        novel_w = flat_w[novel_mask]              # (n_novel, d_w)
        n_novel = novel_w.size(0)

        # If we have more novel tokens than slots, keep the most novel ones (top by gate).
        if n_novel > self.n_slots:
            top_gate_vals, top_gate_idx = flat_gate[novel_mask].topk(self.n_slots, largest=True)
            novel_w = novel_w[top_gate_idx]
            n_novel = self.n_slots

        # Pick the n_novel least-recently-used slots.
        _, lru_idx = self.last_used.topk(n_novel, largest=False)

        self.keys[lru_idx] = novel_w
        self.values[lru_idx] = novel_w
        self.step += 1
        self.last_used[lru_idx] = self.step
