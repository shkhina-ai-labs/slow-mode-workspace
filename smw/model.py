"""
SMWModel — decoder-only Transformer with the Slow-Mode Workspace retrofit.

Architecture (when all flags are on):
  tokens -> embed
         -> Block_1 -> ... -> Block_k -> [ S (shared, weight-tied) ]
                                          ^
                                          | workspace bottleneck + episodic R/W +
                                          | metacognitive spectral mask
                                          v
                                         lift back into residual stream
         -> Block_{k+1} -> ... -> Block_N -> output

The key knob is `config.share_operator`:
  True  -> SMW: ONE SlowModeProjector instance, weight-tied across all bottleneck sites.
  False -> All-Independent ablation: a fresh SlowModeProjector at each site.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import SMWConfig
from .projector import SlowModeProjector
from .metacognitive import MetacognitiveController
from .episodic import EpisodicLedger


# ---------- standard decoder block (no SMW components) -----------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: SMWConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # scaled-dot-product with causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: SMWConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.fc2 = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: SMWConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------- SMW bottleneck site ---------------------------------------------

class SMWSite(nn.Module):
    """
    A single bottleneck insertion point.  Owns (or shares) S, M, E.

    If `cfg.share_operator` is True, callers should pass shared instances of
    projector / controller / ledger.  Otherwise, fresh ones are created.
    """

    def __init__(
        self,
        cfg: SMWConfig,
        projector: SlowModeProjector | None = None,
        controller: MetacognitiveController | None = None,
        ledger: EpisodicLedger | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.use_workspace = cfg.use_workspace_bottleneck
        self.use_episodic = cfg.use_episodic_memory

        if self.use_workspace:
            self.projector = projector or SlowModeProjector(cfg.d_model, cfg.d_w)
            self.controller = controller or MetacognitiveController(cfg.d_model, cfg.d_w, gate=cfg.metacognitive_gate)
        else:
            self.projector = None
            self.controller = None

        if self.use_episodic:
            self.ledger = ledger or EpisodicLedger(cfg.n_episodic_slots, cfg.d_w, cfg.episodic_top_k)
        else:
            self.ledger = None

        self.last_mask = None  # for diagnostics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projector is None:
            return x  # no-op when ablated

        # Project into workspace.
        w = self.projector.project(x)                  # (B, T, d_w)
        w = self.projector.propagate(w)                # apply Koopman propagator

        # Per-token spectral-band mask (continuous selector).
        mask = self.controller(x)                     # (B, T, d_w)
        w = w * mask
        self.last_mask = mask.detach()

        # Episodic R/W in workspace space.
        if self.ledger is not None:
            retrieved = self.ledger.read(w)           # (B, T, d_w)
            w = w + retrieved
            # Use mask sparsity as a novelty proxy: tokens with high active-dim count are novel.
            novelty_gate = mask.mean(dim=-1)           # (B, T) in [0, 1]
            self.ledger.write(w, novelty_gate=novelty_gate)

        # Lift back into residual stream and add as residual.
        return x + self.projector.lift(w)


# ---------- the full model --------------------------------------------------

class SMWModel(nn.Module):
    """
    Decoder-only Transformer with optional SMW retrofit.

    The bottleneck site is inserted every `cfg.k_bottleneck` blocks.
    """

    def __init__(self, cfg: SMWConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Build (optionally shared) SMW components.
        if cfg.use_workspace_bottleneck and cfg.share_operator:
            self.shared_projector = SlowModeProjector(cfg.d_model, cfg.d_w)
            self.shared_controller = MetacognitiveController(cfg.d_model, cfg.d_w, gate=cfg.metacognitive_gate)
        else:
            self.shared_projector = None
            self.shared_controller = None

        if cfg.use_episodic_memory and cfg.share_operator:
            self.shared_ledger = EpisodicLedger(cfg.n_episodic_slots, cfg.d_w, cfg.episodic_top_k)
        else:
            self.shared_ledger = None

        # Build blocks and sites.
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # SMW sites: one after every k blocks.
        self.sites = nn.ModuleList()
        for layer_idx in range(cfg.n_layers):
            insert_here = (layer_idx + 1) % cfg.k_bottleneck == 0 and (
                cfg.use_workspace_bottleneck or cfg.use_episodic_memory
            )
            if insert_here:
                self.sites.append(
                    SMWSite(
                        cfg,
                        projector=self.shared_projector,
                        controller=self.shared_controller,
                        ledger=self.shared_ledger,
                    )
                )
            else:
                self.sites.append(nn.Identity())

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie input/output embeddings.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])

        for block, site in zip(self.blocks, self.sites):
            x = block(x)
            x = site(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        reg_loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            reg_loss = self.regularizer(x)
        return logits, loss, reg_loss

    # ---- regularizers (the 3 commitments) ----

    def regularizer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sum of S's three constraint losses across all SMW sites.

        Stiefel + contractivity always; scale-equivariance only if cfg.use_rg_constraint.
        """
        cfg = self.cfg
        total = x.new_zeros(())
        seen = set()
        for site in self.sites:
            if isinstance(site, nn.Identity) or site.projector is None:
                continue
            # In the shared-operator case (SMW), the same projector appears at every site
            # — apply the regularizer once, not n_sites times.
            pid = id(site.projector)
            if pid in seen:
                continue
            seen.add(pid)
            x_for_eq = x if cfg.use_rg_constraint else None
            total = total + site.projector.total_regularizer(
                x=x_for_eq,
                w_stiefel=cfg.w_stiefel,
                w_scale=cfg.w_scale,
                w_contract=cfg.w_contract,
            )
        return total

    # ---- diagnostics for P1, P3 ----

    @torch.no_grad()
    def slow_band_mass(self, threshold: float = 0.95) -> float | None:
        """P1 diagnostic — only meaningful when share_operator=True."""
        if self.shared_projector is None:
            return None
        return float(self.shared_projector.slow_band_mass(threshold))

    @torch.no_grad()
    def last_mask_entropy(self) -> float | None:
        """P3 diagnostic — entropy of the most recent metacognitive mask."""
        for site in self.sites:
            if isinstance(site, nn.Identity):
                continue
            if site.last_mask is not None:
                return float(MetacognitiveController.mask_entropy(site.last_mask))
        return None

    @torch.no_grad()
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
