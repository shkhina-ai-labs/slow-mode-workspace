"""
SMWConfig — model + ablation conditions.

The 5+1 conditions from the pre-registered experiment (05-experiment-design.md):

  C0  baseline:                       no SMW components
  C1  GW-only:                        workspace bottleneck only
  C2  E-only:                         episodic ledger only
  C3  RG-only:                        scale-equivariance constraint only
  C4  All-Independent:                all three, but with SEPARATE operators
  C5  SMW (shared):                   all three, with ONE shared S operator

The critical comparison is C5 vs. C4.  Synergy term = (C5 − C0) − Σ_i (C_i − C0).
If synergy ≤ 0, the shared-operator identity is unnecessary and H1 is rejected.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace


@dataclass
class SMWConfig:
    # --- core transformer ---
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    vocab_size: int = 1024
    block_size: int = 256
    dropout: float = 0.0

    # --- SMW-specific ---
    d_w: int = 16          # workspace width
    k_bottleneck: int = 2  # insert S every k blocks
    n_episodic_slots: int = 128
    episodic_top_k: int = 4

    # --- ablation flags ---
    use_workspace_bottleneck: bool = True   # turns on the S projector
    use_episodic_memory: bool = True        # turns on the ledger E
    use_rg_constraint: bool = True          # turns on the scale-equivariance loss
    share_operator: bool = True             # CRITICAL: True for SMW, False for All-Indep

    # --- regularizer weights ---
    w_stiefel: float = 1e-2
    w_scale: float = 1e-3
    w_contract: float = 1e-2

    # --- controller ---
    metacognitive_gate: str = "sigmoid"  # 'sigmoid' or 'softmax'

    # --- bookkeeping ---
    name: str = "smw"


CONDITIONS = {
    "C0_baseline": SMWConfig(
        name="C0_baseline",
        use_workspace_bottleneck=False,
        use_episodic_memory=False,
        use_rg_constraint=False,
        share_operator=False,
    ),
    "C1_gw_only": SMWConfig(
        name="C1_gw_only",
        use_workspace_bottleneck=True,
        use_episodic_memory=False,
        use_rg_constraint=False,
        share_operator=False,
    ),
    "C2_episodic_only": SMWConfig(
        name="C2_episodic_only",
        use_workspace_bottleneck=False,
        use_episodic_memory=True,
        use_rg_constraint=False,
        share_operator=False,
    ),
    "C3_rg_only": SMWConfig(
        name="C3_rg_only",
        use_workspace_bottleneck=False,
        use_episodic_memory=False,
        use_rg_constraint=True,
        share_operator=False,
    ),
    "C4_all_independent": SMWConfig(
        name="C4_all_independent",
        use_workspace_bottleneck=True,
        use_episodic_memory=True,
        use_rg_constraint=True,
        share_operator=False,
    ),
    "C5_smw": SMWConfig(
        name="C5_smw",
        use_workspace_bottleneck=True,
        use_episodic_memory=True,
        use_rg_constraint=True,
        share_operator=True,
    ),
}
