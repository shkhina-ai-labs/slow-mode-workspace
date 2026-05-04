"""
Smoke test — verifies all 5+1 conditions instantiate, do a forward pass,
and that the SMW config produces a non-trivial regularizer.
"""

import torch

from smw import SMWModel, CONDITIONS
from smw.metacognitive import MetacognitiveController


def _forward(cfg):
    model = SMWModel(cfg)
    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss, reg = model(idx, targets)
    return model, logits, loss, reg


def test_all_conditions_forward():
    """Every ablation condition should run a forward pass without error."""
    for name, cfg in CONDITIONS.items():
        # shrink for speed
        cfg.d_model = 64
        cfg.n_layers = 4
        cfg.d_w = 8
        cfg.block_size = 32
        model, logits, loss, reg = _forward(cfg)
        assert logits.shape == (2, 16, cfg.vocab_size), f"{name}: wrong shape"
        assert loss.isfinite(), f"{name}: non-finite loss"
        assert reg.isfinite(), f"{name}: non-finite regularizer"


def test_smw_has_shared_operator():
    """In SMW (C5), the projector should be SHARED across all sites."""
    cfg = CONDITIONS["C5_smw"]
    cfg.d_model = 64
    cfg.n_layers = 4
    cfg.d_w = 8
    cfg.block_size = 32
    model = SMWModel(cfg)

    # Collect all projector ids.
    pids = set()
    for site in model.sites:
        if hasattr(site, "projector") and site.projector is not None:
            pids.add(id(site.projector))
    assert len(pids) == 1, f"Expected 1 shared projector in SMW, got {len(pids)}"


def test_all_independent_has_separate_operators():
    """In C4 (All-Independent), each site should own its own projector."""
    cfg = CONDITIONS["C4_all_independent"]
    cfg.d_model = 64
    cfg.n_layers = 4
    cfg.d_w = 8
    cfg.block_size = 32
    model = SMWModel(cfg)

    pids = set()
    for site in model.sites:
        if hasattr(site, "projector") and site.projector is not None:
            pids.add(id(site.projector))
    # k_bottleneck=2, n_layers=4 -> 2 sites, 2 distinct projectors.
    assert len(pids) >= 2, f"Expected multiple projectors in C4, got {len(pids)}"


def test_regularizer_drops_when_constraints_off():
    """Baseline (C0) should have a near-zero regularizer."""
    cfg = CONDITIONS["C0_baseline"]
    cfg.d_model = 64
    cfg.n_layers = 4
    cfg.d_w = 8
    cfg.block_size = 32
    _, _, _, reg = _forward(cfg)
    assert float(reg) < 1e-6, f"baseline reg should be ~0, got {float(reg)}"


def test_regularizer_active_when_smw_on():
    """SMW (C5) should have a non-trivial regularizer."""
    cfg = CONDITIONS["C5_smw"]
    cfg.d_model = 64
    cfg.n_layers = 4
    cfg.d_w = 8
    cfg.block_size = 32
    _, _, _, reg = _forward(cfg)
    assert float(reg) > 0, f"SMW reg should be > 0, got {float(reg)}"


def test_mask_entropy_diagnostic():
    """The P3 diagnostic should produce a finite scalar after a forward pass."""
    cfg = CONDITIONS["C5_smw"]
    cfg.d_model = 64
    cfg.n_layers = 4
    cfg.d_w = 8
    cfg.block_size = 32
    model = SMWModel(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    model(idx)
    H = model.last_mask_entropy()
    assert H is not None, "expected mask entropy diagnostic to be available"
    assert 0 <= H, f"entropy must be non-negative, got {H}"


def test_slow_band_mass_diagnostic():
    """P1 diagnostic should produce a fraction in [0, 1]."""
    cfg = CONDITIONS["C5_smw"]
    cfg.d_model = 64
    cfg.n_layers = 4
    cfg.d_w = 8
    cfg.block_size = 32
    model = SMWModel(cfg)
    m = model.slow_band_mass()
    assert m is not None
    assert 0.0 <= m <= 1.0, f"slow band mass must be in [0,1], got {m}"


if __name__ == "__main__":
    test_all_conditions_forward()
    test_smw_has_shared_operator()
    test_all_independent_has_separate_operators()
    test_regularizer_drops_when_constraints_off()
    test_regularizer_active_when_smw_on()
    test_mask_entropy_diagnostic()
    test_slow_band_mass_diagnostic()
    print("OK - all smoke tests passed.")
