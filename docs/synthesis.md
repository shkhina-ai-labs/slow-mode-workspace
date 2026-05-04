---
title: "Synthesis (Narrowed) — The Slow-Mode Workspace, honestly positioned"
type: experiment-synthesis-narrowed
created: 2026-05-04
status: hypothesis-narrowed
supersedes: 04-synthesis.md
---

# Synthesis (Narrowed) — The Slow-Mode Workspace (SMW)

After Round 2 of the panel debate against three newly-found close neighbors (Inhibitory Cross-Talk 2603.03355, AMOR 2602.13215, SleepGate 2603.14517), the synthesis is narrowed and honestly positioned.

## Honest concessions (what is NOT novel)

- "Single operator, multiple roles in a Transformer" — **claimed by Inhibitory Cross-Talk (2603.03355, Feb 2026)** with the phrase "attention serves simultaneously as retrieval, consolidation, and write-back operator." We do not own that framing.
- "Per-token metacognitive gating routing computation" — **claimed by AMOR (2602.13215, Jan 2026)**.
- "Sleep-inspired memory consolidation in Transformers" — **claimed by SleepGate (2603.14517) and the 2023 NeurIPS NMDAR paper**.
- "Hippocampal/CLS-style memory hybrids" — **claimed by Larimar, EM-LLM, Pink et al., Princeton CompMem 2025**.

## What survives as novel

The novel contribution is a **specific instantiation** with three precise commitments that no prior work makes jointly:

> **The Slow-Mode Workspace (SMW)** — a Koopman-spectral instantiation of the latent-memory framing of [Inhibitory Cross-Talk (2603.03355)](https://arxiv.org/abs/2603.03355), extended with a renormalization-group role and a continuous per-token spectral-band selector.

Three structural commitments distinguishing SMW from prior work:

### Commitment 1 — Koopman projector with three constraints

Replace the Gram-matrix `A^T A` of latent-memory architectures with a **single low-rank Koopman projector** `S : ℝ^d → ℝ^{d_w}`, `d_w ≪ d`, satisfying simultaneously:
1. **Stiefel orthogonality** (`S^T S ≈ I`) — S is a genuine projector, not a rescaling.
2. **Scale-equivariance** under a learned multiplicative dilation group (RG fixed-point condition).
3. **Contractive propagator** — the induced propagator `K = S^T A S` has spectral radius `ρ(K) < 1`, guaranteeing dissipative dynamics.

A Gram-matrix formulation cannot satisfy (2) or (3) generically — `A^T A` lives in the wrong representation for dilation equivariance and has spectral radius `‖A‖² ≥ 1`. This is the *mechanism-level* novelty.

### Commitment 2 — Joint role of S as workspace AND consolidation, in the SAME slow-mode subspace

The same `S` projects the residual stream into the workspace AND defines the subspace into which a fast-write episodic ledger consolidates. This places **consolidation inside a Koopman slow-mode invariant subspace** — a placement no prior consolidation work makes (SleepGate compresses KV cache; 2603.03355 uses Gram cross-talk; NMDAR paper nonlinearizes). Biological grounding: hippocampal replay manifolds during sharp-wave ripples live on ~20-60 latent dimensions (Nieh 2021, Gava 2024, Schuck/Niv 2024-25), which is exactly the regime predicted by a slow-mode subspace.

### Commitment 3 — Continuous spectral-band selector (vs AMOR's binary gate)

A small per-token controller `M` outputs a **soft mask over the eigenbands of K**, not a binary architecture switch. AMOR-style entropy gates produce step-function engagement — saturate above threshold. A spectral selector produces graded computation: bandwidth grows continuously with item difficulty.

## Three falsifiable predictions (one per discipline)

Each is a clean experiment that, if it fails, kills the corresponding part of the claim.

### P1 (Math/physics — Avraham)
The trained S will exhibit eigenvalue spectra concentrated near unit modulus along a slow band, with measurable RG-style scaling collapse of layerwise representations under depth rescaling. Probes on Pythia / Llama-3 should reveal the same structure as a posthoc property if the inductive bias is real.

### P2 (Biology — Bar-Tal)
During the consolidation phase, SMW will exhibit emergent replay events whose participation ratio falls in **20-60 dimensions**, matching empirical hippocampal-cortical replay manifolds. Ablating S will abolish workspace bottleneck AND replay-manifold dimensionality **jointly**, not separately. If the two dissociate under ablation, the shared-operator unification fails.

### P3 (Cognition — Roth)
Eigenband-mask entropy will rise **monotonically without saturation** with item difficulty on graduated benchmarks (GPQA tiers, MATH levels 1-5), correlation coefficient `r > 0.6`. AMOR-style binary gates predict a saturating sigmoid; SMW predicts a smooth monotone with non-saturating tail. If we see saturation, the continuity claim fails.

## Position relative to the literature

| Paper | Closest claim | Delta vs SMW |
|---|---|---|
| Inhibitory Cross-Talk (2603.03355) | Attention as retrieval/consolidation/write-back operator | We use a Koopman projector with Stiefel + scale-equivariance + contractivity, not Gram matrices. We add the RG role. |
| AMOR (2602.13215) | Per-token entropy gate System 1↔2 | We replace binary architecture switch with continuous spectral-band selection over one operator. |
| SleepGate (2603.14517) | Sleep-inspired KV-cache consolidation | Our consolidation is into a Koopman slow-mode subspace, weight-shared with the workspace bottleneck. |
| Associative Transformer (2309.12862) | Global Workspace bottleneck attention | We tie the same operator to consolidation + RG + spectral selector. |

## What this experiment is and is not

- **It is** a specific, narrow, falsifiable claim about an architectural inductive bias.
- **It is not** a paradigm-shifting unification. The "single operator, multiple roles" framing is owned by 2603.03355.
- **It is** publishable as a workshop or short paper if any 2 of the 3 predictions hold.
- **It is not** a Transformer replacement. SMW is a retrofit onto existing decoder-only stacks.

## Pre-registered experiment

See [05-experiment-design.md](05-experiment-design.md), updated to test the three sharpened predictions. The 5-condition factorial remains; we add three new probe-style measurements (eigenvalue spectrum, replay manifold dimensionality, mask-entropy monotonicity) as primary diagnostics for the three predictions above.
