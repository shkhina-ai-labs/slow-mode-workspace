---
title: "Experiment Design — Falsifying the Slow-Mode Conservation Hypothesis"
type: experiment-design
created: 2026-05-04
status: pre-registered
---

# Experiment Design — Pre-Registration

This document follows the scientific method: hypothesis → operational definitions → IV/DV → predictions with magnitudes → controls → statistical plan → falsification criteria. It is **pre-registered** — the criteria are committed before data is collected.

---

## 1. Question

Does conserving a single low-rank Koopman-invariant slow-mode subspace across the residual flow — and reusing it as Global Workspace bottleneck, episodic-cortical consolidation channel, and RG projector — produce **synergistic** improvements on continual learning, length extrapolation, and calibration that exceed the sum of its independent module ablations?

## 2. Hypotheses

- **H1 (Slow-Mode Conservation, novel):** A single shared operator `S` produces synergistic gains. The interaction effect (SMW − sum of ablations) is significantly positive on at least 2 of 3 primary outcomes.
- **H0 (Modular Composition, null):** No interaction. Gains are additive across the three modules.
- **H_alt (Single-module dominant):** One ablation accounts for ≥80% of the SMW gain; the shared-operator identity is unnecessary.

H0 and H_alt are both grounds for rejecting the novel claim.

## 3. Operational definitions

| Term | Definition |
|---|---|
| Slow-mode subspace | The top-r Koopman eigenfunctions of the linearized residual propagator, with `r = 64` for `d = 4096`. |
| Shared operator `S` | A single `R^d → R^r` projector matrix used in three places: workspace broadcast, consolidation, RG projection. Weight-tied. |
| Catastrophic forgetting | Average accuracy drop on `task_i` after training on `task_{i+1..i+5}` in a 5-task continual stream. |
| Length extrapolation | Accuracy on test sequences 4× longer than the longest training sequence. |
| Calibration (ECE) | Expected Calibration Error on multiple-choice questions, 15-bin reliability diagram. |

## 4. Independent variable (architecture conditions)

A **2×2×2 factorial** plus the full-SMW cell, total 5 conditions:

| Cond | GW bottleneck | Episodic mem | RG/scale-equiv | Shared `S`? |
|---|---|---|---|---|
| C0 (Baseline) | – | – | – | – |
| C1 (GW only) | ✓ | – | – | – |
| C2 (E only) | – | ✓ | – | – |
| C3 (RG only) | – | – | ✓ | – |
| C4 (All-Independent) | ✓ | ✓ | ✓ | three separate operators |
| C5 (SMW, novel) | ✓ | ✓ | ✓ | one shared `S` |

The critical comparison is **C5 vs C4**. If H1 is true, C5 > C4 by a non-trivial margin on at least 2 of 3 primary outcomes. If H0 is true, C5 ≈ C4.

All conditions are **parameter-matched** (within ±0.5%) and **FLOP-matched** (within ±2%). The shared-operator condition (C5) deliberately uses fewer parameters than C4 — the saved parameters are spent uniformly on the residual stream so total parameters match.

## 5. Dependent variables

**Primary outcomes (powered for):**
1. SCAN OOD length-split accuracy
2. MQAR (Multi-Query Associative Recall) at sequence length 4× training max
3. Continual-learning retention (5-task stream from CLiMB or similar)

**Secondary outcomes:**
4. Calibration (ECE on TruthfulQA + ARC-Easy)
5. One-shot factual edit (CounterFact + locality)
6. Inference FLOPs (mean, P95)
7. ARC-AGI subset
8. GSM8K-hard

## 6. Predicted magnitudes (committed)

| DV | C0 baseline | C5 (SMW) | Synergy threshold (C5 − C4) |
|---|---|---|---|
| SCAN OOD | ~30% | ≥50% | ≥+5 pts |
| MQAR 4× | ~55% | ≥75% | ≥+5 pts |
| CL retention | ~40% | ≥80% | ≥+10 pts |
| ECE | ~12% | ≤7% | (any improvement) |
| FLOPs | 1.0× | 0.7-0.85× | – |

A "win" is declared only if **C5 beats baseline AND beats the All-Independent C4 by the synergy threshold on at least 2 of 3 primary outcomes**. Beating only the baseline is not sufficient — that would be consistent with H0.

## 7. Controls

- **Parameter-matched, FLOP-matched** baseline and ablations (within tolerances above).
- **Same training tokens, same optimizer, same schedule** — only architecture varies.
- **Same data splits** — fixed seed for split selection across conditions.
- **Three independent training seeds per condition** (5 conditions × 3 seeds = 15 runs).
- **Held-out novelty domains** — at least 1 of 3 evaluation tasks must be drawn from data published *after* the pre-training cutoff to control for memorization.

## 8. Statistical plan

- Per outcome, **paired bootstrap (10,000 resamples)** comparing C5 vs each ablation, computing 95% CIs on the difference.
- For the synergy claim: compute `Δ_synergy = (C5 − C0) − [(C1 − C0) + (C2 − C0) + (C3 − C0)]`. Bootstrap CI for Δ_synergy.
- **Pre-registered alpha = 0.01** per primary outcome (Bonferroni-adjusted across 3 primary).
- **Power calculation:** n = 3 seeds × 5 conditions, so we need a large effect (Cohen's d ≥ 1.0 on per-seed differences) to claim H1. This is intentional — small effects do not justify a new architectural category.

## 9. Falsification criteria (commit before data)

The hypothesis **is rejected** if any of the following hold:

- ❌ C5 fails to beat C0 by ≥10 absolute points on any 2 of the 3 primary outcomes after full budget.
- ❌ Δ_synergy ≤ 0 (95% CI lower bound) on at least 2 of 3 primary outcomes — meaning the shared-operator identity adds nothing beyond modular composition.
- ❌ One single ablation (C1, C2, or C3) reaches ≥80% of the C5 gain on all primary outcomes — meaning one module alone is doing all the work.
- ❌ FLOPs increase ≥10% over baseline — defeats the "no compute increase" constraint.

If rejected, write a short post-mortem in `06-postmortem.md` explaining which alternative (H0 or H_alt) the data supports.

## 10. Resources & timeline

| Phase | Resource | Wall-clock |
|---|---|---|
| Implementation | 1 ML eng × 2 weeks | 2 wk |
| Pre-training (15 runs, 1.3B param, 50B tokens each) | 8×H100 cluster | 4 wk |
| Eval suite | 1 ML eng × 1 week | 1 wk |
| Analysis & write-up | 1 ML eng × 1 week | 1 wk |

**Total: 8 weeks, ~$60-80K compute** (at vast.ai or contracted A100/H100 hours). Cheaper if we drop to 350M-param scale (recommended pilot).

## 11. Pilot first — 350M scale

Before committing to the full 1.3B run, do a **350M-parameter pilot** (5 conditions × 1 seed = 5 runs, ~1 week, ~$5K). Decision rule:
- If pilot shows Δ_synergy > 0 with point estimate above 50% of committed magnitudes → proceed to full.
- If pilot is null → write postmortem and abandon, OR pivot to one of the three ablations.

Pilot saves us from spending $60K on a dead hypothesis.

## 12. Threats to validity

| Threat | Mitigation |
|---|---|
| Implementation bug favoring C5 | Independent code review; release ablation code as part of the paper |
| Hyperparameter overfitting to C5 | Hyperparameters tuned on C0; same hyperparameters used for all conditions |
| Cherry-picked benchmarks | Pre-register the 3 primary + 5 secondary; report all of them |
| Memorization of OOD test | Use post-cutoff domains for at least 1 primary task |
| Compute leak (C5 secretly using more FLOPs) | FLOP audit logged per run; reject if outside ±2% |

## 13. Outputs

- `results.json` — per-run metrics
- `analysis.ipynb` — bootstrap CIs + synergy plots
- `06-postmortem.md` if rejected, OR `06-results.md` if confirmed
- arXiv preprint (if confirmed)

---

## Status

- [x] Hypothesis registered
- [x] Falsification criteria committed
- [ ] Pilot run scheduled
- [ ] Full run scheduled
- [ ] Results
