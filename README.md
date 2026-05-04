# Slow-Mode Workspace (SMW)

> A reference implementation of a narrowed, falsifiable architectural hypothesis: **a single Koopman-spectral projector S that simultaneously serves as a Global-Workspace bottleneck and as the slow-mode subspace into which an episodic ledger consolidates, modulated per token by a continuous metacognitive controller over S's eigenbands.**

**Status:** research scaffold. Smoke tests pass. The full pre-registered experiment has *not* been run yet. We're publishing the code so the community can critique the design before pilot compute is spent.

**Mirror:** Hugging Face — https://huggingface.co/Shkhina-AI-Labs/slow-mode-workspace

---

## Honest positioning

The general framing of "single attention/memory operator playing multiple roles in a Transformer" is **not** novel — it was claimed by [Liu et al., *Inhibitory Cross-Talk in Attention-Coupled Latent Memory* (arXiv:2603.03355, Feb 2026)](https://arxiv.org/abs/2603.03355). Per-token metacognitive gating is **not** novel — it was claimed by [AMOR (arXiv:2602.13215, Jan 2026)](https://arxiv.org/abs/2602.13215). Sleep-inspired KV-cache consolidation is **not** novel — see [SleepGate (arXiv:2603.14517)](https://arxiv.org/abs/2603.14517).

What we propose is a **specific instantiation** that those papers do not make:

1. Replace the Gram-matrix `A^T A` of latent-memory operators with a **Koopman projector** `S` satisfying three joint constraints:
   - **Stiefel orthogonality** — `S^T S ≈ I` (S is a genuine projector, not a rescaling)
   - **Scale-equivariance** under multiplicative dilation — `S(λx) ≈ λ^Δ S(x)` (RG fixed-point condition)
   - **Contractive propagator** — `ρ(K = S^T A S) < 1` (dissipative dynamics)
2. Use the **same** `S` as workspace bottleneck **and** as the subspace into which the episodic ledger consolidates — placing consolidation inside a Koopman slow-mode invariant subspace (a placement no prior consolidation work makes).
3. Replace AMOR's binary entropy gate with a **continuous per-token mask over S's eigenbands** — bandwidth allocation on the slow-mode manifold, not architecture switching.

## Three falsifiable predictions (one per discipline)

| Prediction | Discipline | If false → |
|---|---|---|
| **P1.** Trained S exhibits eigenvalue concentration near \|λ\|=1 along a slow band, with RG-style scaling collapse under depth rescaling. | math/physics | Koopman-projector inductive bias is doing nothing. Hypothesis dies. |
| **P2.** Emergent replay during the consolidation phase has participation ratio in **20–60** dimensions (matching empirical hippocampal-cortical replay manifolds, e.g. Nieh 2021, Gava 2024). Ablating S abolishes workspace AND replay manifold **jointly**. | biology | Shared-operator unification is wrong; we go back to two operators. |
| **P3.** Eigenband-mask entropy correlates **monotonically without saturation** with item difficulty on graduated benchmarks (GPQA tiers, MATH 1–5), `r > 0.6`. | cognition | Continuity claim fails; SMW reduces to AMOR-with-extra-dims. |

## Architecture

![SMW Architecture v2](https://raw.githubusercontent.com/shkhina-ai-labs/slow-mode-workspace/main/docs/smw-architecture-v2.jpeg)

(Figure: 4-panel paper diagram showing residual stack → S projector with three constraints → eigenvalue spectrum with slow band → dual role as workspace + consolidation channel → continuous metacognitive selector contrasted with AMOR's binary gate.)

## Code structure

```
smw/
  projector.py         SlowModeProjector — the shared S, with the 3 constraints as soft regularizers
  metacognitive.py     MetacognitiveController — continuous per-token spectral mask
  episodic.py          EpisodicLedger — Larimar-style sparse-distributed memory (in workspace space)
  model.py             SMWModel + SMWSite — decoder-only Transformer with the SMW retrofit
  configs.py           SMWConfig + the 5+1 ablation conditions
tests/
  test_smoke.py        Forward-pass + diagnostic smoke tests for all conditions
scripts/
  train_pilot.py       Toy MQAR (Multi-Query Associative Recall) training script
```

## The 5+1 ablation conditions

The pre-registered experiment uses a 5+1 factorial. The **critical comparison** is `C5_smw` vs. `C4_all_independent`. If they tie, the shared-operator identity is unnecessary and `H1` is rejected.

| Condition | GW bottleneck | Episodic mem | RG constraint | Shared S? |
|---|---|---|---|---|
| C0 baseline | – | – | – | – |
| C1 GW-only | ✓ | – | – | – |
| C2 E-only | – | ✓ | – | – |
| C3 RG-only | – | – | ✓ | – |
| C4 All-Independent | ✓ | ✓ | ✓ | three separate |
| C5 SMW (ours) | ✓ | ✓ | ✓ | one shared |

All conditions are designed to be parameter-matched and FLOP-matched.

## Quick start

```bash
git clone https://github.com/shkhina-ai-labs/slow-mode-workspace
cd slow-mode-workspace
pip install -e .
python -m pytest tests/                                            # smoke tests
PYTHONPATH=. python scripts/train_pilot.py --condition C5_smw --steps 1000
PYTHONPATH=. python scripts/train_pilot.py --condition C4_all_independent --steps 1000
```

The train script logs `slow_band` (P1 diagnostic), `mask_H` (P3 diagnostic) at every checkpoint.

## What this is, what it isn't

- **It is** a runnable reference implementation of a falsifiable hypothesis.
- **It is not** a trained model. There are no weights yet.
- **It is** a starting point for fine-tuning experiments by the community.
- **It is not** a Transformer replacement. SMW is a retrofit onto existing decoder-only stacks.

## Looking for

- Critique of the falsification design (predictions P1, P2, P3) **before** we commit pilot compute
- Anyone who has tried weight-tied Koopman bottlenecks under similar constraints — confirm or reject
- Lab partners for the 350M pilot if predictions hold

## How the hypothesis was generated

Via a structured agentic 3-PhD panel debate (mathematical-physics + computational-neuroscience + cognitive-science perspectives), with a novelty gate that surfaced and credited prior art. Two debate rounds, narrowing the claim each round. Full debate transcript and pre-registration:
👉 see the [Shkhina AI Labs research wiki](https://github.com/shkhina-ai-labs/slow-mode-workspace/blob/main/docs/research-debate.md) (link added when wiki is mirrored).

## Citation

If this scaffold is useful, please cite:

```bibtex
@misc{shkhina2026smw,
  title  = {{Slow-Mode Workspace}: A Koopman-Spectral Instantiation of Latent-Memory Transformer Architectures},
  author = {{Shkhina AI Labs}},
  year   = {2026},
  note   = {Research scaffold; pre-registered experiment in progress},
  url    = {https://github.com/shkhina-ai-labs/slow-mode-workspace}
}
```

And cite the priors we extend:

- Liu et al., *Inhibitory Cross-Talk in Attention-Coupled Latent Memory*, arXiv:2603.03355
- AMOR, arXiv:2602.13215
- DeepKoopFormer, arXiv:2508.02616
- Larimar, arXiv:2403.11901
- Associative Transformer, arXiv:2309.12862

## License

MIT — see [LICENSE](LICENSE).
