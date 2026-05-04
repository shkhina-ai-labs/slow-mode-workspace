---
title: "Debate — Three PhD speeches verbatim"
type: experiment-debate
created: 2026-05-04
---

# The Debate (verbatim)

## Speech 1 — Prof. Lior Avraham (Mathematician / Theoretical Physicist)

Colleagues, my position is this: the Transformer is an information-geometry catastrophe disguised as an engineering triumph. Permit me a physicist's diagnosis.

The residual stream is, formally, a discrete dynamical system on a high-dimensional manifold, yet we equip it with no metric, no symmetry group, and no conservation law. Each block is a vector field with arbitrary Jacobian; nothing constrains the flow. The softmax attention kernel is a Boltzmann distribution at fixed temperature one — a thermodynamic accident, not a choice — and it lives in a token coordinate system that is neither scale-equivariant nor translation-covariant in any meaningful sense. The consequence, visible in the layerwise covariance spectrum, is that representations undergo an unplanned phase transition near a critical normalized depth around 0.42 in large models — an emergent renormalization-group flow that the architecture neither anticipates nor exploits. We are, in effect, doing statistical mechanics with a Hamiltonian we refuse to write down. The O(n²) attention cost is then merely a symptom; the deeper pathology is that the model must rediscover, from data alone, the geometric invariances that physics hands us for free. No wonder sample efficiency collapses.

My proposal — and I will defend it against my colleagues' wetware enthusiasms in a moment — is a Renormalization-Group-Equivariant Transformer with Koopman-linearized residual dynamics. Concretely: replace the stack of identical attention blocks with a continuous-depth neural ODE whose vector field is constrained to commute with a learned multiplicative scaling group acting on token positions and embedding norms — a discrete dilation symmetry, the natural generalization of Anson et al.'s scale-invariant attention. Inside this flow, the residual stream is lifted into a Koopman observable space where the layerwise propagator is constrained to be linear with a spectrally-bounded operator, in the manner of DeepKoopFormer. Two consequences follow with mathematical force. First, depth becomes a continuous renormalization parameter rather than a discrete hyperparameter; the model literally performs coarse-graining as a built-in operation. Second, the Koopman linearization gives us a closed-form composition law — long-range dependencies become operator powers, computable in O(n log n) via spectral decomposition rather than O(n²) materialization.

My falsifiable prediction. On compositional generalization benchmarks where length extrapolation is the failure mode — SCAN, COGS, and the long-context sub-task of BIG-Bench Hard — a 1.3B-parameter RG-Koopman model will exceed a parameter-matched Llama-style baseline by at least 15 absolute points on out-of-distribution length splits, while matching it on in-distribution perplexity within 2%. The mechanism is precise: scale-equivariance forces the learned operator to act identically on rescaled inputs, so the inductive bias for "the same rule at a longer length" is hard-coded rather than empirically estimated.

To my colleagues' anticipated objections — that brains are not Hamiltonian and cognition is not group-theoretic — I reply: the cortex exhibits scale-free criticality, predictive-coding hierarchies are renormalization flows, and Friston's free-energy principle is precisely a variational Lagrangian. Biology already obeys the physics. We are merely proposing the architecture catch up.

**Sources:**
- [Latent Object Permanence (2026)](https://arxiv.org/abs/2601.19942)
- [Scale-invariant attention, Anson et al. (2025)](https://arxiv.org/html/2505.17083v1)
- [DeepKoopFormer (2025)](https://arxiv.org/html/2508.02616)
- [Neural ODE Transformers (2025)](https://arxiv.org/html/2503.01329v2)
- [Platonic Transformers (2025)](https://arxiv.org/html/2510.03511v1)
- [Geometric Field Theory Framework for Transformers (2025)](https://arxiv.org/html/2511.08243v1)

---

## Speech 2 — Prof. Maya Bar-Tal (Computational Neuroscientist)

Distinguished colleagues, thank you. I'll speak as a neurobiologist, and I'll be blunt: the Transformer is an exquisite cortex without a hippocampus, without neuromodulation, and without sleep. We have built a frozen neocortex and are surprised it cannot remember yesterday.

**The diagnosis.** The mammalian brain solves the stability-plasticity dilemma through Complementary Learning Systems. The hippocampus performs sparse, pattern-separated, one-shot encoding of episodes within roughly 100 milliseconds via NMDA-dependent LTP in CA3. The neocortex, by contrast, learns slowly across thousands of exposures, building overlapping distributed representations. During slow-wave sleep, sharp-wave ripples replay hippocampal traces to cortex, gradually consolidating them into semantic structure. Crucially, this entire choreography is gated by neuromodulators: dopamine signals reward-prediction error and opens plasticity windows; acetylcholine biases the hippocampus toward encoding versus retrieval; norepinephrine flags surprise and amplifies learning rate. The Transformer has none of this. It has one timescale, one weight matrix, no separation between episode and gist, and zero plasticity at inference. When it "remembers," it does so by re-reading its entire context window, an O(n²) act of desperate re-perception. That is not memory. That is rehearsal under fluorescent light.

**The proposal.** I argue for a CLS-hybrid architecture with three structural commitments. First, a fast-write episodic module — a Kanerva-style sparse distributed memory, addressable by content, with one-shot writes at inference. Larimar (Das et al., 2024) demonstrated this is tractable: 8-10× faster knowledge edits than retraining, with no parameter inflation. Second, a slow consolidation pathway — offline replay from the episodic store back into a small set of low-rank cortical adapters during "sleep" cycles, mimicking ripple-driven consolidation. Third, and most underappreciated: a neuromodulatory gating signal — a learned scalar, computed from prediction error and novelty, that controls whether a given token triggers an episodic write, a cortical update, or neither. This is the dopamine-acetylcholine analog. Plastic Transformers (Schmidgall et al., 2025) show such three-factor Hebbian gates are trainable end-to-end without exploding compute.

**Falsifiable predictions.** A 7B model augmented this way, with no parameter increase, should: (i) reduce catastrophic forgetting on sequential task streams from current ~40% retention to above 85%; (ii) achieve one-shot factual edit accuracy above 90% with locality preserved, versus ~60% for ROME-class editors; (iii) improve few-shot transfer to genuinely novel domains by 15-25 points on benchmarks like MMLU-Pro held-out splits; (iv) cut the context length needed for equivalent task performance by roughly 4×, because episodic recall replaces brute re-attention.

**Pre-emption.** To my mathematician colleague: yes, attention is elegant linear algebra, but elegance without a memory hierarchy is a beautiful amnesiac — the bottleneck is architectural, not numerical. To my psychologist colleague: behavioral analogies to human memory are necessary but insufficient; without a mechanistic substrate — sparse codes, three-factor plasticity, replay — you describe the symptom, not the circuit.

The next leap is not larger. It is hippocampal.

**Sources:**
- [Larimar — Episodic Memory Control (Das et al., 2024)](https://arxiv.org/abs/2403.11901)
- [Episodic Memory is the Missing Piece for Long-Term LLM Agents (Pink et al., 2025)](https://arxiv.org/pdf/2502.06975)
- [Gradient of CLS through meta-learning (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.07.10.664201v1.full)
- [Latent learning: episodic memory complements parametric learning (2025)](https://arxiv.org/html/2509.16189v1)
- [Adaptation in Transformers with Hebbian and Gradient-Based Plasticity (2025)](https://www.arxiv.org/pdf/2510.21908)
- [Towards LLMs with human-like episodic memory (Princeton, 2025)](https://compmem.princeton.edu/wp/wp-content/uploads/2025/07/dongetal25.pdf)

---

## Speech 3 — Prof. Daniel Roth (Cognitive Psychologist)

Colleagues, distinguished panel — thank you.

My friends here will speak of elegance, and of biology. I will speak of something more humbling: **a missing cognitive function**. The transformer, magnificent as it is, computes every token with the same dull uniformity. It does not know what it does not know. It does not pause. It does not say, "this is hard — slow down." It has no metacognition, no goal stack, no working-memory bottleneck forcing abstraction. In dual-process terms, it is pure System 1 — a 70-billion-parameter intuition pump with no System 2 sitting behind it.

The diagnosis is **two coupled deficits**: (1) absence of a *Global Workspace* — Baars's and Dehaene's winner-take-all serial bottleneck that broadcasts a single coherent content across distributed modules — and (2) absence of *metacognitive monitoring* — the judgments-of-confidence that gate cognitive control in humans.

**My proposal** is a two-part retrofit that adds zero parameters and reduces FLOPs in expectation:

**First, a Global-Workspace Bottleneck Layer.** Every k transformer blocks, all module activations must compete to write into a low-rank workspace of dimension d_w ≪ d_model — say 64 dimensions against a 12,288-dimensional residual stream. A top-k attention selects the winners; that low-bandwidth slate is then *broadcast* back to every subsequent layer. The bottleneck IS the feature. Capacity limits — Miller's 7±2, Cowan's 4 — are not bugs of biology. They are the evolutionary pressure that *forces* abstraction, compositionality, and serial binding.

**Second, a Metacognitive Controller Head.** A small auxiliary head — trained via existing logits — emits a per-token "judgment of learning." When confidence is high, the model exits early (System 1). When confidence is low, the controller adaptively allocates: deeper recurrence on the workspace, additional chain-of-thought tokens, expert-router calls. This realizes Kahneman's two systems as a *resource allocator*, not a parallel module.

**Falsifiable prediction.** On a 7B base, this retrofit will deliver: ARC-AGI +12 to +18 points; GSM8K-hard +8 points; HumanEval-hard +5; and — most diagnostically — expected calibration error cut by 40%, hallucination rate on TruthfulQA cut by 25%.

**Pre-emption.** To my mathematician: elegance is not a function. The brain is a kludge of thalamocortical loops and it thinks; smooth manifolds do not. To my neurobiologist: I agree wetware is irrelevant — which is precisely my point. What matters is the COMPUTATION. Global broadcast, metacognitive gating, capacity-limited working memory — these are computational signatures, implementable in silicon as readily as in cortex.

Give the transformer a bottleneck and a conscience. The rest follows.

**Sources:**
- [Associative Transformer (Sun et al., 2024-2025)](https://arxiv.org/html/2309.12862)
- [Metacognition and Uncertainty Communication in LLMs (Steyvers & Peters, 2025)](https://arxiv.org/html/2504.14045v1)
- [LLMs Are Capable of Metacognitive Monitoring (Binder et al., 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12136483/)
- [The Relational Bottleneck as Inductive Bias (Webb et al., 2024)](https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(24)00080-9)
- [LLMs Coupled with Metacognition (Didolkar et al., 2025)](https://arxiv.org/html/2508.17959v1)
- [Theater of Mind for LLMs (2026)](https://arxiv.org/abs/2604.08206)
