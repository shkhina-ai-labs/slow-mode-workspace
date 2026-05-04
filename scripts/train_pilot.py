"""
train_pilot.py — toy MQAR (Multi-Query Associative Recall) training script.

Generates synthetic data of the form:
    [k1] [v1] [k2] [v2] ... [kN] [vN] | [query=k_i] -> v_i

The model must recall the value associated with a queried key seen earlier
in the same sequence.  Length-OOD generalization is tested by training on
short sequences and evaluating on longer ones.

This is a SMOKE-LEVEL pilot, intended to verify the implementation runs and
to establish a baseline-vs-SMW signal for the eventual full pilot.  It is
NOT the 350M / 1.3B pre-registered run.

Usage:
    python scripts/train_pilot.py --condition C5_smw --steps 1000
    python scripts/train_pilot.py --condition C0_baseline --steps 1000
    python scripts/train_pilot.py --condition C4_all_independent --steps 1000
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F

from smw import SMWModel, CONDITIONS


# ---- synthetic MQAR data ----

VOCAB_SIZE = 64
KEY_RANGE = (0, 16)
VAL_RANGE = (16, 64)
SEP_TOKEN = 0  # not used as a key/val
QUERY_TOKEN = 1  # special "go retrieve" marker


def make_mqar_batch(batch_size: int, n_pairs: int, device="cpu"):
    """
    Returns (idx, targets) where targets are -100 (ignore) except at the
    query-answer position.
    """
    seq_len = 2 * n_pairs + 2  # pairs + [query_marker, query_key]
    idx = torch.zeros(batch_size, seq_len + 1, dtype=torch.long, device=device)
    targets = torch.full((batch_size, seq_len + 1), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        keys = torch.randperm(KEY_RANGE[1] - KEY_RANGE[0])[:n_pairs] + KEY_RANGE[0]
        vals = torch.randint(VAL_RANGE[0], VAL_RANGE[1], (n_pairs,))
        # Lay out: k1 v1 k2 v2 ...
        for i in range(n_pairs):
            idx[b, 2 * i] = keys[i]
            idx[b, 2 * i + 1] = vals[i]
        # Query marker + a sampled key
        q_i = torch.randint(0, n_pairs, (1,)).item()
        idx[b, 2 * n_pairs] = QUERY_TOKEN
        idx[b, 2 * n_pairs + 1] = keys[q_i]
        # Answer slot
        idx[b, 2 * n_pairs + 2] = vals[q_i]
        targets[b, 2 * n_pairs + 2] = vals[q_i]
    return idx, targets


def evaluate(model, n_pairs, n_batches=8, batch_size=16, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            idx, targets = make_mqar_batch(batch_size, n_pairs, device=device)
            logits, _, _ = model(idx)
            # Answer position is the LAST token in idx (we predict from the prior context)
            ans_pos = idx.size(1) - 1
            preds = logits[:, ans_pos - 1].argmax(-1)
            true = targets[:, ans_pos]
            mask = true != -100
            correct += (preds[mask] == true[mask]).sum().item()
            total += mask.sum().item()
    model.train()
    return correct / max(total, 1)


# ---- training ----

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", type=str, default="C5_smw", choices=list(CONDITIONS.keys()))
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_pairs_train", type=int, default=4)
    p.add_argument("--n_pairs_eval_id", type=int, default=4)   # in-distribution
    p.add_argument("--n_pairs_eval_ood", type=int, default=8)  # length-OOD (P1-style probe)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg = CONDITIONS[args.condition]
    cfg.vocab_size = VOCAB_SIZE
    cfg.block_size = max(2 * args.n_pairs_eval_ood + 4, cfg.block_size)
    cfg.d_model = 128
    cfg.n_layers = 4
    cfg.d_w = 16
    cfg.k_bottleneck = 2

    model = SMWModel(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{cfg.name}] params: {n_params:,}  device: {args.device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx, targets = make_mqar_batch(args.batch_size, args.n_pairs_train, device=args.device)
        logits, ce_loss, reg_loss = model(idx, targets)
        loss = ce_loss + (reg_loss if reg_loss is not None else 0.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % max(args.steps // 10, 1) == 0 or step == args.steps:
            id_acc = evaluate(model, args.n_pairs_eval_id, device=args.device)
            ood_acc = evaluate(model, args.n_pairs_eval_ood, device=args.device)
            slow_band = model.slow_band_mass()
            mask_H = model.last_mask_entropy()
            print(
                f"step {step:4d}  ce={float(ce_loss):.3f}  reg={float(reg_loss) if reg_loss is not None else 0.0:.3f}"
                f"  id_acc={id_acc:.2f}  ood_acc={ood_acc:.2f}"
                f"  slow_band={f'{slow_band:.2f}' if slow_band is not None else '-'}"
                f"  mask_H={f'{mask_H:.2f}' if mask_H is not None else '-'}"
            )

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
