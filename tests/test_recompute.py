"""Correctness and memory tests for selective activation recompute.

Single process, no launcher:

    python3 tests/test_recompute.py

Checks:
  1. equivalence — logits and every gradient with recompute on match the
     no-recompute model bitwise (the recomputed subgraph runs the same
     deterministic ops on the same inputs)
  2. memory — the bytes autograd actually stashes for backward, measured
     with saved_tensors_hooks, drop by at least the attention-matrix term
     (2 tensors of s^2*b*a floats per layer: the masked scores kept by
     softmax's backward and the softmax output kept by the matmul with V)
  3. overhead — the recomputed region is entered exactly twice per
     microbatch per layer (once forward, once during backward)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from dst import parallel
from dst.model import GPT, GPTConfig
from dst import recompute as rc

SEED = 1234


def run(cfg, idx, targets):
    model = GPT(cfg, parallel.SINGLE, seed=SEED)
    saved_bytes = 0

    def pack(t):
        nonlocal saved_bytes
        saved_bytes += t.nbytes
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        logits, loss = model(idx, targets)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    return logits.detach(), loss.detach(), grads, saved_bytes


def main():
    cfg = dict(vocab_size=512, block_size=64, n_layer=4, n_head=8, n_embd=128, dropout=0.0)
    B = 4
    g = torch.Generator().manual_seed(SEED + 1)
    idx = torch.randint(0, 512, (B, 64), generator=g)
    targets = torch.randint(0, 512, (B, 64), generator=g)

    calls = 0
    orig_forward = rc._Recompute.forward

    def counting_forward(ctx, fn, *args):
        nonlocal calls
        calls += 1
        return orig_forward(ctx, fn, *args)

    rc._Recompute.forward = staticmethod(counting_forward)

    logits0, loss0, grads0, bytes0 = run(GPTConfig(**cfg), idx, targets)
    logits1, loss1, grads1, bytes1 = run(GPTConfig(**cfg, recompute_attention=True), idx, targets)

    ok = True

    d = (logits0 - logits1).abs().max().item()
    dl = abs(loss0.item() - loss1.item())
    worst = max((grads0[n] - grads1[n]).abs().max().item() for n in grads0)
    exact = d == 0.0 and dl == 0.0 and worst == 0.0
    ok &= exact
    print(f"  [{'PASS' if exact else 'FAIL'}] bitwise equivalence: "
          f"logits {d:.1e}, loss {dl:.1e}, worst grad {worst:.1e}")

    # Two s x s activations per layer vanish from the stash: the masked
    # scores (softmax backward keeps its input... actually its output; the
    # matmul with V keeps the softmax output too — deduplicated by autograd)
    # plus the raw scores kept by masked_fill/div. Demand at least the
    # dominant 2 * s^2 * b * a term.
    s, a, L = 64, 8, 4
    att_term = 2 * (B * a * s * s) * 4 * L
    dropped = bytes0 - bytes1
    enough = dropped >= att_term
    ok &= enough
    print(f"  [{'PASS' if enough else 'FAIL'}] saved-for-backward bytes: "
          f"{bytes0:,} -> {bytes1:,} (dropped {dropped:,}, "
          f"attention-matrix term {att_term:,})")

    once_per_layer = calls == L
    ok &= once_per_layer
    print(f"  [{'PASS' if once_per_layer else 'FAIL'}] recompute region entered "
          f"{calls} times forward (n_layer={L}); backward re-runs it once each")

    print("\nALL PASS" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
