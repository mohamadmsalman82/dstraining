"""Numerical correctness suite for tensor parallelism.

Run under torchrun with N processes; TP degree = world size:

    torchrun --standalone --nproc_per_node=2 tests/test_tp_correctness.py

Because initialization is tp-invariant (layers draw full weights from a
shared seed and keep slices), every rank builds BOTH the TP model and a
plain single-process reference (tp=1) holding the exact full weights the
TP model shards. The suite then asserts, in fp32 on CPU:

  1. conjugacy   — f / f-bar reproduce a full matmul and its gradients
                   on a toy column+row split
  2. init        — every TP shard equals the corresponding slice of the
                   reference weight
  3. forward     — TP logits match reference logits, and are identical
                   across ranks
  4. backward    — every TP gradient shard matches the corresponding
                   slice of the reference gradient (this is also the
                   proof that replicated params — LayerNorms, embeddings
                   — need no extra sync)
  5. training    — 10 Adam steps on a fixed batch track the reference
                   loss trajectory, and the loss falls
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F

from dst import parallel
from dst.ops import (
    copy_to_tp_region,
    reduce_from_tp_region,
    gather_along_seq,
    reduce_scatter_along_seq,
    scatter_along_seq,
)
from dst.layers import ColumnParallelLinear, RowParallelLinear
from dst.model import GPT, GPTConfig

ATOL = 1e-5
SEED = 1234


def log(tp, msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def check(tp, name, a, b, atol=ATOL):
    diff = (a - b).abs().max().item()
    status = "PASS" if diff <= atol else "FAIL"
    log(tp, f"  [{status}] {name}: max abs diff {diff:.3e}")
    return diff <= atol


def test_conjugacy(tp):
    """A two-layer column-then-row split of Y = GeLU(XA)B against the
    unsplit computation, checking outputs and all gradients."""
    log(tp, "conjugacy: f / f-bar on a column+row split MLP")
    g = torch.Generator().manual_seed(SEED)
    n, h = 8, 16
    X = torch.randn(4, n, generator=g, requires_grad=True)
    A = torch.randn(n, h, generator=g, requires_grad=True)
    B = torch.randn(h, n, generator=g, requires_grad=True)

    # Reference: unsplit.
    Z_ref = F.gelu(X @ A) @ B
    Z_ref.sum().backward()

    # Split: A by columns, B by rows.
    h_local = h // tp.world
    Xp = X.detach().clone().requires_grad_(True)
    Ai = A.detach()[:, tp.rank * h_local : (tp.rank + 1) * h_local].clone().requires_grad_(True)
    Bi = B.detach()[tp.rank * h_local : (tp.rank + 1) * h_local, :].clone().requires_grad_(True)

    Y_i = F.gelu(copy_to_tp_region(Xp, tp) @ Ai)
    Z = reduce_from_tp_region(Y_i @ Bi, tp)
    Z.sum().backward()

    ok = check(tp, "forward Z", Z, Z_ref)
    ok &= check(tp, "grad X", Xp.grad, X.grad)
    ok &= check(tp, "grad A shard", Ai.grad, A.grad[:, tp.rank * h_local : (tp.rank + 1) * h_local])
    ok &= check(tp, "grad B shard", Bi.grad, B.grad[tp.rank * h_local : (tp.rank + 1) * h_local, :])
    return ok


def test_sp_conjugacy(tp):
    """g / g-bar with a column+row split against the unsplit computation,
    input and output sequence-sharded, checking outputs and gradients."""
    log(tp, "conjugacy: g / g-bar on a sequence-sharded column+row split")
    g = torch.Generator().manual_seed(SEED)
    b, s, n, h = 2, 8, 6, 12
    X = torch.randn(b, s, n, generator=g, requires_grad=True)
    A = torch.randn(n, h, generator=g, requires_grad=True)
    B = torch.randn(h, n, generator=g, requires_grad=True)

    Z_ref = F.gelu(X @ A) @ B
    Z_ref.sum().backward()

    h_local = h // tp.world
    s_local = s // tp.world
    Xp = X.detach().clone().requires_grad_(True)
    Ai = A.detach()[:, tp.rank * h_local : (tp.rank + 1) * h_local].clone().requires_grad_(True)
    Bi = B.detach()[tp.rank * h_local : (tp.rank + 1) * h_local, :].clone().requires_grad_(True)

    x_shard = scatter_along_seq(Xp, tp)          # enter the sharded region
    Y_i = F.gelu(gather_along_seq(x_shard, tp) @ Ai)   # g: full seq inside
    Z_shard = reduce_scatter_along_seq(Y_i @ Bi, tp)   # g-bar: sharded out
    Z_shard.sum().backward()

    z_ref_shard = Z_ref[:, tp.rank * s_local : (tp.rank + 1) * s_local, :]
    ok = check(tp, "forward Z shard", Z_shard, z_ref_shard)
    ok &= check(tp, "grad X", Xp.grad, X.grad)
    ok &= check(tp, "grad A shard", Ai.grad, A.grad[:, tp.rank * h_local : (tp.rank + 1) * h_local])
    ok &= check(tp, "grad B shard", Bi.grad, B.grad[tp.rank * h_local : (tp.rank + 1) * h_local, :])
    return ok


def ref_slice(tp_model, name, p_tp, ref_tensor, tp):
    """Slice a reference tensor (weight or gradient) down to the piece the
    TP parameter `name` holds. Replicated params map to the whole tensor."""
    if p_tp.shape == ref_tensor.shape:
        return ref_tensor
    module = tp_model.get_submodule(name.rsplit(".", 1)[0])
    leaf = name.rsplit(".", 1)[1]
    if isinstance(module, ColumnParallelLinear):
        n_local = p_tp.shape[0]
        return ref_tensor[tp.rank * n_local : (tp.rank + 1) * n_local]
    if isinstance(module, RowParallelLinear) and leaf == "weight":
        n_local = p_tp.shape[1]
        return ref_tensor[:, tp.rank * n_local : (tp.rank + 1) * n_local]
    raise AssertionError(f"unexpected sharded param {name}")


def build_models(cfg, tp):
    model_tp = GPT(cfg, tp, generator=torch.Generator().manual_seed(SEED))
    model_ref = GPT(cfg, parallel.SINGLE, generator=torch.Generator().manual_seed(SEED))
    return model_tp, model_ref


def make_batch(cfg, g):
    idx = torch.randint(0, cfg.vocab_size, (4, cfg.block_size), generator=g)
    targets = torch.randint(0, cfg.vocab_size, (4, cfg.block_size), generator=g)
    return idx, targets


def test_model(tp, model_tp, model_ref, cfg):
    ref_params = dict(model_ref.named_parameters())

    log(tp, "init: every shard is a slice of the reference weights")
    ok = True
    worst = 0.0
    for name, p in model_tp.named_parameters():
        sl = ref_slice(model_tp, name, p, ref_params[name].data, tp)
        worst = max(worst, (p.data - sl).abs().max().item())
    ok &= check(tp, f"all {len(ref_params)} params", torch.tensor(worst), torch.tensor(0.0), atol=0.0)

    g = torch.Generator().manual_seed(SEED + 1)
    idx, targets = make_batch(cfg, g)

    log(tp, "forward: TP logits vs reference, and across ranks")
    logits_tp, loss_tp = model_tp(idx, targets)
    logits_ref, loss_ref = model_ref(idx, targets)
    ok &= check(tp, "logits", logits_tp, logits_ref)
    ok &= check(tp, "loss", loss_tp, loss_ref)

    gathered = [torch.empty_like(logits_tp) for _ in range(tp.world)]
    dist.all_gather(gathered, logits_tp.contiguous(), group=tp.group)
    rank_diff = max((t - gathered[0]).abs().max().item() for t in gathered)
    ok &= check(tp, "cross-rank logit identity", torch.tensor(rank_diff), torch.tensor(0.0), atol=0.0)

    log(tp, "backward: every gradient shard vs the reference slice")
    loss_tp.backward()
    model_tp.finalize_grads()
    loss_ref.backward()
    worst_name, worst = None, -1.0
    for name, p in model_tp.named_parameters():
        grad_sl = ref_slice(model_tp, name, p, ref_params[name].grad, tp)
        d = (p.grad - grad_sl).abs().max().item()
        if d > worst:
            worst_name, worst = name, d
    ok &= check(tp, f"all grads (worst: {worst_name})", torch.tensor(worst), torch.tensor(0.0))
    model_tp.zero_grad(set_to_none=True)
    model_ref.zero_grad(set_to_none=True)
    return ok


def test_training(tp, model_tp, model_ref, cfg, steps=10):
    log(tp, f"training: {steps} Adam steps track the reference trajectory")
    g = torch.Generator().manual_seed(SEED + 2)
    idx, targets = make_batch(cfg, g)
    opt_tp = torch.optim.Adam(model_tp.parameters(), lr=1e-3)
    opt_ref = torch.optim.Adam(model_ref.parameters(), lr=1e-3)

    first = last = None
    worst = 0.0
    for step in range(steps):
        _, loss_tp = model_tp(idx, targets)
        _, loss_ref = model_ref(idx, targets)
        worst = max(worst, abs(loss_tp.item() - loss_ref.item()))
        if step == 0:
            first = loss_tp.item()
        last = loss_tp.item()
        opt_tp.zero_grad(set_to_none=True)
        opt_ref.zero_grad(set_to_none=True)
        loss_tp.backward()
        model_tp.finalize_grads()
        loss_ref.backward()
        opt_tp.step()
        opt_ref.step()

    ok = check(tp, "loss trajectory", torch.tensor(worst), torch.tensor(0.0), atol=1e-4)
    fell = last < first
    log(tp, f"  [{'PASS' if fell else 'FAIL'}] loss falls: {first:.4f} -> {last:.4f}")
    return ok and fell


def test_seq_sharding(tp, model_tp, cfg):
    """With SP on, activations entering every LayerNorm must be
    sequence-sharded: [B, T/tp, C], not [B, T, C]."""
    log(tp, "sharding: LayerNorm-region activations are [B, T/tp, C]")
    seen = []
    hooks = [
        b.ln_1.register_forward_hook(lambda m, args, out: seen.append(args[0].shape))
        for b in model_tp.blocks
    ]
    idx, targets = make_batch(cfg, torch.Generator().manual_seed(SEED + 1))
    with torch.no_grad():
        model_tp(idx, targets)
    for h in hooks:
        h.remove()
    want = (idx.shape[0], cfg.block_size // tp.world, cfg.n_embd)
    ok = all(s == torch.Size(want) for s in seen)
    log(tp, f"  [{'PASS' if ok else 'FAIL'}] {len(seen)} blocks saw {tuple(seen[0])}, want {want}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp", action="store_true", help="enable sequence parallelism")
    args = parser.parse_args()

    parallel.init_distributed()
    torch.manual_seed(SEED)  # dropout etc.; the suite runs with dropout=0
    tp = parallel.make_tp_context()

    cfg = GPTConfig(
        vocab_size=512,
        block_size=64,
        n_layer=4,
        n_head=8,
        n_embd=128,
        dropout=0.0,
        sequence_parallel=args.sp,
    )
    log(tp, f"tp={tp.world}, sp={args.sp}, fp32, cpu, config={cfg}\n")

    ok = test_conjugacy(tp)
    if args.sp:
        ok &= test_sp_conjugacy(tp)
    model_tp, model_ref = build_models(cfg, tp)
    if args.sp and tp.enabled:
        ok &= test_seq_sharding(tp, model_tp, cfg)
    ok &= test_model(tp, model_tp, model_ref, cfg)
    ok &= test_training(tp, model_tp, model_ref, cfg)

    # All ranks must agree on the verdict.
    verdict = torch.tensor(0 if ok else 1)
    dist.all_reduce(verdict)
    log(tp, "\nALL PASS" if verdict.item() == 0 else "\nFAILED")
    dist.destroy_process_group()
    sys.exit(0 if verdict.item() == 0 else 1)


if __name__ == "__main__":
    main()
