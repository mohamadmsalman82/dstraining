"""Numerical correctness suite for data parallelism composed with
everything else: dp x tp x pp x (interleaved) x SP.

    scripts/launch_local.sh 2 tests/test_dp_correctness.py --dp 2
    scripts/launch_local.sh 4 tests/test_dp_correctness.py --dp 2 --tp 2 --sp
    scripts/launch_local.sh 4 tests/test_dp_correctness.py --dp 2 --pp 2 --micro 1
    scripts/launch_local.sh 8 tests/test_dp_correctness.py --dp 2 --pp 2 --tp 2 --chunks 2 --micro 1 --sp

Each DP replica takes its slice of the global batch; after backward the
gradients are all-reduced and averaged across the DP group. The suite
asserts, in fp32, against the full single-process model on the FULL
global batch:

  1. backward  — every gradient (post DP average) matches the reference
  2. loss      — the DP-averaged loss matches the full-batch loss
  3. training  — 5 Adam steps track the reference trajectory and fall
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import argparse

import torch
import torch.distributed as dist

from dst import parallel
from dst.dp import allreduce_gradients
from dst.model import GPT, GPTConfig, GPTStage, GPTChunks
from dst.pipeline import Pipeline1F1B, PipelineInterleaved

from test_tp_correctness import ref_slice, SEED, ATOL


def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--micro", type=int, default=1)
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--chunks", type=int, default=1)
    args = parser.parse_args()

    parallel.init_distributed()
    torch.manual_seed(SEED)
    tp, pp, dp = parallel.make_topology(args.tp, args.pp, args.dp)

    cfg = GPTConfig(
        vocab_size=512,
        block_size=64,
        n_layer=4,
        n_head=8,
        n_embd=128,
        dropout=0.0,
        sequence_parallel=args.sp,
    )
    B_global = 8
    B_local = B_global // args.dp
    log(f"dp={args.dp}, tp={args.tp}, pp={args.pp}, chunks={args.chunks}, "
        f"sp={args.sp}, global batch={B_global}, local={B_local}, fp32, cpu\n")

    model_ref = GPT(cfg, parallel.SINGLE, seed=SEED)
    pipelined = args.pp > 1
    if args.chunks > 1:
        model = GPTChunks(cfg, tp, pp, v=args.chunks, seed=SEED)
        pipe = PipelineInterleaved(model, pp, micro_batch_size=args.micro, seq_len=cfg.block_size)
    elif pipelined:
        model = GPTStage(cfg, tp, pp, seed=SEED)
        pipe = Pipeline1F1B(model, pp, micro_batch_size=args.micro, seq_len=cfg.block_size)
    else:
        model = GPT(cfg, tp, seed=SEED)
        pipe = None

    g = torch.Generator().manual_seed(SEED + 1)
    idx = torch.randint(0, cfg.vocab_size, (B_global, cfg.block_size), generator=g)
    targets = torch.randint(0, cfg.vocab_size, (B_global, cfg.block_size), generator=g)
    my_idx = idx[dp.rank * B_local : (dp.rank + 1) * B_local]
    my_tgt = targets[dp.rank * B_local : (dp.rank + 1) * B_local]

    def one_step():
        """Local forward/backward + grad finalization; returns the
        DP-averaged loss on ranks that hold it (last stage), else None."""
        model.zero_grad(set_to_none=True)
        if pipe is not None:
            loss = pipe.train_step(my_idx, my_tgt)
        else:
            _, loss = model(my_idx, my_tgt)
            loss.backward()
        model.finalize_grads()
        allreduce_gradients(model, dp)
        if pp.is_last:
            avg = loss.detach().clone()
            dist.all_reduce(avg, group=dp.group)
            return avg / dp.world
        return None

    ref_params = dict(model_ref.named_parameters())
    ok = True

    log("backward: DP-averaged grads vs full-batch reference")
    loss = one_step()
    _, loss_ref = model_ref(idx, targets)
    loss_ref.backward()

    ref_name = model.ref_param_name if hasattr(model, "ref_param_name") else lambda n: n
    worst_name, worst = None, -1.0
    for name, p in model.named_parameters():
        grad_sl = ref_slice(model, name, p, ref_params[ref_name(name)].grad, tp)
        d = (p.grad - grad_sl).abs().max().item()
        if d > worst:
            worst_name, worst = name, d
    ok &= worst <= ATOL
    log(f"  [{'PASS' if worst <= ATOL else 'FAIL'}] all grads (worst: {worst_name}): "
        f"max abs diff {worst:.3e}")

    if pp.is_last:
        d = abs(loss.item() - loss_ref.item())
        ok &= d <= ATOL
        if tp.rank == 0 and dp.rank == 0:
            print(f"  [{'PASS' if d <= ATOL else 'FAIL'}] DP-averaged loss: "
                  f"max abs diff {d:.3e}", flush=True)

    log("training: 5 Adam steps track the reference trajectory")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_ref = torch.optim.Adam(model_ref.parameters(), lr=1e-3)
    model_ref.zero_grad(set_to_none=True)
    first = last = None
    worst = 0.0
    for step in range(5):
        loss = one_step()
        opt.step()

        opt_ref.zero_grad(set_to_none=True)
        _, loss_ref = model_ref(idx, targets)
        loss_ref.backward()
        opt_ref.step()

        if pp.is_last:
            worst = max(worst, abs(loss.item() - loss_ref.item()))
            if step == 0:
                first = loss.item()
            last = loss.item()
    if pp.is_last:
        ok &= worst <= 1e-4 and last < first
        if tp.rank == 0 and dp.rank == 0:
            print(f"  [{'PASS' if worst <= 1e-4 else 'FAIL'}] loss trajectory: "
                  f"max abs diff {worst:.3e}", flush=True)
            print(f"  [{'PASS' if last < first else 'FAIL'}] loss falls: "
                  f"{first:.4f} -> {last:.4f}", flush=True)

    verdict = torch.tensor(0 if ok else 1)
    dist.all_reduce(verdict)
    log("\nALL PASS" if verdict.item() == 0 else "\nFAILED")
    dist.destroy_process_group()
    sys.exit(0 if verdict.item() == 0 else 1)


if __name__ == "__main__":
    main()
