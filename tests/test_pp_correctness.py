"""Numerical correctness suite for 1F1B pipeline parallelism (with TP/SP
composed in).

    scripts/launch_local.sh 2 tests/test_pp_correctness.py --pp 2
    scripts/launch_local.sh 4 tests/test_pp_correctness.py --pp 2 --tp 2 --sp
    scripts/launch_local.sh 4 tests/test_pp_correctness.py --pp 4 --micro 1

Because initialization is invariant to the parallel decomposition (per-
component seeded generators + full-weight draws sliced per TP rank), every
rank builds its GPTStage AND the full single-process reference with
bit-identical weights. The suite asserts, in fp32:

  1. loss     — the pipelined, microbatched loss equals the full-batch
                reference loss on the last stage
  2. backward — every stage gradient (accumulated over microbatches)
                matches the corresponding reference gradient slice
  3. training — 5 Adam steps track the reference trajectory and the loss
                falls (verifies grads stay right as weights move)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import argparse

import torch
import torch.distributed as dist

from dst import parallel
from dst.model import GPT, GPTConfig, GPTStage
from dst.pipeline import Pipeline1F1B

from test_tp_correctness import check, ref_slice, SEED, ATOL


def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def log_last(pp, tp, msg):
    """The loss lives on the last stage; print from its tp-rank-0."""
    if pp.is_last and tp.rank == 0:
        print(msg, flush=True)


def ref_name(stage, name):
    """Map a stage-local param name to the full model's name."""
    if name.startswith("blocks."):
        parts = name.split(".")
        per_stage = stage.config.n_layer // stage.pp.world
        parts[1] = str(stage.pp.rank * per_stage + int(parts[1]))
        return ".".join(parts)
    return name


def grads_match(stage, model_ref, tp):
    ref_params = dict(model_ref.named_parameters())
    worst_name, worst = None, -1.0
    for name, p in stage.named_parameters():
        rp = ref_params[ref_name(stage, name)]
        grad_sl = ref_slice(stage, name, p, rp.grad, tp)
        d = (p.grad - grad_sl).abs().max().item()
        if d > worst:
            worst_name, worst = name, d
    return worst_name, worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--micro", type=int, default=1, help="micro batch size")
    parser.add_argument("--sp", action="store_true")
    args = parser.parse_args()

    parallel.init_distributed()
    torch.manual_seed(SEED)
    tp, pp = parallel.make_topology(args.tp, args.pp)

    cfg = GPTConfig(
        vocab_size=512,
        block_size=64,
        n_layer=4,
        n_head=8,
        n_embd=128,
        dropout=0.0,
        sequence_parallel=args.sp,
    )
    B = 4
    n_micro = B // args.micro
    log(f"tp={args.tp}, pp={args.pp}, sp={args.sp}, batch={B}, "
        f"micro={args.micro} ({n_micro} microbatches), fp32, cpu\n")

    stage = GPTStage(cfg, tp, pp, seed=SEED)
    model_ref = GPT(cfg, parallel.SINGLE, seed=SEED)
    pipe = Pipeline1F1B(stage, pp, micro_batch_size=args.micro, seq_len=cfg.block_size)

    g = torch.Generator().manual_seed(SEED + 1)
    idx = torch.randint(0, cfg.vocab_size, (B, cfg.block_size), generator=g)
    targets = torch.randint(0, cfg.vocab_size, (B, cfg.block_size), generator=g)

    ok = True

    log("loss: pipelined microbatched loss vs full-batch reference")
    stage.zero_grad(set_to_none=True)
    loss_pp = pipe.train_step(idx, targets)
    stage.finalize_grads()
    _, loss_ref = model_ref(idx, targets)
    loss_ref.backward()
    if pp.is_last:
        assert loss_pp is not None
        d = abs(loss_pp.item() - loss_ref.item())
        ok &= d <= ATOL
        log_last(pp, tp, f"  [{'PASS' if d <= ATOL else 'FAIL'}] loss: max abs diff {d:.3e}")
    else:
        assert loss_pp is None

    log("backward: accumulated stage grads vs reference slices")
    worst_name, worst = grads_match(stage, model_ref, tp)
    ok &= check(tp, f"all grads (worst: {worst_name})", torch.tensor(worst), torch.tensor(0.0))

    log("training: 5 Adam steps track the reference trajectory")
    opt = torch.optim.Adam(stage.parameters(), lr=1e-3)
    opt_ref = torch.optim.Adam(model_ref.parameters(), lr=1e-3)
    first = last = None
    worst = 0.0
    for step in range(5):
        opt.zero_grad(set_to_none=True)
        loss_pp = pipe.train_step(idx, targets)
        stage.finalize_grads()
        opt.step()

        opt_ref.zero_grad(set_to_none=True)
        _, loss_ref = model_ref(idx, targets)
        loss_ref.backward()
        opt_ref.step()

        if pp.is_last:
            worst = max(worst, abs(loss_pp.item() - loss_ref.item()))
            if step == 0:
                first = loss_pp.item()
            last = loss_pp.item()
    if pp.is_last:
        ok &= worst <= 1e-4
        log_last(pp, tp, f"  [{'PASS' if worst <= 1e-4 else 'FAIL'}] loss trajectory: max abs diff {worst:.3e}")
        fell = last < first
        ok &= fell
        log_last(pp, tp, f"  [{'PASS' if fell else 'FAIL'}] loss falls: {first:.4f} -> {last:.4f}")

    verdict = torch.tensor(0 if ok else 1)
    dist.all_reduce(verdict)
    log("\nALL PASS" if verdict.item() == 0 else "\nFAILED")
    dist.destroy_process_group()
    sys.exit(0 if verdict.item() == 0 else 1)


if __name__ == "__main__":
    main()
