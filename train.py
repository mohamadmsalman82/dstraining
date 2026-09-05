"""Train GPT-2 with any composition of tp x pp x dp, interleaving, sequence
parallelism, selective recompute, and bf16 + fp32 master weights.

Local CPU smoke test, all axes at once:

    scripts/launch_local.sh 8 train.py --dp 2 --pp 2 --tp 2 --chunks 2 \
        --sp --recompute --bf16 --steps 20 --data synthetic

On GPUs, launch the same file with torchrun. With --data pointing at a
uint16 token shard (see scripts/prepare_openwebtext.py), loss falling is
the end-to-end proof the composition trains; convergence is not the point.
"""

import argparse
import sys
import time

import torch
import torch.distributed as dist

from dst import parallel
from dst.data import TokenShard, SyntheticShard
from dst.dp import allreduce_gradients
from dst.model import GPT, GPTConfig, GPTStage, GPTChunks
from dst.pipeline import Pipeline1F1B, PipelineInterleaved
from dst.precision import MasterWeightOptimizer


def parse_args():
    p = argparse.ArgumentParser()
    # topology
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--chunks", type=int, default=1, help="model chunks per rank (v>1: interleaved)")
    p.add_argument("--sp", action="store_true", help="sequence parallelism")
    p.add_argument("--recompute", action="store_true", help="selective attention recompute")
    p.add_argument("--bf16", action="store_true", help="bf16 params + fp32 master weights")
    # model (defaults: small; --gpt2 for the real 124M config)
    p.add_argument("--gpt2", action="store_true")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--embd", type=int, default=128)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--vocab", type=int, default=512)
    # training
    p.add_argument("--data", default="synthetic", help="'synthetic' or path to a uint16 token shard")
    p.add_argument("--batch", type=int, default=8, help="batch per DP replica")
    p.add_argument("--micro", type=int, default=1, help="micro batch size (pp only)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    return p.parse_args()


def main():
    args = parse_args()
    parallel.init_distributed()
    tp, pp, dp = parallel.make_topology(args.tp, args.pp, args.dp)
    device = torch.device(args.device) if args.device else parallel.default_device()

    if args.gpt2:
        args.layers, args.heads, args.embd, args.block, args.vocab = 12, 12, 768, 1024, 50257
    cfg = GPTConfig(
        vocab_size=args.vocab,
        block_size=args.block,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embd,
        dropout=0.0,
        sequence_parallel=args.sp,
        recompute_attention=args.recompute,
    )

    if args.chunks > 1:
        model = GPTChunks(cfg, tp, pp, v=args.chunks, seed=args.seed)
    elif args.pp > 1:
        model = GPTStage(cfg, tp, pp, seed=args.seed)
    else:
        model = GPT(cfg, tp, seed=args.seed)
    model = model.to(device)
    if args.bf16:
        model = model.to(torch.bfloat16)

    if args.chunks > 1:
        pipe = PipelineInterleaved(model, pp, micro_batch_size=args.micro, seq_len=cfg.block_size)
    elif args.pp > 1:
        pipe = Pipeline1F1B(model, pp, micro_batch_size=args.micro, seq_len=cfg.block_size)
    else:
        pipe = None

    if args.bf16:
        opt = MasterWeightOptimizer(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.data == "synthetic":
        shard = SyntheticShard(cfg.vocab_size, cfg.block_size, args.batch, seed=args.seed, dp=dp)
    else:
        shard = TokenShard(args.data, cfg.block_size, args.batch, seed=args.seed, dp=dp)

    n_params = sum(p.numel() for p in model.parameters())
    is_logger = pp.is_last and tp.rank == 0 and dp.rank == 0
    if is_logger:
        print(
            f"tp={args.tp} pp={args.pp} dp={args.dp} chunks={args.chunks} "
            f"sp={args.sp} recompute={args.recompute} bf16={args.bf16} "
            f"device={device.type} | {n_params:,} params this rank | "
            f"batch {args.batch}/replica",
            flush=True,
        )

    tokens_per_step = args.batch * cfg.block_size * args.dp
    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        idx, targets = shard.get_batch(step)
        idx, targets = idx.to(device), targets.to(device)
        if pipe is not None:
            loss = pipe.train_step(idx, targets)
        else:
            _, loss = model(idx, targets)
            loss.backward()
        model.finalize_grads()
        allreduce_gradients(model, dp)
        opt.step()

        if pp.is_last and (step % args.log_every == 0 or step == args.steps - 1):
            avg = loss.detach().clone()
            if dp.enabled:
                dist.all_reduce(avg, group=dp.group)
                avg /= dp.world
            if is_logger:
                dt = time.time() - t0
                print(
                    f"step {step:5d}  loss {avg.item():.4f}  "
                    f"{tokens_per_step * (step + 1) / dt:,.0f} tok/s",
                    flush=True,
                )

    dist.destroy_process_group()
    sys.exit(0)


if __name__ == "__main__":
    main()
