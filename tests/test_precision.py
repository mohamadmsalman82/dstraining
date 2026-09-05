"""bf16 + fp32 master weight tests.

    scripts/launch_local.sh 1 tests/test_precision.py
    scripts/launch_local.sh 2 tests/test_precision.py        # under TP

Checks:
  1. training  — a bf16 model with MasterWeightOptimizer overfits a fixed
                 batch, landing near the fp32 trajectory
  2. stall     — the same bf16 model with PLAIN Adam (updates applied in
                 bf16) ends measurably worse: the motivating failure mode,
                 demonstrated rather than asserted from the paper
  3. invariants — params stay bf16 and bitwise-equal to master.to(bf16);
                 masters and Adam state stay fp32
  4. under TP  — bf16 collectives keep logits identical across ranks
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse

import torch
import torch.distributed as dist

from dst import parallel
from dst.model import GPT, GPTConfig
from dst.precision import MasterWeightOptimizer

SEED = 1234
STEPS = 30
DEVICE = torch.device("cpu")  # set in main; cuda under NCCL


def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def train(model, opt, idx, targets, finalize=None):
    losses = []
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        _, loss = model(idx, targets)
        loss.backward()
        if finalize:
            finalize()
        opt.step()
        losses.append(loss.item())
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    args = parser.parse_args()

    parallel.init_distributed()
    global DEVICE
    DEVICE = torch.device(args.device) if args.device else parallel.default_device()
    tp = parallel.make_tp_context()
    cfg = GPTConfig(vocab_size=512, block_size=64, n_layer=4, n_head=8, n_embd=128, dropout=0.0)
    g = torch.Generator().manual_seed(SEED + 1)
    idx = torch.randint(0, cfg.vocab_size, (4, cfg.block_size), generator=g).to(DEVICE)
    targets = torch.randint(0, cfg.vocab_size, (4, cfg.block_size), generator=g).to(DEVICE)
    log(f"tp={tp.world}, {STEPS} steps on a fixed batch\n")
    ok = True

    # fp32 baseline.
    model32 = GPT(cfg, tp, seed=SEED).to(DEVICE)
    fp32 = train(model32, torch.optim.Adam(model32.parameters(), lr=1e-3), idx, targets)

    # bf16 + fp32 masters.
    model16 = GPT(cfg, tp, seed=SEED).to(DEVICE).to(torch.bfloat16)
    opt = MasterWeightOptimizer(model16.parameters(), lr=1e-3)
    bf16 = train(model16, opt, idx, targets)

    # bf16 with updates applied in bf16 (the failure mode).
    model_bad = GPT(cfg, tp, seed=SEED).to(DEVICE).to(torch.bfloat16)
    bad = train(model_bad, torch.optim.Adam(model_bad.parameters(), lr=1e-3), idx, targets)

    fell = bf16[-1] < bf16[0] - 1.0
    ok &= fell
    log(f"  [{'PASS' if fell else 'FAIL'}] bf16+masters trains: {bf16[0]:.4f} -> {bf16[-1]:.4f}"
        f"  (fp32: {fp32[0]:.4f} -> {fp32[-1]:.4f})")

    near = abs(bf16[-1] - fp32[-1]) < 0.3
    ok &= near
    log(f"  [{'PASS' if near else 'FAIL'}] tracks fp32: final gap {abs(bf16[-1]-fp32[-1]):.4f} < 0.3")

    worse = bad[-1] > bf16[-1] + 0.05
    ok &= worse
    log(f"  [{'PASS' if worse else 'FAIL'}] plain-bf16 Adam stalls behind: "
        f"{bad[-1]:.4f} vs {bf16[-1]:.4f} with masters")

    dtypes = all(p.dtype == torch.bfloat16 for p in model16.parameters())
    dtypes &= all(m.dtype == torch.float32 for m in opt.master_params)
    dtypes &= all(
        s["exp_avg"].dtype == torch.float32
        for s in opt.inner.state.values()
    )
    synced = all(
        torch.equal(p.data, m.to(torch.bfloat16))
        for p, m in zip(opt.model_params, opt.master_params)
    )
    ok &= dtypes and synced
    log(f"  [{'PASS' if dtypes and synced else 'FAIL'}] invariants: params bf16 == "
        f"master.to(bf16), masters and Adam state fp32")

    if tp.enabled:
        logits, _ = model16(idx, targets)
        gathered = [torch.empty_like(logits) for _ in range(tp.world)]
        dist.all_gather(gathered, logits.contiguous(), group=tp.group)
        d = max((t - gathered[0]).abs().max().item() for t in gathered)
        ok &= d == 0.0
        log(f"  [{'PASS' if d == 0.0 else 'FAIL'}] cross-rank logit identity in bf16: {d:.1e}")

    verdict = torch.tensor(0 if ok else 1, device=DEVICE)
    dist.all_reduce(verdict)
    log("\nALL PASS" if verdict.item() == 0 else "\nFAILED")
    dist.destroy_process_group()
    sys.exit(0 if verdict.item() == 0 else 1)


if __name__ == "__main__":
    main()
