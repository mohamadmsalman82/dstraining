"""Data parallelism, from scratch.

Each DP replica holds a full copy of its (tp, pp)-sharded model and takes
its own slice of the global batch. After backward — and after
finalize_grads() when sequence parallelism is on; the two all-reduces
commute, both are sums — every gradient is all-reduced and averaged
across the DP group, so every replica steps identically and the weights
never diverge.

This is the whole algorithm; what DistributedDataParallel adds on top is
performance engineering (bucketing grads into flat buffers, overlapping
the all-reduces with the rest of backward via autograd hooks). That lands
with the GPU benchmarking phase; a per-parameter blocking all-reduce is
correct and is what the suites verify.
"""

import torch.distributed as dist
import torch.nn as nn

from .parallel import DPContext


def allreduce_gradients(module: nn.Module, dp: DPContext) -> None:
    if not dp.enabled:
        return
    for p in module.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, group=dp.group)
            p.grad /= dp.world
