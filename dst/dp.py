"""Data parallelism, from scratch.

Each DP replica holds a full copy of its (tp, pp)-sharded model and takes
its own slice of the global batch; after backward every gradient is
all-reduced and averaged across the DP group so replicas step
identically.

Two implementations:

allreduce_gradients — the reference: one blocking all-reduce per
parameter after backward. Correct, simple, and the measured reason dp=2
scaled at 1.64x instead of 2x on 2x4090.

GradReducer — what DDP actually does, rebuilt on raw collectives:
gradients are grouped into ~25MB buckets in reverse parameter order (the
order backward produces them), each bucket is flattened and all-reduced
asynchronously the moment its last gradient lands (via
post-accumulate-grad hooks), so communication overlaps the rest of
backward. finish() waits on the outstanding work, averages, and copies
the reduced values back into param.grad.

Interplay with the other axes:
  - SP's finalize_grads() (TP-group all-reduce of partial grads) can run
    before or after the DP reduce — both are sums over disjoint groups,
    so they commute. The convention here: DP reduce first (overlapped),
    finalize_grads after finish().
  - Pipeline schedules accumulate grads over microbatches, so hooks that
    fire every backward would reduce too early. For pp > 1 construct the
    reducer with overlap=False: reduce() launches all buckets' async
    all-reduces back-to-back after train_step — the buckets still
    pipeline against each other, just not against compute.
"""

from typing import List

import torch
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


class _Bucket:
    def __init__(self, params: List[nn.Parameter]):
        self.params = params
        self.pending = 0
        self.flat = None
        self.work = None

    def launch(self, dp: DPContext) -> None:
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        self.flat = torch._utils._flatten_dense_tensors(grads)
        self.work = dist.all_reduce(self.flat, group=dp.group, async_op=True)

    def finish(self, dp: DPContext) -> None:
        self.work.wait()
        self.flat /= dp.world
        for p, g in zip(
            self.params,
            torch._utils._unflatten_dense_tensors(
                self.flat, [p.grad if p.grad is not None else p for p in self.params]
            ),
        ):
            if p.grad is not None:
                p.grad.copy_(g)
        self.flat = self.work = None


class GradReducer:
    def __init__(self, module: nn.Module, dp: DPContext, *,
                 bucket_bytes: int = 25 * 1024 * 1024, overlap: bool = True):
        self.dp = dp
        self.overlap = overlap and dp.enabled
        self.buckets: List[_Bucket] = []
        self._bucket_of = {}

        if not dp.enabled:
            return

        # Reverse parameter order approximates the order backward finishes
        # gradients, so early buckets fill (and ship) while backward is
        # still working through the front of the model.
        params = [p for p in module.parameters() if p.requires_grad][::-1]
        cur, cur_bytes = [], 0
        for p in params:
            cur.append(p)
            cur_bytes += p.numel() * p.element_size()
            if cur_bytes >= bucket_bytes:
                self.buckets.append(_Bucket(cur))
                cur, cur_bytes = [], 0
        if cur:
            self.buckets.append(_Bucket(cur))
        for b in self.buckets:
            for p in b.params:
                self._bucket_of[p] = b

        if self.overlap:
            for p in params:
                p.register_post_accumulate_grad_hook(self._hook)
            self._reset_pending()

    def _reset_pending(self) -> None:
        for b in self.buckets:
            b.pending = len(b.params)

    def _hook(self, p: nn.Parameter) -> None:
        b = self._bucket_of[p]
        b.pending -= 1
        if b.pending == 0:
            b.launch(self.dp)

    def reduce(self) -> None:
        """Non-overlapped path (pipeline schedules): launch every bucket's
        async all-reduce back to back after backward has fully finished."""
        if not self.dp.enabled:
            return
        for b in self.buckets:
            b.launch(self.dp)

    def finish(self) -> None:
        """Wait for all in-flight reductions, average, write back to
        param.grad. Call after backward (overlap mode) or after reduce()."""
        if not self.dp.enabled:
            return
        for b in self.buckets:
            if b.work is None and self.overlap:
                # Backward never reached this bucket's params (shouldn't
                # happen in training, but don't hang).
                b.launch(self.dp)
            if b.work is not None:
                b.finish(self.dp)
        if self.overlap:
            self._reset_pending()
