"""The conjugate communication operators for tensor parallelism.

Megatron's f / f-bar pair:

  copy_to_tp_region      (f)     forward: identity      backward: all-reduce
  reduce_from_tp_region  (f-bar) forward: all-reduce    backward: identity

A column-parallel linear applies f on its input; a row-parallel linear
applies f-bar on its output. Wrapping a block in the pair makes every
gradient correct with no other bookkeeping: the forward all-reduce sums
partial outputs across ranks, and the backward all-reduce sums the input
gradients that each rank's weight shard contributed.

gather_from_tp_region is the third op, used only where a full tensor is
genuinely needed (gathering vocab-sharded logits before the loss):
forward all-gather along the last dim, backward take the local slice.
"""

import torch
import torch.distributed as dist

from .parallel import TPContext


def _all_reduce(t: torch.Tensor, tp: TPContext) -> torch.Tensor:
    # Clone: collectives write in place, and neither a Function's input nor
    # its incoming gradient may be mutated. gloo additionally requires
    # contiguous tensors.
    t = t.contiguous().clone()
    dist.all_reduce(t, group=tp.group)
    return t


class _CopyToTPRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        ctx.tp = tp
        return x

    @staticmethod
    def backward(ctx, grad):
        return _all_reduce(grad, ctx.tp), None


class _ReduceFromTPRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        return _all_reduce(x, tp)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _GatherFromTPRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        ctx.tp = tp
        x = x.contiguous()
        parts = [torch.empty_like(x) for _ in range(tp.world)]
        dist.all_gather(parts, x, group=tp.group)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        tp = ctx.tp
        local = grad.shape[-1] // tp.world
        return grad[..., tp.rank * local : (tp.rank + 1) * local].contiguous(), None


def copy_to_tp_region(x: torch.Tensor, tp: TPContext) -> torch.Tensor:
    if not tp.enabled:
        return x
    return _CopyToTPRegion.apply(x, tp)


def reduce_from_tp_region(x: torch.Tensor, tp: TPContext) -> torch.Tensor:
    if not tp.enabled:
        return x
    return _ReduceFromTPRegion.apply(x, tp)


def gather_from_tp_region(x: torch.Tensor, tp: TPContext) -> torch.Tensor:
    if not tp.enabled:
        return x
    return _GatherFromTPRegion.apply(x, tp)
