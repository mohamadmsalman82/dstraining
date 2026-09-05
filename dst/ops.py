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

Sequence parallelism replaces the pair with its sequence-sharded conjugates
(Korthikanti et al.). LayerNorm and dropout can't split along the hidden
dimension, so under plain TP every rank redundantly holds the full b*s*h
activation there; splitting those regions along the sequence axis instead
makes ALL per-layer activation memory divide by t. Since an all-reduce IS
a reduce-scatter followed by an all-gather, swapping f/f-bar for g/g-bar
moves the same bytes — the memory is free:

  gather_along_seq          (g)     forward: all-gather(seq)      backward: reduce-scatter(seq)
  reduce_scatter_along_seq  (g-bar) forward: reduce-scatter(seq)  backward: all-gather(seq)
  scatter_along_seq                 forward: take local seq slice backward: all-gather(seq)

g feeds a column-parallel linear from a sequence-sharded region; g-bar
takes a row-parallel linear's partial outputs and lands them sequence-
sharded (reduce and scatter in one collective). scatter_along_seq is the
entry point: it drops the replicated embedding output into the first
sequence-sharded region.
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


SEQ_DIM = 1  # activations are [batch, seq, hidden]

# gloo has no native reduce_scatter; probed lazily, emulated when missing.
_HAS_REDUCE_SCATTER = None


def _all_gather_seq(x: torch.Tensor, tp: TPContext) -> torch.Tensor:
    x = x.contiguous()
    parts = [torch.empty_like(x) for _ in range(tp.world)]
    dist.all_gather(parts, x, group=tp.group)
    return torch.cat(parts, dim=SEQ_DIM)


def _reduce_scatter_seq(x: torch.Tensor, tp: TPContext) -> torch.Tensor:
    global _HAS_REDUCE_SCATTER
    chunk = x.shape[SEQ_DIM] // tp.world
    if _HAS_REDUCE_SCATTER is not False:
        # reduce_scatter_tensor splits along dim 0; move seq there.
        xm = x.movedim(SEQ_DIM, 0).contiguous()
        out = torch.empty((chunk, *xm.shape[1:]), dtype=x.dtype, device=x.device)
        try:
            dist.reduce_scatter_tensor(out, xm, group=tp.group)
            _HAS_REDUCE_SCATTER = True
            return out.movedim(0, SEQ_DIM).contiguous()
        except RuntimeError:
            _HAS_REDUCE_SCATTER = False
    # Emulation: all-reduce then slice. Same result, all-gather's worth of
    # extra bandwidth — acceptable on the CPU correctness path only.
    x = _all_reduce(x, tp)
    return x.narrow(SEQ_DIM, tp.rank * chunk, chunk).contiguous()


def _local_seq_slice(x: torch.Tensor, tp: TPContext) -> torch.Tensor:
    chunk = x.shape[SEQ_DIM] // tp.world
    return x.narrow(SEQ_DIM, tp.rank * chunk, chunk).contiguous()


class _GatherAlongSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        ctx.tp = tp
        return _all_gather_seq(x, tp)

    @staticmethod
    def backward(ctx, grad):
        return _reduce_scatter_seq(grad, ctx.tp), None


class _ReduceScatterAlongSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        ctx.tp = tp
        return _reduce_scatter_seq(x, tp)

    @staticmethod
    def backward(ctx, grad):
        return _all_gather_seq(grad, ctx.tp), None


class _ScatterAlongSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        ctx.tp = tp
        return _local_seq_slice(x, tp)

    @staticmethod
    def backward(ctx, grad):
        return _all_gather_seq(grad, ctx.tp), None


def gather_along_seq(x: torch.Tensor, tp: TPContext) -> torch.Tensor:
    if not tp.enabled:
        return x
    return _GatherAlongSeq.apply(x, tp)


def reduce_scatter_along_seq(x: torch.Tensor, tp: TPContext) -> torch.Tensor:
    if not tp.enabled:
        return x
    return _ReduceScatterAlongSeq.apply(x, tp)


def scatter_along_seq(x: torch.Tensor, tp: TPContext) -> torch.Tensor:
    if not tp.enabled:
        return x
    return _ScatterAlongSeq.apply(x, tp)


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
