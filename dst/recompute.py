"""Selective activation recompute.

The core-attention region — softmax(QK^T/sqrt(d)) V — holds activations
scaling as s^2 * b * a (the attention matrix, its softmax, and their
kin) while using few FLOPs. Recomputing ONLY that region in backward
drops most activation memory for a couple percent overhead, versus
roughly a third of total FLOPs for recomputing entire layers.

recompute(fn, *args) runs fn under no_grad in forward, saving just the
inputs (q, k, v — which the surrounding graph keeps alive anyway); in
backward it re-runs fn with grad enabled and backpropagates through the
fresh subgraph. The region must be deterministic and RNG-free — true of
core attention here (attention dropout would need RNG state capture).

Written against raw autograd rather than torch.utils.checkpoint: the
mechanism is the point of this milestone, and it is 30 lines.
"""

import torch


class _Recompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, *args):
        ctx.fn = fn
        ctx.save_for_backward(*args)
        with torch.no_grad():
            return fn(*args)

    @staticmethod
    def backward(ctx, grad_out):
        args = [
            a.detach().requires_grad_(a.requires_grad) for a in ctx.saved_tensors
        ]
        with torch.enable_grad():
            out = ctx.fn(*args)
        torch.autograd.backward(out, grad_out)
        return (None, *(a.grad for a in args))


def recompute(fn, *args) -> torch.Tensor:
    """fn(*args) -> Tensor, with fn's internals discarded after forward and
    rebuilt during backward. fn must be deterministic and RNG-free."""
    return _Recompute.apply(fn, *args)
