"""Vocab-parallel cross-entropy.

With a column-parallel LM head, gathering logits materializes a
[b, s, padded_vocab] tensor per rank — for GPT-2 that is 50304/h ≈ 65x
the size of the hidden state, easily the largest activation in the model,
and the gather itself is the single biggest collective. Megatron's
alternative computes the loss directly on the vocab shards:

  1. row max: local max, all-reduce(MAX)            [numerical stability]
  2. sum(exp): local sum, all-reduce(SUM)           [the denominator]
  3. target logit: each target id lives on exactly one rank; that rank
     contributes it, everyone else contributes 0, all-reduce(SUM)
  4. loss = log(sumexp) - target_logit

Three all-reduces of [N] scalars replace one all-gather of [N, V_local],
and the full logit matrix never exists. The backward is local: softmax
minus the one-hot, restricted to this rank's columns.

Padded vocab columns (ids >= the real vocab size) are masked to -inf so
they contribute nothing to the denominator and get zero gradient.
Computation runs in fp32 regardless of the model dtype.
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist

from .parallel import TPContext


class _VocabParallelCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, tp, vocab_start, real_vocab):
        # logits: [N, V_local] this rank's vocab shard; targets: [N] global ids
        in_dtype = logits.dtype
        logits = logits.float()
        V_local = logits.shape[1]

        pad_from = real_vocab - vocab_start  # first padded column, locally
        if 0 <= pad_from < V_local:
            logits[:, max(pad_from, 0):] = float("-inf")

        row_max = logits.max(dim=1).values
        if tp.enabled:
            dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=tp.group)
        logits = logits - row_max[:, None]

        exp = logits.exp()  # -inf -> 0 on padded columns
        sum_exp = exp.sum(dim=1)
        if tp.enabled:
            dist.all_reduce(sum_exp, group=tp.group)

        in_range = (targets >= vocab_start) & (targets < vocab_start + V_local)
        t_local = (targets - vocab_start).clamp(0, V_local - 1)
        target_logit = logits.gather(1, t_local[:, None]).squeeze(1) * in_range
        if tp.enabled:
            dist.all_reduce(target_logit, group=tp.group)

        loss = (sum_exp.log() - target_logit).mean()

        ctx.save_for_backward(exp, sum_exp, t_local, in_range)
        ctx.in_dtype = in_dtype
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        exp, sum_exp, t_local, in_range = ctx.saved_tensors
        n = exp.shape[0]
        grad = exp / sum_exp[:, None]  # softmax over the full (global) row
        grad[torch.arange(n, device=exp.device), t_local] -= in_range.float()
        grad = grad * (grad_out / n)
        return grad.to(ctx.in_dtype), None, None, None, None


def vocab_parallel_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, tp: TPContext, real_vocab: int
) -> torch.Tensor:
    """logits: [..., V_local] vocab-sharded (full padded vocab when tp=1);
    targets: [...] global token ids. Returns the mean CE over all tokens."""
    V_local = logits.shape[-1]
    vocab_start = tp.rank * V_local
    return _VocabParallelCE.apply(
        logits.reshape(-1, V_local), targets.reshape(-1), tp, vocab_start, real_vocab
    )
